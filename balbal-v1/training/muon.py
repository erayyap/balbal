"""Muon for semantic linear maps; AdamW for the rest of the Conformer.

Polar Express coefficient schedule: NoahAmsel/PolarExpress (MIT),
optimal_composition(l=1e-3, num_iters=10, degree=5,
                    safety_factor_eps=1e-2, cushion=0.02), first eight steps.
https://github.com/NoahAmsel/PolarExpress/blob/main/polar_express.py
Muon momentum/RMS scaling follow torch.optim.Muon / Moonshot's formulation.
"""
from __future__ import annotations

from collections import defaultdict
import math

import torch
from torch import nn


POLAR_COEFFICIENTS = (
    (8.237312490495555, -23.1577474145582, 16.680568411445915),
    (4.082441999064829, -2.8930477353325843, 0.5252849256975644),
    (3.9263479922546485, -2.854746803476524, 0.5318022422894979),
    (3.2982187133085197, -2.4245419810267106, 0.4863200835884415),
    (2.2970369434552587, -1.6366255812590325, 0.4002628455953631),
    (1.8763805351440381, -1.23478965777222, 0.3589188750166826),
    (1.856442348565278, -1.2132449881003775, 0.3568003487859341),
    (1.85643356156706, -1.2132336029471178, 0.3567976320793584),
)


def polar_update(momentum):
    """Batched independent polar factors; zero/rank-deficient inputs stay finite."""
    if momentum.ndim != 3:
        raise ValueError('Expected batch of 2-D linear maps')
    transpose = momentum.shape[-2] > momentum.shape[-1]
    # Float32 normalization avoids underflow before conversion to tensor-core BF16.
    x = momentum.float()
    x = x / (torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    if x.device.type == 'cuda':
        x = x.bfloat16()
    if transpose:
        x = x.mT
    for a, b, c in POLAR_COEFFICIENTS:
        gram = x @ x.mT
        polynomial = b * gram + c * (gram @ gram)
        x = a * x + polynomial @ x
    return (x.mT if transpose else x).to(momentum.dtype)


def parameter_policy(model):
    """Explicit routing; never infer Muon eligibility from tensor rank alone."""
    rows = []
    for name, parameter in model.named_parameters():
        module_name, _, leaf = name.rpartition('.')
        module = model.get_submodule(module_name)
        optimizer, splits, reason = 'adamw', 1, 'nonlinear/table/vector/convolution'
        if name == 'ctc_extra.acoustic_memory.values.weight':
            optimizer, reason = 'memory_adamw', 'lookup values: preserve 5x LR and zero decay'
        elif isinstance(module, nn.MultiheadAttention) and leaf == 'in_proj_weight':
            optimizer, splits, reason = 'muon', 3 * module.num_heads, 'Q/K/V separately, per head'
        elif isinstance(module, nn.Linear) and leaf == 'weight':
            if name.startswith('decoder.') or name == 'ctc_extra.output.weight':
                reason = 'CTC class output head'
            elif 'gate' in name or 'router' in name:
                reason = 'gating/routing projection'
            elif min(parameter.shape) * 16 < max(parameter.shape):
                reason = 'highly elongated projection: retain AdamW'
            else:
                optimizer, reason = 'muon', 'hidden linear projection'
                if '.self_attn.linear_' in name and leaf == 'weight':
                    projection = module_name.rsplit('.', 1)[-1]
                    if projection in ('linear_q', 'linear_k', 'linear_v', 'linear_pos'):
                        attention = model.get_submodule(module_name.rsplit('.', 1)[0])
                        splits, reason = attention.h, 'attention projection, per head'
                elif name == 'ctc_extra.acoustic_memory.query.weight':
                    memory = model.ctc_extra.acoustic_memory
                    splits, reason = memory.heads * 2, 'independent memory heads and two subkeys'
        if optimizer == 'muon':
            assert parameter.ndim == 2 and parameter.shape[0] % splits == 0, name
        rows.append(dict(name=name, shape=list(parameter.shape), elements=parameter.numel(),
                         optimizer=optimizer, splits=splits, reason=reason))
    assert len({id(p) for p in model.parameters()}) == len(rows)
    assert all(p.requires_grad for p in model.parameters())
    return rows


class HybridAdamWMuon(torch.optim.Optimizer):
    """One scheduler/checkpoint state, with fused AdamW and batched Muon updates."""
    def __init__(self, model, lr=3e-4, memory_lr=1.5e-3, weight_decay=.001):
        self.policy = parameter_policy(model)
        named = dict(model.named_parameters())
        groups = []
        for kind, rate, decay in [('adamw', lr, weight_decay),
                                  ('memory_adamw', memory_lr, 0.),
                                  ('muon', lr, weight_decay)]:
            rows = [r for r in self.policy if r['optimizer'] == kind]
            groups.append(dict(params=[named[r['name']] for r in rows],
                param_names=[r['name'] for r in rows], splits=[r['splits'] for r in rows],
                lr=rate, weight_decay=decay, group_name=kind, momentum=.95))
        super().__init__(groups, dict(lr=lr, weight_decay=weight_decay))
        self.adam = torch.optim.AdamW(self.param_groups[:2], lr=lr, betas=(.9, .98),
                                     fused=next(model.parameters()).device.type == 'cuda')
        self.state = self.adam.state  # Includes the Muon momentum buffers too.
        assert all(a is b for a, b in zip(self.param_groups, self.adam.param_groups))

    def load_state_dict(self, state_dict):
        # Reordering tensors of equal shape must not silently swap their histories.
        for actual, saved in zip(self.param_groups, state_dict['param_groups'], strict=True):
            for key in ('group_name', 'param_names', 'splits'):
                if actual[key] != saved[key]:
                    raise ValueError(f'Optimizer parameter policy changed: {key}')
        super().load_state_dict(state_dict)
        self.adam.param_groups = self.param_groups[:2]
        self.adam.state = self.state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.adam.step()
        for group in self.param_groups[2:]:
            buckets = defaultdict(list)
            momentum = group['momentum']
            for parameter, splits in zip(group['params'], group['splits'], strict=True):
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse or gradient.is_complex():
                    raise ValueError('Muon requires dense real gradients')
                state = self.state[parameter]
                if 'muon_momentum' not in state:
                    state['muon_momentum'] = torch.zeros_like(parameter)
                buffer = state['muon_momentum']
                buffer.lerp_(gradient, 1 - momentum)
                update = gradient.lerp(buffer, momentum)
                height, width = parameter.shape[0] // splits, parameter.shape[1]
                buckets[(height, width, parameter.device, parameter.dtype)].append(
                    (parameter, update.reshape(splits, height, width), splits))
            for (height, width, _, _), items in buckets.items():
                updates = polar_update(torch.cat([u for _, u, _ in items]))
                # Match AdamW update RMS; scale from the TRUE operator shape.
                scale = .2 * math.sqrt(max(height, width))
                offset, params, directions = 0, [], []
                for parameter, _, splits in items:
                    params.append(parameter)
                    directions.append(updates[offset:offset + splits].reshape_as(parameter))
                    offset += splits
                torch._foreach_mul_(params, 1 - group['lr'] * group['weight_decay'])
                torch._foreach_add_(params, directions, alpha=-group['lr'] * scale)
        return loss


def optimizer_for(model, meta):
    return HybridAdamWMuon(model, lr=meta['lr'], memory_lr=meta['memory_embedding_lr'])
