"""Byte-exact training fast path for the loop-memory Conformer (same values as the unpatched model, fewer GPU syncs).

The unpatched training step is host-bound: ~200 GPU syncs and thousands of tiny launches per update. Everything here
computes the same values with fewer syncs/launches; verified bitwise against the original (unpatched) code:
- install(model): memory forward without diagnostics (80 float syncs per update in the original loss), the
  queried-entry bookkeeping without boolean-index nonzero syncs, and one shared valid-frame index per
  refiner forward instead of two boolean-mask nonzeros per pass.
- HybridAdamWMuon: the guarded Muon step with foreach guard reductions and momentum lerps.
"""
from __future__ import annotations

import math
import types
from collections import defaultdict

import torch
from torch.nn import functional as F

from training.common import muon_safe as safe
from balbal.model.conformer_extra_objectives import padding_mask, positions


def memory_queries(self, hidden, valid):
    """RecurrentAcousticMemory.queries with the shared valid-frame index instead of boolean masks."""
    index = self.valid_index
    hidden = hidden.masked_fill(~valid[..., None], 0.)
    length = hidden.shape[1]
    context = torch.cat([F.pad(hidden, (0, 0, delay, 0))[:, :length]
                         for delay in reversed(range(self.context_frames))], -1)
    raw = self.query(context).float()
    selected = raw.flatten(0, 1).index_select(0, index)
    normalized = torch.zeros_like(raw)
    norm = self.query_norm[self.pass_index]
    if selected.shape[0]:
        if self.training and selected.shape[0] > 1:
            selected = norm(selected)
        else:
            selected = F.batch_norm(selected, norm.running_mean, norm.running_var,
                                     training=False, eps=norm.eps)
        normalized = normalized.flatten(0, 1).index_copy(0, index, selected).view_as(raw)
    return F.normalize(normalized.reshape(*hidden.shape[:2], self.heads, 2, self.key_dim),
                       dim=-1, eps=1e-6)


def memory_forward(self, normalized_hidden, _log_probs, valid, dropout_draw=None):
    """MultiHeadAcousticMemory.forward without diagnostics; sync-free queried-entry bookkeeping."""
    memory, indices, weights = self.retrieve(self.queries(normalized_hidden, valid))
    key = F.layer_norm(self.key(memory).float(), (self.width,))
    hidden = F.layer_norm(normalized_hidden.float(), (self.width,))
    gate = torch.sigmoid((hidden * key).sum(-1) / math.sqrt(self.width) + self.gate_bias) * valid
    if self.training and self.dropout:
        keep = (torch.rand((gate.shape[0], 1), device=gate.device)
                if dropout_draw is None else dropout_draw).ge(self.dropout)
        gate = gate * keep / (1 - self.dropout)
    output = self.value(memory) * gate[..., None]
    if self.training:
        with torch.no_grad():
            # Every write stores True, so duplicate indices are harmless; padded frames go to a spill slot.
            entries = self.subkeys * self.subkeys
            flat = indices + torch.arange(self.heads, device=indices.device)[:, None] * entries
            flat = flat.masked_fill(~valid[..., None, None], self.heads * entries)
            hits = torch.zeros(self.heads * entries + 1, dtype=torch.bool, device=indices.device)
            hits[flat.reshape(-1)] = True
            hits = hits[:-1].view(self.heads, entries)
            self.head_entries_queried |= hits
            self.entries_queried |= hits.any(0)
    self.last_diagnostics = {}
    return output


def refiner_forward(self, base_log_probs):
    """MemoryLoopRefiner.forward without diagnostics."""
    if self.last_encoded is None:
        raise RuntimeError('Encoder features were not captured')
    memory = self.memory(self.last_encoded.transpose(1, 2))
    pad = padding_mask(self.last_lengths, memory.shape[1])
    memory = (memory + positions(memory.shape[1], memory.shape[-1],
                                 memory.device, memory.dtype)).masked_fill(pad[..., None], 0.)
    state, probs = memory, base_log_probs.float()
    base_normalizer = probs.logsumexp(-1, keepdim=True)
    count = self.loops if self.training else self.inference_loops
    if not 0 <= count <= self.loops:
        raise ValueError('Untrained loop count')
    valid = ~pad
    self.acoustic_memory.valid_index = valid.reshape(-1).nonzero().squeeze(1)
    dropout_draw = torch.rand((len(state), 1), device=state.device) if self.training else None
    passes = []
    for k in range(count):
        predicted = self.embedding(probs.detach().argmax(-1))
        recalled = self.input_norm(state + memory + predicted)
        if self.memory_enabled:
            self.acoustic_memory.pass_index = k
            retrieved = self.acoustic_memory(recalled, probs.detach(), valid, dropout_draw)
            recalled = recalled + self.memory_strength * retrieved.to(recalled.dtype)
        proposal = self.block(recalled, memory, tgt_key_padding_mask=pad,
                              memory_key_padding_mask=pad)
        mixture = self.mix_logit.sigmoid()
        state = self.state_norm((1 - mixture) * state + mixture * proposal)
        state = state.masked_fill(pad[..., None], 0.)
        delta = self.output(state).float()
        normalization_change = (base_log_probs.float() + delta).logsumexp(-1, keepdim=True) - base_normalizer
        probs = base_log_probs.float() + (delta - normalization_change)
        passes.append(probs)
    self.acoustic_memory.valid_index = None
    if self.training:
        self.last_base, self.last_passes = base_log_probs.float(), passes
        self.last_diagnostics = {}
    else:
        self.clear()
    return probs


def install(model):
    """Patch the model's loop-memory refiner in place (instance-level, the classes stay untouched)."""
    from balbal.model.conformer_loop_acoustic import RecurrentAcousticMemory
    from balbal.model.conformer_loop_memory import MemoryLoopRefiner
    extra = model.ctc_extra
    if type(extra) is not MemoryLoopRefiner or type(extra.acoustic_memory) is not RecurrentAcousticMemory:
        raise TypeError('fast path only covers MemoryLoopRefiner + RecurrentAcousticMemory')
    extra.forward = types.MethodType(refiner_forward, extra)
    extra.acoustic_memory.forward = types.MethodType(memory_forward, extra.acoustic_memory)
    extra.acoustic_memory.queries = types.MethodType(memory_queries, extra.acoustic_memory)
    extra.acoustic_memory.valid_index = None
    return model


class HybridAdamWMuon(safe.HybridAdamWMuon):
    """conformer_muon_safe.HybridAdamWMuon.step with foreach guards/lerps and one host sync."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        params = [p for group in self.param_groups for p in group['params'] if p.grad is not None]
        checks = []
        if params:
            # amax/inf-norm propagate NaN: |p|max < 1000 is False for NaN/Inf weights, |g|max finite iff all finite.
            weights = torch.stack(torch._foreach_norm(params, math.inf))
            grads = torch.stack(torch._foreach_norm([p.grad for p in params], math.inf))
            checks.append((weights < 1000).all() & torch.isfinite(grads).all())
        prepared, buffers, max_direction, max_delta = [], [], [], []
        for group in self.param_groups[2:]:
            momentum, live = group['momentum'], []
            for p, splits in zip(group['params'], group['splits'], strict=True):
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse or g.is_complex():
                    raise ValueError('Muon requires dense real gradients')
                prior = self.state.get(p, {}).get('muon_momentum')
                live.append((p, splits, torch.zeros_like(p) if prior is None else prior))
            if not live:
                continue
            grads = [p.grad for p, _, _ in live]
            new = torch._foreach_lerp([prior for _, _, prior in live], grads, 1 - momentum)
            updates = torch._foreach_lerp(grads, new, momentum)
            buckets = defaultdict(list)
            for (p, splits, _), buffer, update in zip(live, new, updates):
                height, width = p.shape[0] // splits, p.shape[1]
                buckets[(height, width, p.device, p.dtype)].append((p, update.reshape(splits, height, width), splits))
                buffers.append((p, buffer))
            for (height, width, _, _), items in buckets.items():
                directions = safe.polar_update(torch.cat([u for _, u, _ in items]))
                checks.append(safe.direction_is_safe(directions))
                scale = .2 * math.sqrt(max(height, width))
                top = directions.abs().amax()
                max_direction.append(top)
                max_delta.append(top * abs(group['lr'] * scale))
                offset, ps, ds = 0, [], []
                for p, _, splits in items:
                    ps.append(p); ds.append(directions[offset:offset + splits].reshape_as(p)); offset += splits
                prepared.append((group, scale, ps, ds))
        if checks:
            # One sync for the guard and both diagnostics.
            host = torch.stack([torch.stack(checks).all().float(),
                                torch.stack(max_direction).max() if max_direction else checks[0].new_zeros((), dtype=torch.float32),
                                torch.stack(max_delta).max() if max_delta else checks[0].new_zeros((), dtype=torch.float32)]).tolist()
            if not host[0]:
                raise FloatingPointError('Rejected nonfinite/oversized Muon update or invalid input weights/gradients BEFORE parameter mutation')
            self.last_diagnostics = dict(muon_max_direction=host[1] if max_direction else 0.,
                                         muon_max_proposed_delta=host[2] if max_delta else 0.)
        else:
            self.last_diagnostics = dict(muon_max_direction=0., muon_max_proposed_delta=0.)
        self.adam.step()
        for p, buffer in buffers:
            self.state[p]['muon_momentum'] = buffer
        for group, scale, ps, ds in prepared:
            torch._foreach_mul_(ps, 1 - group['lr'] * group['weight_decay'])
            torch._foreach_add_(ps, ds, alpha=-group['lr'] * scale)
        return loss
