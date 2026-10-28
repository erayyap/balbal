"""Guarded Muon + AdamW, keeping the frozen v1 optimizer module unchanged.

Use the standard Muon quintic (KellerJordan/Muon and torch.optim.Muon),
with FP32 matmuls/normalization and eight iterations. Validate every proposed
Muon direction before applying ANY parameter updates. Historical checkpoint
routing and momentum conventions remain unchanged.
"""
from collections import defaultdict
import math
import torch
from training.common.muon import HybridAdamWMuon as OriginalHybrid


def polar_update(momentum):
    if momentum.ndim != 3:
        raise ValueError('Expected independent batched 2-D linear maps')
    with torch.autocast(momentum.device.type, enabled=False):
        x = momentum.float()
        # Max scaling avoids overflow/underflow in the Frobenius normalization.
        maximum = x.abs().amax(dim=(-2, -1), keepdim=True)
        x = x / maximum.clamp_min(torch.finfo(torch.float32).tiny)
        x = x / torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp_min(1e-7)
        transpose = x.shape[-2] > x.shape[-1]
        if transpose:
            x = x.mT
        for _ in range(8):
            gram = x @ x.mT
            x = 3.4445 * x + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ x
        if transpose:
            x = x.mT
    return x.to(momentum.dtype)


def direction_is_safe(x):
    # The standard polynomial's singular values stay below1.5. Both tests
    # have margin, while rejecting large finite outputs as well as NaN/Inf.
    rank = min(x.shape[-2:])
    return (torch.isfinite(x).all() & (x.abs().amax() <= 1.6)
            & (torch.linalg.vector_norm(x.float(), dim=(-2, -1)).amax() <= 1.6 * math.sqrt(rank)))


class HybridAdamWMuon(OriginalHybrid):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        checks, prepared, buffers = [], [], []
        max_direction, max_delta = [], []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    checks.append(torch.isfinite(p).all() & (p.abs().amax() < 1000)
                                  & torch.isfinite(p.grad).all())
        for group in self.param_groups[2:]:
            buckets = defaultdict(list)
            momentum = group['momentum']
            for p, splits in zip(group['params'], group['splits'], strict=True):
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse or g.is_complex():
                    raise ValueError('Muon requires dense real gradients')
                prior = self.state.get(p, {}).get('muon_momentum')
                prior = torch.zeros_like(p) if prior is None else prior
                buffer = prior.lerp(g, 1 - momentum)
                update = g.lerp(buffer, momentum)
                height, width = p.shape[0] // splits, p.shape[1]
                buckets[(height, width, p.device, p.dtype)].append((p, update.reshape(splits, height, width), splits))
                buffers.append((p, buffer))
            for (height, width, _, _), items in buckets.items():
                directions = polar_update(torch.cat([u for _, u, _ in items]))
                checks.append(direction_is_safe(directions))
                scale = .2 * math.sqrt(max(height, width))
                max_direction.append(directions.abs().amax())
                max_delta.append(directions.abs().amax() * abs(group['lr'] * scale))
                offset, params, ds = 0, [], []
                for p, _, splits in items:
                    params.append(p); ds.append(directions[offset:offset+splits].reshape_as(p));offset += splits
                prepared.append((group, scale, params, ds))
        if checks and not bool(torch.stack(checks).all()):
            # No Adam state, Muon momentum, or parameter has changed yet.
            raise FloatingPointError('Rejected nonfinite/oversized Muon update or invalid input weights/gradients BEFORE parameter mutation')
        self.last_diagnostics = dict(muon_max_direction=float(torch.stack(max_direction).max()) if max_direction else 0.,
                                     muon_max_proposed_delta=float(torch.stack(max_delta).max()) if max_delta else 0.)
        self.adam.step()
        for p, buffer in buffers:
            self.state[p]['muon_momentum'] = buffer
        for group, scale, params, ds in prepared:
            torch._foreach_mul_(params, 1 - group['lr'] * group['weight_decay'])
            torch._foreach_add_(params, ds, alpha=-group['lr'] * scale)
        return loss


def gpu_qualification():
    """Real GPU stress of representative encoder/refiner maps, not just CPU."""
    torch.manual_seed(901)
    worst, count = 0., 0
    for h,w in [(44,176),(48,192),(176,176),(192,192),(192,768)]:
        for rank in [1,3,min(h,w)]:
            x=torch.randn(16,h,rank,device='cuda')@torch.randn(16,rank,w,device='cuda')
            x*=torch.logspace(-25,25,16,device='cuda')[:,None,None];x[0]=0
            y=polar_update(x)
            if not bool(direction_is_safe(y)):
                raise FloatingPointError('GPU Muon stress qualification failed')
            worst=max(worst,float(y.abs().amax()));count+=len(x)
    return dict(status='passed',matrices=count,max_direction=worst,device=torch.cuda.get_device_name(0))
