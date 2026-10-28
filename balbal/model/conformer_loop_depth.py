"""Six-pass CTC refinement with depth conditioning and spaced supervision."""
from __future__ import annotations

import torch
from torch import nn

from balbal.model.conformer_extra_objectives import LoopedCTCRefiner, padding_mask, positions


class DepthConditionedCTCRefiner(LoopedCTCRefiner):
    """Reuse one refiner; identify its depth and position in a two-pass cycle.

    The FiLM projection starts at zero, so common weights retain the original
    refiner's computation initially. Odd passes remain on the gradient path to
    subsequent supervised passes. Token feedback retains the original detached
    argmax; no reference tokens enter the forward computation.
    """

    def __init__(self, *args, conditioning_width=32, supervision_interval=2, **kwargs):
        super().__init__(*args, **kwargs)
        if self.loops < 2 or supervision_interval < 2 or self.loops % supervision_interval:
            raise ValueError('Loop budget must be divisible by the supervision interval')
        width = self.mix_logit.numel()
        self.supervision_interval = supervision_interval
        self.depth_film = nn.Sequential(nn.Linear(2, conditioning_width), nn.SiLU(),
                                        nn.Linear(conditioning_width, 2 * width))
        nn.init.zeros_(self.depth_film[-1].weight)
        nn.init.zeros_(self.depth_film[-1].bias)
        features = [[k / (self.loops - 1), (k % supervision_interval) / (supervision_interval - 1)]
                    for k in range(self.loops)]
        self.register_buffer('depth_features', torch.tensor(features, dtype=torch.float32), persistent=False)

    def forward(self, base_log_probs):
        if self.last_encoded is None:
            raise RuntimeError('Encoder features were not captured')
        count = self.loops if self.training else self.inference_loops
        if not 0 <= count <= self.loops:
            raise ValueError('Untrained loop count')
        memory = self.memory(self.last_encoded.transpose(1, 2))
        pad = padding_mask(self.last_lengths, memory.shape[1])
        position = positions(memory.shape[1], memory.shape[-1], memory.device, memory.dtype)
        memory = (memory + position).masked_fill(pad[..., None], 0.)
        state, probs = memory, base_log_probs.float()
        base_normalizer = probs.logsumexp(-1, keepdim=True)
        gamma, beta = self.depth_film(self.depth_features).chunk(2, dim=-1)
        passes = []
        for k in range(count):
            predicted = self.embedding(probs.detach().argmax(-1))
            recalled = self.input_norm(state + memory + predicted)
            recalled = recalled * (1 + gamma[k].to(recalled.dtype)) + beta[k].to(recalled.dtype)
            proposal = self.block(recalled, memory, tgt_key_padding_mask=pad,
                                  memory_key_padding_mask=pad)
            mixture = self.mix_logit.sigmoid()
            state = self.state_norm((1 - mixture) * state + mixture * proposal)
            state = state.masked_fill(pad[..., None], 0.)
            delta = self.output(state).float()
            normalization_change = (base_log_probs.float() + delta).logsumexp(-1, keepdim=True) - base_normalizer
            probs = base_log_probs.float() + (delta - normalization_change)
            passes.append(probs)
        if self.training:
            self.last_base, self.last_passes = base_log_probs.float(), passes
        else:
            self.clear()
        return probs


def attach_depth_refiner(model, config, *, conditioning_width=32, supervision_interval=2):
    if hasattr(model, 'ctc_extra'):
        raise ValueError('Extra module already attached')
    extra = DepthConditionedCTCRefiner(**config, conditioning_width=conditioning_width,
                                       supervision_interval=supervision_interval)
    extra = extra.to(next(model.parameters()).device)
    model.ctc_extra = extra

    def capture(_module, _args, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError('Only offline uncached encoder output is supported')
        extra.last_encoded, extra.last_lengths = output

    model._ctc_extra_encoder_hook = model.encoder.register_forward_hook(capture)
    model._ctc_extra_decoder_hook = model.decoder.register_forward_hook(lambda _m, _a, output: extra(output))
    return extra


def sparse_ctc_loss(base_probs, passes, lengths, labels, target_lengths, ctc_loss,
                    checkpoints=(2, 4, 6), base_weight=.3):
    if not checkpoints or tuple(sorted(set(checkpoints))) != tuple(checkpoints):
        raise ValueError('Supervised passes must be increasing and unique')
    if checkpoints[0] < 1 or checkpoints[-1] != len(passes):
        raise ValueError('Supervision must include the final trained pass')
    if not 0 <= base_weight <= 1:
        raise ValueError('Invalid base loss weight')

    def ctc(probs):
        return ctc_loss(log_probs=probs.float().log_softmax(-1), targets=labels,
                        input_lengths=lengths, target_lengths=target_lengths)

    main = ctc(base_probs)
    losses = [ctc(passes[k - 1]) for k in checkpoints]
    loss = base_weight * main + (1 - base_weight) * torch.stack(losses).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite sparse-refinement CTC loss')
    stats = dict(ctc=float(main.detach()), objective=float(loss.detach()))
    stats.update({f'loop_{k}_ctc': float(value.detach()) for k, value in zip(checkpoints, losses)})
    return loss, stats
