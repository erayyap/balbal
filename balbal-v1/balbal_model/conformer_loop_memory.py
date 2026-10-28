"""Original three-pass CTC refiner with shared acoustic product-key memory.

Keep the original acoustic bank, hard predicted-token feedback and per-pass
supervision. The only inference addition is gated 4x16 acoustic retrieval.
Two-view consistency remains a training objective in the experiment runner.
"""
from __future__ import annotations

import torch

from balbal_model.conformer_extra_objectives import LoopedCTCRefiner, padding_mask, positions
from balbal_model.conformer_loop_acoustic import RecurrentAcousticMemory


class MemoryLoopRefiner(LoopedCTCRefiner):
    def __init__(self, *, memory_config=None, **kwargs):
        super().__init__(**kwargs)
        if self.loops != 3:
            raise ValueError('This experiment uses the original three-pass loop')
        config = dict(subkeys=500, key_dim=64, value_dim=120, context_frames=3,
                      heads=4, top_k=16, temperature=.1, dropout=.1)
        config.update(memory_config or {})
        self.acoustic_memory = RecurrentAcousticMemory(
            width=self.mix_logit.numel(), loops=self.loops, **config)
        self.register_buffer('memory_strength', torch.tensor(0.))
        self.memory_enabled = True
        self.last_diagnostics = {}

    def set_step(self, step):
        self.memory_strength.fill_(min(1., max(0., (step - 128) / 384)))

    def clear(self):
        super().clear()
        self.last_diagnostics = {}
        self.acoustic_memory.last_diagnostics = {}

    def forward(self, base_log_probs):
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
        # Independent per utterance/view, shared across recurrent passes.
        dropout_draw = torch.rand((len(state), 1), device=state.device) if self.training else None
        passes, diagnostics = [], []
        for k in range(count):
            predicted = self.embedding(probs.detach().argmax(-1))
            recalled = self.input_norm(state + memory + predicted)
            if self.memory_enabled:
                self.acoustic_memory.pass_index = k
                retrieved = self.acoustic_memory(recalled, probs.detach(), valid, dropout_draw)
                change = self.memory_strength * retrieved.to(recalled.dtype)
                if self.training:
                    with torch.no_grad():
                        stats = dict(self.acoustic_memory.last_diagnostics)
                        stats['memory_relative_rms'] = (
                            change.float().square().sum() /
                            recalled.float().masked_fill(pad[..., None], 0.).square().sum().clamp_min(1e-8)).sqrt()
                        diagnostics.append(stats)
                recalled = recalled + change
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
            self.last_diagnostics = {'memory_strength': self.memory_strength.detach().clone()}
            if diagnostics:
                for key in diagnostics[0]:
                    self.last_diagnostics[key] = sum(d[key] for d in diagnostics) / count
                for k, stats in enumerate(diagnostics, 1):
                    self.last_diagnostics[f'pass_{k}_memory_relative_rms'] = stats['memory_relative_rms']
        else:
            self.clear()
        return probs


def attach_loop_memory(model, config, *, memory_config=None):
    if hasattr(model, 'ctc_extra'):
        raise ValueError('Extra module already attached')
    extra = MemoryLoopRefiner(**config, memory_config=memory_config).to(next(model.parameters()).device)
    model.ctc_extra = extra

    def capture(_module, _args, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError('Only offline uncached encoder output is supported')
        extra.last_encoded, extra.last_lengths = output

    model._ctc_extra_encoder_hook = model.encoder.register_forward_hook(capture)
    model._ctc_extra_decoder_hook = model.decoder.register_forward_hook(lambda _m, _a, output: extra(output))
    return extra
