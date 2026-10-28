"""Multi-resolution acoustic recall and soft feedback inside a shared CTC loop.

The encoder runs once. A small 20 ms branch consumes the same augmented mel
features as the 40 ms encoder. Earlier encoder layers remain differentiable.
One product-key value table is queried afresh at each pass; query running
statistics are separate per pass so refinement depths do not mix their scales.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from balbal.model.conformer_acoustic_memory_multihead import MultiHeadAcousticMemory
from balbal.model.conformer_extra_objectives import padding_mask
from balbal.model.conformer_loop_depth import DepthConditionedCTCRefiner


class RecurrentAcousticMemory(MultiHeadAcousticMemory):
    def __init__(self, *, loops, **kwargs):
        super().__init__(**kwargs)
        self.query_norm = nn.ModuleList([
            nn.BatchNorm1d(self.heads * 2 * self.key_dim, eps=1e-5,
                           momentum=.1, affine=False) for _ in range(loops)])
        self.pass_index = 0

    def queries(self, hidden, valid):
        hidden = hidden.masked_fill(~valid[..., None], 0.)
        length = hidden.shape[1]
        context = torch.cat([F.pad(hidden, (0, 0, delay, 0))[:, :length]
                             for delay in reversed(range(self.context_frames))], -1)
        raw = self.query(context).float()
        selected = raw[valid]
        normalized = torch.zeros_like(raw)
        norm = self.query_norm[self.pass_index]
        if selected.shape[0]:
            if self.training and selected.shape[0] > 1:
                selected = norm(selected)
            else:
                selected = F.batch_norm(selected, norm.running_mean, norm.running_var,
                                         training=False, eps=norm.eps)
            normalized[valid] = selected
        return F.normalize(normalized.reshape(*hidden.shape[:2], self.heads, 2, self.key_dim),
                           dim=-1, eps=1e-6)


def timed_positions(size, width, device, dtype, stride):
    # Time is expressed in 40 ms encoder frames, including for the 20 ms bank.
    times = torch.arange(size, device=device, dtype=torch.float32)[:, None] * stride
    frequency = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32)
                          * (-math.log(10000.) / width))
    result = torch.empty(size, width, device=device)
    result[:, 0::2], result[:, 1::2] = (times * frequency).sin(), (times * frequency).cos()
    return result.to(dtype)


class AcousticLoopRefiner(DepthConditionedCTCRefiner):
    def __init__(self, *, mel_bins=80, capture_layers=(3, 7), memory_config=None, **kwargs):
        super().__init__(**kwargs)
        width = self.mix_logit.numel()
        acoustic_width = self.memory[0].normalized_shape[0]
        self.capture_layers = tuple(capture_layers)
        if len(set(self.capture_layers)) != len(self.capture_layers):
            raise ValueError('Duplicate feature layer')
        self.earlier = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(acoustic_width), nn.Linear(acoustic_width, width),
                          nn.LayerNorm(width)) for _ in self.capture_layers])
        self.fine_stem = nn.Conv1d(mel_bins, width, 5, stride=2, padding=2)
        self.fine_depthwise = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.fine_pointwise = nn.Conv1d(width, width, 1)
        self.fine_norm = nn.LayerNorm(width)
        self.bank_embeddings = nn.Parameter(torch.zeros(len(self.capture_layers) + 2, width))
        self.confidence = nn.Linear(3, width, bias=False)
        nn.init.zeros_(self.confidence.weight)
        config = dict(subkeys=500, key_dim=64, value_dim=120, context_frames=3,
                      heads=4, top_k=16, temperature=.1, dropout=.1)
        config.update(memory_config or {})
        self.acoustic_memory = RecurrentAcousticMemory(width=width, loops=self.loops, **config)
        self.register_buffer('memory_strength', torch.tensor(0.))
        self.memory_enabled = True
        self.use_auxiliary_banks = True
        self.soft_feedback_enabled = True
        self.last_features = self.last_feature_lengths = None
        self.last_intermediate = {}
        self.last_diagnostics = {}

    def set_step(self, step):
        self.memory_strength.fill_(min(1., max(0., (step - 128) / 384)))

    def clear(self):
        super().clear()
        self.last_features = self.last_feature_lengths = None
        self.last_intermediate = {}
        self.last_diagnostics = {}
        self.acoustic_memory.last_diagnostics = {}

    def soft_feedback(self, log_probs):
        # Only prediction feedback is detached; continuous loop states, acoustic
        # banks and retrieved vectors retain their gradients through all passes.
        probabilities = log_probs.detach().float().softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        features = torch.stack((probabilities[..., -1], probabilities.max(-1).values,
                                entropy / math.log(probabilities.shape[-1])), -1)
        expected = F.linear(probabilities.to(self.embedding.weight.dtype), self.embedding.weight.T)
        return expected + self.confidence(features.to(self.confidence.weight.dtype)), features

    def acoustic_banks(self):
        if self.last_encoded is None or self.last_features is None:
            raise RuntimeError('Encoder and augmented input features must both be captured')
        anchor = self.memory(self.last_encoded.transpose(1, 2))
        pad = padding_mask(self.last_lengths, anchor.shape[1])
        pos = timed_positions(anchor.shape[1], anchor.shape[-1], anchor.device, anchor.dtype, 1.)
        anchor = (anchor + pos).masked_fill(pad[..., None], 0.)
        if not self.use_auxiliary_banks:
            return anchor, anchor, pad, pad

        features = self.last_features
        feature_pad = padding_mask(self.last_feature_lengths, features.shape[-1])
        fine = F.gelu(self.fine_stem(features.masked_fill(feature_pad[:, None, :], 0.)))
        fine_pad = padding_mask((self.last_feature_lengths + 1) // 2, fine.shape[-1])
        fine = fine.masked_fill(fine_pad[:, None, :], 0.)
        fine = fine + self.fine_pointwise(F.gelu(self.fine_depthwise(fine)))
        fine = self.fine_norm(fine.transpose(1, 2))
        fine = fine + timed_positions(fine.shape[1], fine.shape[-1], fine.device, fine.dtype, .5)
        banks, pads = [fine], [fine_pad]
        for layer, projection in zip(self.capture_layers, self.earlier):
            hidden = self.last_intermediate[layer]
            if hidden.shape[:2] != anchor.shape[:2]:
                raise ValueError('Intermediate encoder time grids differ')
            banks.append(projection(hidden) + pos)
            pads.append(pad)
        banks.append(anchor)
        pads.append(pad)
        banks = [(bank + tag).masked_fill(mask[..., None], 0.)
                 for bank, tag, mask in zip(banks, self.bank_embeddings, pads)]
        return anchor, torch.cat(banks, 1), pad, torch.cat(pads, 1)

    def forward(self, base_log_probs):
        count = self.loops if self.training else self.inference_loops
        if not 0 <= count <= self.loops:
            raise ValueError('Untrained loop count')
        anchor, banks, pad, bank_pad = self.acoustic_banks()
        state, probs = anchor, base_log_probs.float()
        normalizer = probs.logsumexp(-1, keepdim=True)
        gamma, beta = self.depth_film(self.depth_features).chunk(2, -1)
        passes, diagnostic_passes = [], []
        dropout_draw = torch.rand((len(state), 1), device=state.device) if self.training else None
        valid, denominator = ~pad, (~pad).sum().clamp_min(1)
        for k in range(count):
            predicted, uncertainty = self.soft_feedback(probs)
            if not self.soft_feedback_enabled:
                predicted = self.embedding(probs.detach().argmax(-1))
            recalled = self.input_norm(state + anchor + predicted)
            recalled = recalled * (1 + gamma[k].to(recalled.dtype)) + beta[k].to(recalled.dtype)
            self.acoustic_memory.pass_index = k
            if self.memory_enabled:
                retrieved = self.acoustic_memory(recalled, probs.detach(), valid, dropout_draw)
                memory_change = self.memory_strength * retrieved.to(recalled.dtype)
                recalled = recalled + memory_change
                with torch.no_grad():
                    diagnostics = dict(self.acoustic_memory.last_diagnostics)
                    diagnostics['memory_relative_rms'] = (
                        memory_change.float().square().sum() /
                        recalled.float().masked_fill(pad[..., None], 0.).square().sum().clamp_min(1e-8)).sqrt()
                    diagnostic_passes.append(diagnostics)
            proposal = self.block(recalled, banks, tgt_key_padding_mask=pad,
                                  memory_key_padding_mask=bank_pad)
            mixture = self.mix_logit.sigmoid()
            state = self.state_norm((1 - mixture) * state + mixture * proposal)
            state = state.masked_fill(pad[..., None], 0.)
            delta = self.output(state).float()
            normalization_change = (base_log_probs.float() + delta).logsumexp(-1, keepdim=True) - normalizer
            probs = base_log_probs.float() + (delta - normalization_change)
            passes.append(probs)
        if self.training:
            self.last_base, self.last_passes = base_log_probs.float(), passes
            with torch.no_grad():
                self.last_diagnostics = {
                    'feedback_entropy': (uncertainty[..., 2] * valid).sum() / denominator,
                    'feedback_blank_probability': (uncertainty[..., 0] * valid).sum() / denominator,
                    'memory_strength': self.memory_strength.clone(),
                    'acoustic_bank_frames': banks.shape[1], 'query_frames': state.shape[1],
                }
                if diagnostic_passes:
                    for key in diagnostic_passes[0]:
                        self.last_diagnostics[key] = sum(d[key] for d in diagnostic_passes) / count
                    for k, diagnostics in enumerate(diagnostic_passes):
                        self.last_diagnostics[f'pass_{k + 1}_memory_relative_rms'] = diagnostics['memory_relative_rms']
        else:
            self.clear()
        return probs


def attach_acoustic_loop(model, config, *, capture_layers=(3, 7), memory_config=None,
                         conditioning_width=32, supervision_interval=2):
    if hasattr(model, 'ctc_extra'):
        raise ValueError('Extra module already attached')
    if model.encoder._feat_in != 80 or model.encoder.subsampling_factor != 4:
        raise ValueError('This trial requires the pinned 80-bin, 4x encoder')
    extra = AcousticLoopRefiner(**config, capture_layers=capture_layers,
        memory_config=memory_config, conditioning_width=conditioning_width,
        supervision_interval=supervision_interval).to(next(model.parameters()).device)
    model.ctc_extra = extra

    def capture_input(_module, args, kwargs):
        if kwargs.get('bypass_pre_encode', False) or kwargs.get('cache_last_channel') is not None:
            raise ValueError('Only offline uncached inference is supported')
        extra.clear()
        extra.last_features = kwargs.get('audio_signal', args[0] if args else None)
        extra.last_feature_lengths = kwargs.get('length', args[1] if len(args) > 1 else None)

    handles = [model.encoder.register_forward_pre_hook(capture_input, with_kwargs=True)]
    for index in capture_layers:
        if not 0 <= index < len(model.encoder.layers) - 1:
            raise ValueError('Invalid intermediate feature layer')
        def capture_layer(_module, _args, output, index=index):
            if not torch.is_tensor(output):
                raise ValueError('Only offline uncached layer outputs are supported')
            extra.last_intermediate[index] = output
        handles.append(model.encoder.layers[index].register_forward_hook(capture_layer))

    def capture_encoder(_module, _args, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError('Only offline uncached encoder output is supported')
        extra.last_encoded, extra.last_lengths = output

    handles.append(model.encoder.register_forward_hook(capture_encoder))
    handles.append(model.decoder.register_forward_hook(lambda _m, _a, output: extra(output)))
    model._ctc_extra_hooks = handles
    return extra
