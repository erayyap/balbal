"""Training-only AED and a weight-shared, audio-conditioned CTC refiner.

The refiner borrows recurrent-depth design principles, not LLM weights: fixed
per-depth supervision, normalized input recall, bounded state mixing, and no
learned halting gate. All frame positions remain available, including blanks.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def padding_mask(lengths, size):
    return torch.arange(size, device=lengths.device)[None, :] >= lengths[:, None]


def positions(size, width, device, dtype):
    time = torch.arange(size, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32)
                          * (-math.log(10000.) / width))
    result = torch.empty(size, width, device=device)
    result[:, 0::2], result[:, 1::2] = (time * frequency).sin(), (time * frequency).cos()
    return result.to(dtype)


class AuxiliaryAttentionDecoder(nn.Module):
    """Teacher forcing only in TRAIN; no reference tokens in model inference."""
    def __init__(self, acoustic_width=176, width=192, vocabulary=1024, layers=2,
                 heads=4, dropout=.1):
        super().__init__()
        self.vocabulary = vocabulary
        self.bos, self.eos, self.pad = vocabulary, vocabulary + 1, vocabulary + 2
        self.memory = nn.Sequential(nn.LayerNorm(acoustic_width), nn.Linear(acoustic_width, width))
        self.embedding = nn.Embedding(vocabulary + 3, width, padding_idx=self.pad)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            width, heads, width * 4, dropout, activation='gelu', batch_first=True,
            norm_first=True), layers, norm=nn.LayerNorm(width))
        self.output = nn.Linear(width, vocabulary + 3)
        self.last_encoded = self.last_lengths = None

    def clear(self):
        self.last_encoded = self.last_lengths = None

    def logits(self, encoded, lengths, tokens):
        hidden = self.embedding(tokens)
        hidden = hidden + positions(hidden.shape[1], hidden.shape[-1], hidden.device, hidden.dtype)
        causal = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool,
                            device=tokens.device).triu(1)
        output = self.decoder(hidden, self.memory(encoded.transpose(1, 2)), tgt_mask=causal,
                              tgt_key_padding_mask=tokens.eq(self.pad),
                              memory_key_padding_mask=padding_mask(lengths, encoded.shape[-1]))
        return self.output(output).float()

    def loss(self, labels, target_lengths):
        if self.last_encoded is None: raise RuntimeError('Encoder features were not captured')
        size = labels.shape[1] + 1
        inputs = labels.new_full((len(labels), size), self.pad)
        targets = labels.new_full((len(labels), size), self.pad)
        inputs[:, 0] = self.bos
        valid = ~padding_mask(target_lengths, labels.shape[1])
        inputs[:, 1:] = labels.masked_fill(~valid, self.pad)
        targets[:, :-1] = labels.masked_fill(~valid, self.pad)
        targets.scatter_(1, target_lengths[:, None], self.eos)
        logits = self.logits(self.last_encoded, self.last_lengths, inputs)
        # Match NeMo CTC mean_batch: sum over target positions, mean over clips.
        return F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=self.pad,
                               reduction='none').sum(1).mean()


class LoopedCTCRefiner(nn.Module):
    """One shared two-layer block, reused at every recurrent depth.

    The expensive Conformer encoder runs once. Normalized acoustic memory is
    available at every loop. Continuous state is carried between loops; token
    feedback is predicted, detached, and includes blank frames. A zero output
    projection initially preserves the acoustic model's predictions.
    """
    def __init__(self, acoustic_width=176, width=192, vocabulary=1024, layers=2,
                 heads=4, dropout=.1, loops=3):
        super().__init__()
        self.loops = self.inference_loops = loops
        self.memory = nn.Sequential(nn.LayerNorm(acoustic_width), nn.Linear(acoustic_width, width),
                                    nn.LayerNorm(width))
        self.embedding = nn.Embedding(vocabulary + 1, width)
        self.input_norm = nn.LayerNorm(width)
        self.state_norm = nn.LayerNorm(width)
        self.block = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            width, heads, width * 4, dropout, activation='gelu', batch_first=True,
            norm_first=True), layers, norm=nn.LayerNorm(width))
        self.mix_logit = nn.Parameter(torch.zeros(width))
        self.output = nn.Linear(width, vocabulary + 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.last_encoded = self.last_lengths = self.last_base = None
        self.last_passes = []

    def clear(self):
        self.last_encoded = self.last_lengths = self.last_base = None
        self.last_passes = []

    def forward(self, base_log_probs):
        if self.last_encoded is None: raise RuntimeError('Encoder features were not captured')
        memory = self.memory(self.last_encoded.transpose(1, 2))
        pad = padding_mask(self.last_lengths, memory.shape[1])
        position = positions(memory.shape[1], memory.shape[-1], memory.device, memory.dtype)
        memory = (memory + position).masked_fill(pad[..., None], 0.)
        state = memory
        probs = base_log_probs.float()
        base_normalizer = probs.logsumexp(-1, keepdim=True)
        passes = []
        count = self.loops if self.training else self.inference_loops
        if not 0 <= count <= self.loops: raise ValueError('Untrained loop count')
        for _ in range(count):
            predicted = self.embedding(probs.detach().argmax(-1))
            recalled = self.input_norm(state + memory + predicted)
            proposal = self.block(recalled, memory, tgt_key_padding_mask=pad,
                                  memory_key_padding_mask=pad)
            mixture = self.mix_logit.sigmoid()
            state = self.state_norm((1 - mixture) * state + mixture * proposal)
            state = state.masked_fill(pad[..., None], 0.)
            # Anchor each correction to the same acoustic evidence, avoiding
            # repeatedly adding previous logit corrections as independent evidence.
            delta = self.output(state).float()
            normalization_change = (base_log_probs.float() + delta).logsumexp(-1, keepdim=True) - base_normalizer
            probs = base_log_probs.float() + (delta - normalization_change)
            passes.append(probs)
        if self.training:
            self.last_base, self.last_passes = base_log_probs.float(), passes
        else:
            self.clear()
        return probs


def attach_extra(model, kind, config):
    if hasattr(model, 'ctc_extra'): raise ValueError('Extra module already attached')
    cls = AuxiliaryAttentionDecoder if kind == 'aed' else LoopedCTCRefiner
    extra = cls(**config).to(next(model.parameters()).device)
    model.ctc_extra = extra

    def capture(_module, _args, output):
        if kind == 'aed' and not model.training: return
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError('Only offline uncached encoder output is supported')
        extra.last_encoded, extra.last_lengths = output

    model._ctc_extra_encoder_hook = model.encoder.register_forward_hook(capture)
    if kind == 'loop_refine':
        def refine(_module, _args, output):
            return extra(output)
        model._ctc_extra_decoder_hook = model.decoder.register_forward_hook(refine)
    return extra


def clear_extra(model):
    if hasattr(model, 'ctc_extra'): model.ctc_extra.clear()
