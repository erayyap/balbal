"""Multi-head acoustic PKM with a shared value table and masked query statistics.

Each head has its own query projection and two subkey banks. Head outputs sum,
as in product-key memory; increasing heads does not duplicate the value table.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAcousticMemory(nn.Module):
    def __init__(self, width=176, subkeys=500, key_dim=64, value_dim=120,
                 context_frames=3, top_k=16, heads=4, temperature=.1, dropout=.1):
        super().__init__()
        if not (heads >= 1 and 1 <= top_k <= subkeys and context_frames >= 1
                and key_dim >= 1 and value_dim >= 1 and temperature > 0 and 0 <= dropout < 1):
            raise ValueError('Invalid multi-head acoustic memory configuration')
        self.width, self.subkeys, self.key_dim = width, subkeys, key_dim
        self.heads, self.top_k, self.value_dim = heads, top_k, value_dim
        self.context_frames, self.temperature, self.dropout = context_frames, temperature, dropout
        self.query = nn.Linear(width * context_frames, heads * 2 * key_dim, bias=False)
        self.query_norm = nn.BatchNorm1d(heads * 2 * key_dim, eps=1e-5, momentum=.1, affine=False)
        self.keys = nn.Parameter(torch.randn(heads, 2, subkeys, key_dim))
        self.values = nn.Embedding(subkeys * subkeys, value_dim)
        nn.init.normal_(self.values.weight, std=.02)
        self.key = nn.Linear(value_dim, width, bias=False)
        self.value = nn.Linear(value_dim, width, bias=False)
        nn.init.zeros_(self.value.weight)
        self.gate_bias = nn.Parameter(torch.tensor(-2.))
        self.register_buffer('entries_queried', torch.zeros(subkeys * subkeys, dtype=torch.bool))
        self.register_buffer('head_entries_queried', torch.zeros(heads, subkeys * subkeys, dtype=torch.bool))
        self.last_diagnostics = {}

    def queries(self, hidden, valid):
        hidden = hidden.masked_fill(~valid[..., None], 0.)
        length = hidden.shape[1]
        context = torch.cat([F.pad(hidden, (0, 0, delay, 0))[:, :length]
                             for delay in reversed(range(self.context_frames))], -1)
        raw = self.query(context).float()
        selected = raw[valid]
        normalized = torch.zeros_like(raw)
        if selected.shape[0]:
            if self.training and selected.shape[0] > 1:
                selected = self.query_norm(selected)
            else:
                selected = F.batch_norm(selected, self.query_norm.running_mean,
                    self.query_norm.running_var, training=False, eps=self.query_norm.eps)
            normalized[valid] = selected
        return F.normalize(normalized.reshape(*hidden.shape[:2], self.heads, 2, self.key_dim),
                           dim=-1, eps=1e-6)

    def retrieve(self, query):
        with torch.autocast(device_type=query.device.type, enabled=False):
            keys = F.normalize(self.keys.float(), dim=-1, eps=1e-6)
            left = torch.einsum('...hd,hsd->...hs', query[..., 0, :].float(), keys[:, 0])
            right = torch.einsum('...hd,hsd->...hs', query[..., 1, :].float(), keys[:, 1])
            left_scores, left_ids = left.topk(self.top_k, dim=-1)
            right_scores, right_ids = right.topk(self.top_k, dim=-1)
            # Exact top-k is contained in these k*k pairs. Recover addresses
            # after selection, avoiding an equally large int64 candidate tensor.
            candidates = (left_scores[..., :, None] + right_scores[..., None, :]).flatten(-2)
            scores, pairs = candidates.topk(self.top_k, dim=-1)
            indices = (left_ids.gather(-1, pairs // self.top_k) * self.subkeys
                       + right_ids.gather(-1, pairs % self.top_k))
            weights = (scores / self.temperature).softmax(-1)
            # Fused weighted gather/reduction avoids materializing B*T*H*K*V
            # activations. Both the query weights and shared values get gradients.
            memory = F.embedding_bag(indices.reshape(-1, self.top_k), self.values.weight,
                mode='sum', per_sample_weights=weights.reshape(-1, self.top_k))
            memory = memory.reshape(*indices.shape[:-1], self.value_dim).sum(-2)
        return memory, indices, weights

    def forward(self, normalized_hidden, _log_probs, valid, dropout_draw=None):
        memory, indices, weights = self.retrieve(self.queries(normalized_hidden, valid))
        key = F.layer_norm(self.key(memory).float(), (self.width,))
        hidden = F.layer_norm(normalized_hidden.float(), (self.width,))
        gate = torch.sigmoid((hidden * key).sum(-1) / math.sqrt(self.width) + self.gate_bias) * valid
        if self.training and self.dropout:
            keep = (torch.rand((gate.shape[0], 1), device=gate.device)
                    if dropout_draw is None else dropout_draw).ge(self.dropout)
            gate = gate * keep / (1 - self.dropout)
        output = self.value(memory) * gate[..., None]
        if not self.training:   # diagnostics are training-only; the refiners clear them after an eval forward
            return output
        with torch.no_grad():
            self.entries_queried[indices[valid].flatten()] = True
            for h in range(self.heads):
                self.head_entries_queried[h, indices[..., h, :][valid].flatten()] = True
            denominator = valid.sum().clamp_min(1)
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
            top = weights.max(-1).values
            self.last_diagnostics = {
                'memory_gate': gate.sum() / denominator,
                'memory_context_fraction': valid.sum() / denominator,
                'memory_top_weight': (top * valid[..., None]).sum() / (denominator * self.heads),
                'memory_retrieval_entropy': (entropy * valid[..., None]).sum() / (denominator * self.heads),
                'memory_entries_queried': self.entries_queried.sum(),
                'memory_query_coverage': self.entries_queried.float().mean(),
                'memory_heads': self.heads,
                'memory_reads_per_frame': self.heads * self.top_k,
                'memory_output_rms': (output.float().square().sum() / (denominator * self.width)).sqrt(),
            }
            for h in range(self.heads):
                self.last_diagnostics[f'memory_head_{h}_coverage'] = self.head_entries_queried[h].float().mean()
                self.last_diagnostics[f'memory_head_{h}_top_weight'] = (top[..., h] * valid).sum() / denominator
        return output
