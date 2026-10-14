"""Export balbal-v1 (NeMo Conformer-CTC + 3-pass loop-memory refiner) to one ONNX graph: 16 kHz PCM in, natural-log
CTC probabilities out, so runtime_onnx.py needs neither NeMo nor torch.

Batch 1, all frames valid (the serving path). Export-only substitutions, each checked against the NeMo model here:
- mel frontend: NeMo FilterbankFeatures (eval: no dither, preemph .97, centered constant-pad STFT n_fft 512, 400-sample
  symmetric hann, |X|^2, slaney mel, log(x + 2^-24), per-feature mean/unbiased-std + 1e-5 over the first L//160 frames,
  the extra last frame zeroed) as a real DFT conv1d;
- refiner nn.MultiheadAttention (its traced reshapes freeze the example length) as a shape-agnostic attention with
  the same weights and key padding masks (the frontend's extra last STFT frame is padding, as in NeMo);
- product-key memory read: F.embedding_bag(sum, per_sample_weights) as a gather + weighted sum; eval batch-norm of queries
  written out, padded frames zeroed with masked_fill instead of boolean-mask indexing.
Writes OUT/model.onnx, OUT/export.json (parity numbers) and, unless --no-externalize, OUT/model-ext.onnx +
OUT/model-ext.data (tensors >= 64 KiB moved to a 64 KiB-aligned file that onnxruntime memory-maps).
  python export_onnx.py --model-dir MODEL_DIR --out MODEL_DIR clip1.wav clip2.wav ...   # 16 kHz parity clips
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from transcribe import build_model, read_audio


class Frontend(nn.Module):
    def __init__(self, featurizer):
        super().__init__()
        f = featurizer
        assert (f.n_fft, f.hop_length, f.win_length, f.preemph, f.mag_power, f.normalize, f.frame_splicing, f.pad_to,
                f.log_zero_guard_type, f.exact_pad) == (512, 160, 400, .97, 2., 'per_feature', 1, 0, 'add', False)
        self.guard = float(f.log_zero_guard_value)
        window = torch.zeros(512); window[56:456] = f.window.float()   # torch.stft centers the window in n_fft
        k = torch.arange(257, dtype=torch.float64)[:, None] * torch.arange(512, dtype=torch.float64)[None] * (2*math.pi/512)
        basis = torch.cat([torch.cos(k), -torch.sin(k)]) * window.double()
        self.register_buffer('basis', basis.float()[:, None, :])        # [514, 1, 512]
        self.register_buffer('fb', f.fb.float()[0])                       # [80, 257]

    def forward(self, audio):                                             # [1, L]
        valid = torch._shape_as_tensor(audio)[1:] // 160                  # [1], traced (dynamic) frame count
        x = torch.cat([audio[:, :1], audio[:, 1:] - .97*audio[:, :-1]], 1)
        x = F.pad(x, (256, 256))[:, None]                                 # center=True, constant
        spec = F.conv1d(x, self.basis, stride=160)                        # [1, 514, T]
        power = spec[:, :257].square() + spec[:, 257:].square()
        mel = torch.log(torch.matmul(self.fb, power) + self.guard)        # [1, 80, T]
        mask = (torch.arange(mel.shape[-1]) < valid)[None, None]
        count = valid.float()
        mean = mel.masked_fill(~mask, 0.).sum(-1, keepdim=True) / count
        std = ((mel - mean).masked_fill(~mask, 0.).square().sum(-1, keepdim=True) / (count - 1.)).sqrt() + 1e-5
        return ((mel - mean) / std).masked_fill(~mask, 0.), valid.long()


def patch_memory(memory):
    def queries(hidden, valid):
        hidden = hidden.masked_fill(~valid[..., None], 0.)
        length = hidden.shape[1]
        context = torch.cat([F.pad(hidden, (0, 0, d, 0))[:, :length] for d in reversed(range(memory.context_frames))], -1)
        norm = memory.query_norm[memory.pass_index]
        raw = (memory.query(context).float() - norm.running_mean) * torch.rsqrt(norm.running_var + norm.eps)
        raw = raw.masked_fill(~valid[..., None], 0.)
        return F.normalize(raw.reshape(*hidden.shape[:2], memory.heads, 2, memory.key_dim), dim=-1, eps=1e-6)

    def retrieve(query):
        keys = F.normalize(memory.keys.float(), dim=-1, eps=1e-6)
        left = torch.einsum('...hd,hsd->...hs', query[..., 0, :], keys[:, 0])
        right = torch.einsum('...hd,hsd->...hs', query[..., 1, :], keys[:, 1])
        ls, li = left.topk(memory.top_k, dim=-1)
        rs, ri = right.topk(memory.top_k, dim=-1)
        scores, pairs = (ls[..., :, None] + rs[..., None, :]).flatten(-2).topk(memory.top_k, dim=-1)
        indices = li.gather(-1, pairs // memory.top_k) * memory.subkeys + ri.gather(-1, pairs % memory.top_k)
        weights = (scores / memory.temperature).softmax(-1)
        values = F.embedding(indices, memory.values.weight)               # [1, T, H, K, V]
        return (values * weights[..., None]).sum(-2).sum(-2), indices, weights

    memory.queries, memory.retrieve = queries, retrieve


class Attention(nn.Module):
    def __init__(self, mha):
        super().__init__()
        assert mha.batch_first and mha._qkv_same_embed_dim and mha.in_proj_bias is not None
        self.heads, self.mha, self.batch_first = mha.num_heads, mha, True

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False, is_causal=False):
        assert attn_mask is None and not is_causal
        w, b, h = self.mha.in_proj_weight, self.mha.in_proj_bias, self.heads
        q, k, v = (F.linear(x, wi, bi) for x, wi, bi in zip((query, key, value), w.chunk(3), b.chunk(3)))
        split = lambda x: x.reshape(x.shape[0], x.shape[1], h, -1).transpose(1, 2)    # [B, H, T, D]
        q, k, v = split(q), split(k), split(v)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -.5)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
        out = torch.matmul(scores.softmax(-1), v).transpose(1, 2)
        return self.mha.out_proj(out.reshape(out.shape[0], out.shape[1], -1)), None


def patch_attention(block):
    for layer in block.layers:
        layer.self_attn, layer.multihead_attn = Attention(layer.self_attn), Attention(layer.multihead_attn)


class Exported(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.frontend, self.model = Frontend(model.preprocessor.featurizer), model

    def forward(self, audio):
        features, length = self.frontend(audio)
        log_probs, encoded_length, _ = self.model(processed_signal=features, processed_signal_length=length)
        return log_probs[0, :encoded_length[0]]                            # [frames, 1025]


def externalize(path, align=1 << 16, minimum=1 << 16):
    """MODEL.onnx -> MODEL-ext.onnx + MODEL-ext.data; returns the number of tensors moved."""
    import onnx
    from onnx.external_data_helper import set_external_data
    model = onnx.load(path); stem = path.stem + '-ext'; moved = 0
    with open(path.with_name(stem + '.data'), 'wb') as data:
        for t in model.graph.initializer:
            if not t.HasField('raw_data') or len(t.raw_data) < minimum:
                continue
            data.seek(-data.tell() % align, 1); offset = data.tell(); data.write(t.raw_data)
            set_external_data(t, location=stem + '.data', offset=offset, length=len(t.raw_data))
            t.ClearField('raw_data'); moved += 1
    onnx.save_model(model, path.with_name(stem + '.onnx'))
    return moved


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('clips', type=Path, nargs='+', help='audio files for the parity check (the middle one is the trace example)')
    p.add_argument('--model-dir', type=Path, default=Path(os.getenv('BALBAL_MODEL_DIR', ROOT.parent / 'models')))
    p.add_argument('--out', type=Path, default=None, help='default: MODEL_DIR')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--no-externalize', action='store_true')
    args = p.parse_args()
    out = args.out or args.model_dir
    out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(args.threads)
    model = build_model(args.model_dir).float().eval()
    waves = [read_audio(c) for c in args.clips]
    with torch.inference_mode():
        reference = []
        for w in waves:
            lp, n, _ = model(input_signal=torch.from_numpy(w)[None], input_signal_length=torch.tensor([len(w)]))
            reference.append(lp[0, :int(n[0])].numpy())
        model.ctc_extra.clear()
    wrapper = Exported(model).eval()
    feature_diff = 0.
    with torch.inference_mode():
        for w in waves:
            a, _ = model.preprocessor(input_signal=torch.from_numpy(w)[None], length=torch.tensor([len(w)]))
            b, _ = wrapper.frontend(torch.from_numpy(w)[None])
            feature_diff = max(feature_diff, float((a - b).abs().max()))
    patch_memory(model.ctc_extra.acoustic_memory); patch_attention(model.ctc_extra.block)
    diffs, agree = [], 0
    with torch.inference_mode():
        for w, r in zip(waves, reference):
            x = wrapper(torch.from_numpy(w)[None]).numpy()
            assert x.shape == r.shape, (x.shape, r.shape)
            diffs.append(float(np.abs(x - r).max())); agree += int((x.argmax(-1) == r.argmax(-1)).all())
    started = time.time()
    example = torch.from_numpy(waves[len(waves)//2])[None]
    target = out / 'model.onnx'
    torch.onnx.export(wrapper, (example,), str(target), input_names=['audio'], output_names=['log_probs'],
                      dynamic_axes={'audio': {1: 'samples'}, 'log_probs': {0: 'frames'}}, opset_version=17, dynamo=False)
    result = dict(source_weights_sha256=hashlib.sha256((args.model_dir / 'conformer.pt').read_bytes()).hexdigest(),
                  onnx_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), onnx_bytes=target.stat().st_size,
                  export_seconds=round(time.time() - started, 1), frontend_max_abs_diff=feature_diff,
                  torch_patched_max_abs_logprob_diff=max(diffs), torch_patched_argmax_identical_clips=agree,
                  clips=len(waves))
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
        result['onnxruntime_max_abs_logprob_diff'] = max(
            float(np.abs(session.run(None, {'audio': w[None]})[0] - r).max()) for w, r in zip(waves, reference))
    except ImportError:
        result['onnxruntime_max_abs_logprob_diff'] = None   # onnxruntime not installed in this environment
    if not args.no_externalize:
        result['externalized_tensors'] = externalize(target)
    (out / 'export.json').write_text(json.dumps(result, indent=1)); print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
