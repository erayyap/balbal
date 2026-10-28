"""Export balbal-v2 / balbal-v2-manas to the two CPU graphs of runtime.py: 16 kHz PCM in, natural-log CTC probabilities out.

  python inference/export_onnx.py --model-dir MODEL_DIR clip1.wav clip2.wav ...        # needs MODEL_DIR/model.pt + tokenizer/
writes MODEL_DIR/onnx/{model.onnx, model.onnx.data, model-plain.onnx, model-plain.onnx.data, export.json}. The clips (any 16 kHz-readable
audio, a few seconds each) are the parity check; the middle one is the trace example.

Both graphs are "lean batch-1" graphs: at batch 1 every frame is valid except the frontend's extra last STFT frame (NeMo zeroes it and
masks it as padding), so the frontend drops that frame and every padding mask goes:
- mel frontend as a real DFT conv1d (NeMo FilterbankFeatures in eval: preemph .97, centered STFT n_fft 512 / hop 160 / 400-sample hann,
  |X|^2, slaney mel, log(x + 2^-24), per-feature mean / unbiased std over the L // 160 frames);
- Conformer layers without attention / padding masks; rel-pos attention with the positional biases in [B, H, T, d] layout and the
  Transformer-XL relative shift as one GatherElements; the subsampling stem without its time masks; the pointwise convs of the conv
  modules as MatMul (so dynamic int8 quantization covers them);
- refiner attention shape-agnostic, refiner and memory read without masks.
The only numerical change: the stem convolutions see zero padding at the very last frame instead of the zeroed extra frame (an edge
effect on the last frame's log-probabilities; frame argmax agreement is reported in export.json).

model.onnx (fast, needs inference/kernels/libbalbal_ops.so): the product-key memory read as one balbal::PkmRead node, then its value
table stored in int8 (PkmReadQ8), the rel-pos score chain fused into balbal::RelPosScore, and (--int8, default on for balbal-v2-manas)
every MatMul weight dynamically quantized to int8 (per channel).
model-plain.onnx: standard operators only (the memory read as MatMul + TopK + Gather), same --int8 choice.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import onnx_tools
from runtime import OPS_LIB, model_name, ort_session
from balbal import models as M
from balbal.audio import read_audio
from balbal.model.conformer_extra_objectives import positions

INT8_DEFAULT = {'balbal-v2': False, 'balbal-v2-manas': True}


class Frontend(nn.Module):
    def __init__(self, featurizer):
        super().__init__()
        f = featurizer
        assert (f.n_fft, f.hop_length, f.win_length, f.preemph, f.mag_power, f.normalize, f.frame_splicing, f.pad_to,
                f.log_zero_guard_type, f.exact_pad) == (512, 160, 400, .97, 2., 'per_feature', 1, 0, 'add', False)
        self.guard = float(f.log_zero_guard_value)
        window = torch.zeros(512); window[56:456] = f.window.float()   # torch.stft centers the window in n_fft
        k = torch.arange(257, dtype=torch.float64)[:, None] * torch.arange(512, dtype=torch.float64)[None] * (2 * math.pi / 512)
        basis = torch.cat([torch.cos(k), -torch.sin(k)]) * window.double()
        self.register_buffer('basis', basis.float()[:, None, :])        # [514, 1, 512]
        self.register_buffer('fb', f.fb.float()[0])                       # [80, 257]

    def forward(self, audio):                                             # [1, L]
        valid = torch._shape_as_tensor(audio)[1:] // 160                  # traced (dynamic) frame count
        x = torch.cat([audio[:, :1], audio[:, 1:] - .97 * audio[:, :-1]], 1)
        x = F.pad(x, (256, 256))[:, None]                                 # center=True, constant
        spec = F.conv1d(x, self.basis, stride=160)                        # [1, 514, T]
        power = spec[:, :257].square() + spec[:, 257:].square()
        mel = torch.log(torch.matmul(self.fb, power) + self.guard)[..., :valid[0]]   # drop the extra (padding) frame
        mean = mel.mean(-1, keepdim=True)
        std = ((mel - mean).square().sum(-1, keepdim=True) / (valid.float() - 1.)).sqrt() + 1e-5
        return (mel - mean) / std, valid.long()


class Exported(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.frontend, self.model = Frontend(model.preprocessor.featurizer), model

    def forward(self, audio):
        features, length = self.frontend(audio)
        log_probs, encoded_length, _ = self.model(processed_signal=features, processed_signal_length=length)
        return log_probs[0, :encoded_length[0]]                            # [frames, V + 1]


class Attention(nn.Module):
    """Refiner nn.MultiheadAttention without masks (its traced reshapes would freeze the example length)."""
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
            scores = scores + torch.zeros_like(key_padding_mask, dtype=scores.dtype).masked_fill(
                key_padding_mask, float('-inf'))[:, None, None, :]
        out = torch.matmul(scores.softmax(-1), v).transpose(1, 2)
        return self.mha.out_proj(out.reshape(out.shape[0], out.shape[1], -1)), None


def lean_encoder(encoder):
    """Conformer layers without masks, rel-pos attention as one GatherElements shift, unmasked stem, pointwise convs as MatMul."""
    from nemo.collections.asr.parts.submodules.subsampling import calculate_conv_output_size
    assert list(encoder.att_context_size) == [-1, -1], encoder.att_context_size
    for layer in encoder.layers:
        at, cm = layer.self_attn, layer.conv
        assert type(at).__name__ == 'RelPositionMultiHeadAttention', type(at)
        def attention(query, key, value, mask=None, pos_emb=None, cache=None, at=at):
            q, k, v = at.forward_qkv(query, key, value)                          # [B, H, T, d]
            p = at.linear_pos(pos_emb).view(pos_emb.size(0), -1, at.h, at.d_k).transpose(1, 2)   # [1, H, 2T-1, d]
            bd = torch.matmul(q + at.pos_bias_v[:, None, :], p.transpose(-2, -1))  # [B, H, T, 2T-1]
            t = torch.arange(q.shape[2], device=q.device)
            idx = (t[None, :] - t[:, None] + (q.shape[2] - 1)).expand(bd.shape[0], bd.shape[1], -1, -1)
            scores = (torch.matmul(q + at.pos_bias_u[:, None, :], k.transpose(-2, -1)) + bd.gather(-1, idx)) / at.s_d_k
            x = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).reshape(q.shape[0], -1, at.h * at.d_k)
            return at.linear_out(x)
        at.forward = attention
        assert cm.pointwise_conv1.kernel_size == (1,) and cm.pointwise_conv2.kernel_size == (1,) and cm.pointwise_activation == 'glu_'
        def conv(x, pad_mask=None, cache=None, cm=cm):
            w1, b1 = cm.pointwise_conv1.weight[..., 0], cm.pointwise_conv1.bias
            w2, b2 = cm.pointwise_conv2.weight[..., 0], cm.pointwise_conv2.bias
            x = F.glu(F.linear(x, w1, b1), dim=-1).transpose(1, 2)
            x = cm.depthwise_conv(x, cache=None)
            x = cm.batch_norm(x.transpose(1, 2)).transpose(1, 2) if cm.norm_type == 'layer_norm' else cm.batch_norm(x)
            return F.linear(cm.activation(x).transpose(1, 2), w2, b2)
        cm.forward = conv
        orig = layer.forward
        layer.forward = (lambda orig: lambda x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None:
                         orig(x, att_mask=None, pos_emb=pos_emb, pad_mask=None, cache_last_channel=cache_last_channel,
                              cache_last_time=cache_last_time))(orig)
    seq = encoder.pre_encode.conv
    assert type(seq).__name__ == 'MaskedConvSequential', type(seq)
    def stem(x, lengths, seq=seq):   # at batch 1 every time mask is all ones; lengths still follow the conv strides
        x = x.unsqueeze(1); cur = lengths.clone().float()
        for layer in seq:
            x = layer(x)
            if hasattr(layer, 'stride') and layer.stride != (1, 1):
                cur = calculate_conv_output_size(cur, layer.kernel_size[0], layer.stride[0],
                                                 (layer._left_padding, layer._right_padding) if hasattr(layer, '_left_padding') else layer.padding)
        return x, cur.long()
    seq.forward = stem


class PkmReadFn(torch.autograd.Function):
    """One balbal::PkmRead node (inference/kernels/pkm_read.cc); the torch forward is the same read (for the parity check)."""
    @staticmethod
    def forward(ctx, query, keys, values, inv_temp, top_k):
        S = keys.shape[2]
        q = query.permute(2, 3, 0, 1, 4).flatten(2, 3)                     # [H, 2, T, D]
        top, ids = torch.matmul(q, keys.transpose(-1, -2)).topk(top_k, dim=-1, sorted=False)
        scores, pairs = (top[:, 0, :, :, None] + top[:, 1, :, None, :]).flatten(-2).topk(top_k, dim=-1, sorted=False)
        indices = ids[:, 0].gather(-1, pairs // top_k) * S + ids[:, 1].gather(-1, pairs % top_k)
        weights = (scores * inv_temp.reshape(-1, 1, 1)).softmax(-1)
        return (F.embedding(indices, values) * weights[..., None]).sum(-2).sum(0)[None]

    @staticmethod
    def symbolic(g, query, keys, values, inv_temp, top_k):
        return g.op('balbal::PkmRead', query, keys, values, inv_temp, top_k_i=top_k)


def lean_memory(mem, custom):
    """Memory read without masks: custom = one PkmRead node, else MatMul + unsorted TopK + Gather (standard operators)."""
    t = mem.temperature
    inv = (1. / t).float().reshape(-1) if torch.is_tensor(t) else torch.full((mem.heads,), 1. / t)
    inv = inv.expand(mem.heads).contiguous() if inv.numel() == 1 else inv
    def retrieve(query):                                                   # query [1, T, H, 2, D]
        keys = F.normalize(mem.keys.float(), dim=-1, eps=1e-6)             # [H, 2, S, D]
        if custom: return PkmReadFn.apply(query, keys, mem.values.weight, inv, mem.top_k)
        q = query.permute(2, 3, 0, 1, 4).flatten(2, 3)                     # [H, 2, T, D]
        top, ids = torch.matmul(q, keys.transpose(-1, -2)).topk(mem.top_k, dim=-1, sorted=False)
        scores, pairs = (top[:, 0, :, :, None] + top[:, 1, :, None, :]).flatten(-2).topk(mem.top_k, dim=-1, sorted=False)
        indices = ids[:, 0].gather(-1, pairs // mem.top_k) * mem.subkeys + ids[:, 1].gather(-1, pairs % mem.top_k)
        weights = (scores * inv.reshape(-1, 1, 1)).softmax(-1)
        return (F.embedding(indices, mem.values.weight) * weights[..., None]).sum(-2).sum(0)[None]
    def forward(hidden, _log_probs, valid, dropout_draw=None):
        length = hidden.shape[1]
        context = torch.cat([F.pad(hidden, (0, 0, d, 0))[:, :length] for d in reversed(range(mem.context_frames))], -1)
        norm = mem.query_norm[mem.pass_index]
        raw = (mem.query(context).float() - norm.running_mean) * torch.rsqrt(norm.running_var + norm.eps)
        query = F.normalize(raw.reshape(*hidden.shape[:2], mem.heads, 2, mem.key_dim), dim=-1, eps=1e-6)
        memory = retrieve(query)
        key = F.layer_norm(mem.key(memory).float(), (mem.width,))
        h = F.layer_norm(hidden.float(), (mem.width,))
        gate = torch.sigmoid((h * key).sum(-1) / math.sqrt(mem.width) + mem.gate_bias)
        return mem.value(memory) * gate[..., None]
    mem._forward_pre_hooks.clear()   # the learned temperature is folded into inv
    mem.forward = forward


def lean_refiner(ex):
    for layer in ex.block.layers:
        layer.self_attn, layer.multihead_attn = Attention(layer.self_attn), Attention(layer.multihead_attn)
    def forward(base_log_probs):
        memory = ex.memory(ex.last_encoded.transpose(1, 2))
        memory = memory + positions(memory.shape[1], memory.shape[-1], memory.device, memory.dtype)
        state, probs = memory, base_log_probs.float()
        base_normalizer = probs.logsumexp(-1, keepdim=True)
        for k in range(ex.inference_loops):
            recalled = ex.input_norm(state + memory + ex.embedding(probs.argmax(-1)))
            if ex.memory_enabled:
                ex.acoustic_memory.pass_index = k
                recalled = recalled + ex.memory_strength * ex.acoustic_memory(recalled, probs, None).to(recalled.dtype)
            proposal = ex.block(recalled, memory)
            mixture = ex.mix_logit.sigmoid()
            state = ex.state_norm((1 - mixture) * state + mixture * proposal)
            delta = ex.output(state).float()
            probs = base_log_probs.float() + (delta - ((base_log_probs.float() + delta).logsumexp(-1, keepdim=True) - base_normalizer))
        ex.clear()
        return probs
    ex.forward = forward


def lean_model(name, model_dir, custom):
    model = M.build_model(name, Path(model_dir) / 'tokenizer', Path(model_dir) / 'model.pt')
    mem = model.ctc_extra.acoustic_memory
    if hasattr(mem, 'log_temp'): mem.temperature = mem.log_temp.detach().exp()
    wrapper = Exported(model).eval()
    lean_encoder(model.encoder); lean_memory(mem, custom); lean_refiner(model.ctc_extra)
    return wrapper


def parity(f, waves, reference):
    diff, same, frames = 0., 0, 0
    for w, r in zip(waves, reference):
        x = f(w); assert x.shape == r.shape, (x.shape, r.shape)
        diff = max(diff, float(np.abs(x - r).max())); same += int((x.argmax(-1) == r.argmax(-1)).sum()); frames += len(r)
    return dict(max_abs_logprob_diff=round(diff, 5), frame_argmax_agreement=round(same / frames, 5))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('clips', type=Path, nargs='+', help='audio files for the parity check (the middle one is the trace example)')
    p.add_argument('--model-dir', type=Path, default=os.getenv('BALBAL_MODEL_DIR'), required=not os.getenv('BALBAL_MODEL_DIR'))
    p.add_argument('--model', choices=M.MODELS, default=None, help='default: MODEL_DIR/model.json')
    p.add_argument('--out', type=Path, default=None, help='default: MODEL_DIR/onnx')
    p.add_argument('--int8', choices=('auto', 'on', 'off'), default='auto', help='int8 MatMul weights (auto: on for balbal-v2-manas)')
    p.add_argument('--only', choices=('fast', 'plain'), default=None, help='export one of the two graphs')
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    name = model_name(args.model_dir, args.model)
    int8 = INT8_DEFAULT[name] if args.int8 == 'auto' else args.int8 == 'on'
    out = args.out or args.model_dir / 'onnx'; out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(args.threads)
    waves = [read_audio(c) for c in args.clips]
    model = M.build_model(name, args.model_dir / 'tokenizer', args.model_dir / 'model.pt')
    with torch.inference_mode():
        reference = []
        for w in waves:
            lp, n, _ = model(input_signal=torch.from_numpy(w)[None], input_signal_length=torch.tensor([len(w)]))
            reference.append(lp[0, :int(n[0])].float().numpy())
    del model
    result = dict(model=name, int8_matmul=int8, clips=len(waves), frames=sum(len(r) for r in reference))
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    for kind in ('fast', 'plain'):
        if args.only and kind != args.only: continue
        t0 = time.time(); custom = kind == 'fast'
        wrapper = lean_model(name, args.model_dir, custom)
        with torch.inference_mode():
            res = dict(torch_lean=parity(lambda w: wrapper(torch.from_numpy(w)[None]).numpy(), waves, reference))
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            graph = Path(tmp) / 'lean.onnx'
            torch.onnx.export(wrapper, (torch.from_numpy(waves[len(waves) // 2])[None],), str(graph), input_names=['audio'],
                              output_names=['log_probs'], dynamic_axes={'audio': {1: 'samples'}, 'log_probs': {0: 'frames'}},
                              opset_version=17, dynamo=False, custom_opsets={'balbal': 1} if custom else None)
            del wrapper
            if int8:
                quantize_dynamic(graph, Path(tmp) / 'int8.onnx', op_types_to_quantize=['MatMul'], per_channel=True, weight_type=QuantType.QInt8)
                graph = Path(tmp) / 'int8.onnx'
            m = onnx.load(str(graph))
            if custom:
                res['pkm_int8'] = onnx_tools.quantize_pkm(m)
                res['relpos'] = onnx_tools.fuse_relpos(m)
            target = out / ('model.onnx' if custom else 'model-plain.onnx')
            res['files'] = onnx_tools.externalize(m, target); del m
        session = ort_session(target, args.threads, custom_ops=custom)
        res['onnxruntime'] = parity(lambda w: session.run(None, {'audio': w[None]})[0], waves, reference)
        res['export_seconds'] = round(time.time() - t0, 1); result[kind] = res
        print(json.dumps({kind: res}), flush=True)
    (out / 'export.json').write_text(json.dumps(result, indent=1) + '\n')
    (args.model_dir / 'model.json').exists() or (args.model_dir / 'model.json').write_text(json.dumps(dict(model=name)) + '\n')


if __name__ == '__main__':
    main()
