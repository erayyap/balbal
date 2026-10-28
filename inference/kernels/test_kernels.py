#!/usr/bin/env python3
"""Unit test of the custom operators in libbalbal_ops.so against numpy references, on random inputs:
  PkmRead      vs the PKM read in float64 (max abs difference, should be ~1e-6)
  PkmReadQ8    vs the same reference on the de-quantized int8 table (kernel exactness) and on the float32 table (quantization error)
  RelPosScore  vs onnxruntime's own GatherElements + Add + Div chain (must be bit-identical)
  python inference/kernels/test_kernels.py"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

LIB = str(Path(__file__).resolve().parent / 'libbalbal_ops.so')
F, I8 = TensorProto.FLOAT, TensorProto.INT8


def ref(q, keys, values, inv_t, K):
    T, H, _, D = q.shape[1:]; S = keys.shape[2]; out = np.zeros((T, values.shape[1]), np.float64)
    for t in range(T):
        for h in range(H):
            l = keys[h, 0] @ q[0, t, h, 0]; r = keys[h, 1] @ q[0, t, h, 1]
            li = np.argsort(-l)[:K]; ri = np.argsort(-r)[:K]
            c = (l[li][:, None] + r[ri][None]).ravel(); ci = np.argsort(-c)[:K]
            w = np.exp((c[ci] - c[ci].max()) * inv_t[h]); w /= w.sum()
            out[t] += (w[:, None] * values[li[ci // K] * S + ri[ci % K]]).sum(0)
    return out[None]


def q8(values):
    s = np.abs(values).max(1) / 127.; s[s == 0] = 1.
    return np.clip(np.round(values / s[:, None]), -127, 127).astype(np.int8), s.astype(np.float32)


def session(nodes, inputs, outputs, custom=True):
    g = helper.make_graph(nodes, 'g', [helper.make_tensor_value_info(n, t, None) for n, t in inputs],
                          [helper.make_tensor_value_info(n, F, None) for n in outputs])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17), helper.make_opsetid('balbal', 1)])
    o = ort.SessionOptions(); o.intra_op_num_threads = 4
    if custom: o.register_custom_ops_library(LIB)
    return ort.InferenceSession(m.SerializeToString(), o, providers=['CPUExecutionProvider'])


def main():
    rng = np.random.default_rng(0); ok = True
    n = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    for H, S, D, V, K, T in ((4, 500, 64, 120, 16, 37), (8, 1000, 64, 192, 32, 21)):   # balbal-v2 / balbal-v2-manas memory sizes
        q = n(rng.standard_normal((1, T, H, 2, D))).astype(np.float32); keys = n(rng.standard_normal((H, 2, S, D))).astype(np.float32)
        values = rng.standard_normal((S * S, V)).astype(np.float32); inv_t = np.full(H, 10., np.float32); vq, sc = q8(values)
        y = session([helper.make_node('PkmRead', list('qkvt'), ['o'], domain='balbal', top_k=K)], [(x, F) for x in 'qkvt'], ['o']) \
            .run(None, dict(q=q, k=keys, v=values, t=inv_t))[0]
        y8 = session([helper.make_node('PkmReadQ8', ['q', 'k', 'vq', 's', 't'], ['o'], domain='balbal', top_k=K)],
                     [('q', F), ('k', F), ('vq', I8), ('s', F), ('t', F)], ['o']).run(None, dict(q=q, k=keys, vq=vq, s=sc, t=inv_t))[0]
        r, r8 = ref(q, keys, values, inv_t, K), ref(q, keys, vq.astype(np.float32) * sc[:, None], inv_t, K)
        res = dict(H=H, S=S, K=K, T=T, pkm=float(np.abs(y - r).max()), pkm_q8_vs_dequantized=float(np.abs(y8 - r8).max()),
                   pkm_q8_vs_float=float(np.abs(y8 - r).max()), scale=float(np.abs(r).max()))
        ok &= res['pkm'] < 1e-4 and res['pkm_q8_vs_dequantized'] < 1e-4
        print(res)
    for H, T in ((4, 1), (8, 57)):
        ac = rng.standard_normal((1, H, T, T)).astype(np.float32); bd = rng.standard_normal((1, H, T, 2 * T - 1)).astype(np.float32)
        div = np.array(8., np.float32); t = np.arange(T); idx = np.broadcast_to(t[None, :] - t[:, None] + T - 1, (1, H, T, T)).astype(np.int64)
        y = session([helper.make_node('RelPosScore', ['ac', 'bd', 'd'], ['o'], domain='balbal')], [('ac', F), ('bd', F), ('d', F)], ['o']) \
            .run(None, dict(ac=ac, bd=bd, d=div))[0]
        chain = session([helper.make_node('GatherElements', ['bd', 'i'], ['g'], axis=-1), helper.make_node('Add', ['ac', 'g'], ['s']),
                         helper.make_node('Div', ['s', 'd'], ['o'])], [('ac', F), ('bd', F), ('i', TensorProto.INT64), ('d', F)], ['o'], custom=False)
        r = chain.run(None, dict(ac=ac, bd=bd, i=idx, d=div))[0]
        same = bool(np.array_equal(y, r)); ok &= same
        print(dict(op='RelPosScore', H=H, T=T, bit_identical=same))
    print('OK' if ok else 'FAILED'); sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
