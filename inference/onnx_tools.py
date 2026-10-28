"""ONNX graph rewrites used by export_onnx.py (all on onnx.ModelProto, no torch).

quantize_pkm   every balbal::PkmRead whose value table is an initializer becomes balbal::PkmReadQ8(query, keys, values int8 [N, V],
               scales [N], inv_temp): symmetric int8 per row (values ~= q * scale), 4x less memory; a table shared by the three refiner
               passes is quantized once. Other weights are untouched.
fuse_relpos    each encoder layer's rel-shift chain GatherElements(bd, idx) -> Add(ac, .) -> Div(., c) becomes balbal::RelPosScore(ac, bd, c);
               the dead index subgraph (Range / Sub / Expand ...) is removed.
externalize    initializers >= 4 KiB move to one 64 KiB-aligned PATH.data file next to PATH, which onnxruntime memory-maps (file-backed,
               shared, evictable pages instead of a private copy).
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def quantize_pkm(m):
    inits = {t.name: t for t in m.graph.initializer}
    nodes = [n for n in m.graph.node if n.domain == 'balbal' and n.op_type == 'PkmRead' and n.input[2] in inits]
    assert nodes, 'no PkmRead node with an initializer value table'
    report = []
    for name in sorted({n.input[2] for n in nodes}):
        assert all(n in nodes for n in m.graph.node if name in n.input), 'value table also used by a non-PkmRead node'
        v = numpy_helper.to_array(inits[name]).astype(np.float32)
        s = np.abs(v).max(1) / 127.; s[s == 0] = 1.
        vq = np.clip(np.round(v / s[:, None]), -127, 127).astype(np.int8)
        idx = [i for i, x in enumerate(m.graph.initializer) if x.name == name]; assert len(idx) == 1
        del m.graph.initializer[idx[0]]
        m.graph.initializer.extend([numpy_helper.from_array(vq, name + '_q8'), numpy_helper.from_array(s.astype(np.float32), name + '_q8_scale')])
        users = [n for n in nodes if n.input[2] == name]
        for node in users:
            q, k, _, it = node.input; del node.input[:]
            node.input.extend([q, k, name + '_q8', name + '_q8_scale', it]); node.op_type = 'PkmReadQ8'
        report.append(dict(table=list(v.shape), nodes=len(users), max_abs_error=float(np.abs(vq.astype(np.float32) * s[:, None] - v).max())))
    return report


def fuse_relpos(m):
    g = m.graph; users = {}
    for n in g.node:
        for i in n.input: users.setdefault(i, []).append(n)
    fused, drop = 0, set()
    for ge in [n for n in g.node if n.op_type == 'GatherElements']:
        axis = [a.i for a in ge.attribute if a.name == 'axis']
        u = users.get(ge.output[0], [])
        if axis not in ([-1], [3]) or len(u) != 1 or u[0].op_type != 'Add': continue
        add = u[0]; ac = add.input[1] if add.input[0] == ge.output[0] else add.input[0]
        u2 = users.get(add.output[0], [])
        if len(u2) != 1 or u2[0].op_type != 'Div' or u2[0].input[0] != add.output[0]: continue
        div = u2[0]
        node = helper.make_node('RelPosScore', [ac, ge.input[0], div.input[1]], [div.output[0]],
                                name=ge.name.replace('GatherElements', 'RelPosScore'), domain='balbal')
        g.node.insert(list(g.node).index(div), node)
        drop |= {id(ge), id(add), id(div)}; fused += 1
    keep = [n for n in g.node if id(n) not in drop]
    need, live = {o.name for o in g.output}, []   # dead-node elimination
    for n in reversed(keep):
        if any(o in need for o in n.output):
            live.append(n); need |= set(n.input)
    live.reverse(); removed = len(g.node) - len(live)
    del g.node[:]; g.node.extend(live)
    if not any(o.domain == 'balbal' for o in m.opset_import): m.opset_import.append(helper.make_opsetid('balbal', 1))
    used = {i for n in g.node for i in n.input}
    inits = [t for t in g.initializer if t.name in used]; del g.initializer[:]; g.initializer.extend(inits)
    return dict(fused=fused, nodes_removed=removed)


def externalize(m, path, align=1 << 16, minimum=4096):
    path = Path(path); data = path.with_name(path.name + '.data'); moved = 0
    with open(data, 'wb') as f:
        def walk(g):
            nonlocal moved
            for t in g.initializer:
                if t.HasField('raw_data') and len(t.raw_data) >= minimum:
                    f.write(b'\0' * (-f.tell() % align)); off = f.tell(); f.write(t.raw_data)
                    n = len(t.raw_data); t.ClearField('raw_data'); t.data_location = TensorProto.EXTERNAL
                    del t.external_data[:]
                    for k, v in (('location', data.name), ('offset', str(off)), ('length', str(n))):
                        e = t.external_data.add(); e.key = k; e.value = v
                    moved += 1
            for node in g.node:
                for a in node.attribute:
                    if a.type == onnx.AttributeProto.GRAPH: walk(a.g)
                    for sg in a.graphs: walk(sg)
        walk(m.graph)
    onnx.save(m, str(path))
    return dict(tensors=moved, onnx_bytes=path.stat().st_size, data_bytes=data.stat().st_size)
