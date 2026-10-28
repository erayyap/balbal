"""Weight soups.

  python training/soup.py OUT.pt A.pt B.pt [...]               uniform average of checkpoints (the balbal-v2-manas release recipe)
  python training/soup.py OUT.pt --mix A.pt:0.5 INIT.pt:0.5    weighted mix (weights sum to 1), e.g. trained x w + init x (1 - w)
Float tensors are averaged in float64; integer buffers (BatchNorm counters) come from the last checkpoint. Inputs may be weight files
or training states (state.pt). A balbal-v2 *.lang.pt next to every input is souped the same way.
"""
import argparse
import os
from pathlib import Path
import sys

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
from balbal.models import load_state


def soup(paths, weights):
    states = [load_state(p) for p in paths]
    keys = list(states[-1])
    if any(list(s) != keys for s in states): raise ValueError('checkpoints differ in keys')
    return {k: (sum(w * s[k].double() for w, s in zip(weights, states))).to(states[-1][k].dtype) if states[-1][k].is_floating_point()
            else states[-1][k] for k in keys}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out'); p.add_argument('inputs', nargs='*'); p.add_argument('--mix', nargs='+', default=None, help='PATH:WEIGHT ...')
    a = p.parse_args()
    if a.mix:
        paths = [m.rsplit(':', 1)[0] for m in a.mix]; weights = [float(m.rsplit(':', 1)[1]) for m in a.mix]
        if abs(sum(weights) - 1) > 1e-6: p.error('--mix weights must sum to 1')
    else:
        paths = a.inputs; weights = [1 / len(paths)] * len(paths) if paths else []
    if not paths: p.error('no checkpoints')
    for src, dst in [(paths, a.out)] + ([([q[:-3] + '.lang.pt' for q in paths], a.out[:-3] + '.lang.pt')]
                                        if all(os.path.exists(q[:-3] + '.lang.pt') for q in paths) else []):
        state = soup(src, weights); tmp = dst + '.tmp'; torch.save(state, tmp); os.replace(tmp, dst)
        print(f'{dst}: {len(src)} inputs, {sum(v.numel() for v in state.values())} values', flush=True)


if __name__ == '__main__':
    main()
