"""Weight soup of same-architecture checkpoints (the --soup evaluation of train.py, as a standalone tool).
  python training/soup.py OUT.pt 0.5:checkpoints/step6000.pt 0.5:init.pt
Weighted sum of floating-point tensors; integer buffers (e.g. BN counters) come from the first part. Shares should sum to 1.
"""
import sys

import torch


def mix(parts):
    """Weighted sum of floating-point tensors; integer buffers (e.g. BN counters) from the first part."""
    base = parts[0][1]; out = {}
    for k, v in base.items():
        out[k] = sum(w*s[k].float() for w, s in parts).to(v.dtype) if v.is_floating_point() else v.clone()
    return out


def main():
    if len(sys.argv) < 3: raise SystemExit(__doc__)
    parts = []
    for item in sys.argv[2:]:
        share, _, path = item.partition(':')
        parts.append((float(share), torch.load(path, map_location='cpu', weights_only=True)))
    if abs(sum(w for w, _ in parts) - 1) > 1e-6: raise SystemExit('Shares must sum to 1')
    torch.save(mix(parts), sys.argv[1])


if __name__ == '__main__':
    main()
