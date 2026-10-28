"""Train / fine-tune balbal-v2 (Conformer-CTC 16 x d176 + 3-pass loop refiner + product-key acoustic memory, SentencePiece 1024).

  python training/balbal_v2/train.py runs/NAME --tokenizer-dir TOKENIZER_DIR --init WEIGHTS.pt --train train.jsonl --dev dev.jsonl
  torchrun --nproc_per_node 4 training/balbal_v2/train.py runs/NAME ...          # multi-GPU
Defaults = the balbal-v2 release run: from balbal-v1-architecture weights, 5 epochs of updates of 64 clips (micro 16), LR 1.4e-4,
500 warmup updates, flat, linear decay over the last 20 % to 10 %, memory values at 5x LR, Muon + AdamW (weight decay .001),
frame language head (weight .1), eval + 0.5 soup every quarter epoch. The language head is saved next to every checkpoint as *.lang.pt
(the checkpoint itself loads with balbal.models.build_model). Manifest format and the loop: training/common/data.py, loop.py.
"""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[2])]
from training.common import loop
from training.common.init import load_init

DEFAULTS = dict(epochs=5., batch=64, micro=16, lr=1.4e-4, warmup=500, decay=.2, weight_decay=.001, step_offset=512, seed=0)


def main():
    args = loop.arguments(__doc__, DEFAULTS).parse_args()
    from balbal import models as M
    model = M.build_model('balbal-v2', args.tokenizer_dir)
    report = load_init(model, 'balbal-v2', args.init, args.cond)
    print('INIT', report, flush=True)
    loop.run(args, 'balbal-v2', model, head_prefix=M.LANG_HEAD['balbal-v2'], separate_head=True, eval_bf16=False)


if __name__ == '__main__':
    main()
