"""Train / fine-tune balbal-v2-manas (FastConformer-CTC 18 x d512 at 4x + 3-pass loop refiner + product-key acoustic memory with a
learned per-head temperature + frame language head, SentencePiece 2048 Turkish + English).

  python training/manas/train.py runs/NAME --tokenizer-dir TOKENIZER_DIR --init WEIGHTS.pt --train train.jsonl --dev dev.jsonl
  python training/manas/train.py runs/NAME --tokenizer-dir TOKENIZER_DIR --nemo stt_en_fastconformer_ctc_large.nemo --step-offset 0 ...
  torchrun --nproc_per_node 4 training/manas/train.py runs/NAME ...                   # multi-GPU
--nemo starts the encoder and feature frontend from NVIDIA's English FastConformer-CTC-Large (CC-BY-4.0); the CTC head (new tokenizer),
refiner, memory and language head start fresh (--step-offset 0: the memory read strength ramps in over updates 128-512; the default
512 = full strength from the first update, for continuing trained weights).
Defaults = the main balbal-v2-manas training stage (continuing an earlier stage): 4.5 epochs of updates of 64
clips (micro 8), LR 1.4e-4, 500 warmup updates, flat to epoch 2.5, then linear decay to 10 % (--decay .4444), memory values and temperature at 5x LR, Muon (bf16 Newton-Schulz) + AdamW (weight decay .01),
frame language head (weight .1). The released weights then had a fine-tune stage
  --lr 5e-5 --epochs 2 --decay .5 --soup ""
and are the uniform soup (training/soup.py) of its 7 evaluations from epoch 0.5 to 2.0.
"""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[2])]
from training.common import loop
from training.common.init import load_init

DEFAULTS = dict(epochs=4.5, batch=64, micro=8, lr=1.4e-4, warmup=500, decay=.4444, weight_decay=.01, step_offset=512, seed=0,
                muon_bf16=True)


def from_nemo(model, path):
    """Encoder + feature frontend weights of a NeMo FastConformer-CTC checkpoint (same encoder configuration)."""
    from nemo.collections.asr.models import EncDecCTCModelBPE
    src = EncDecCTCModelBPE.restore_from(str(path), map_location='cpu').state_dict()
    take = {k: v for k, v in src.items() if k.startswith(('encoder.', 'preprocessor.'))}
    own = model.state_dict()
    bad = [k for k, v in take.items() if k not in own or own[k].shape != v.shape]
    if bad: raise ValueError(f'{len(bad)} encoder tensors do not match, e.g. {bad[:3]}')
    model.load_state_dict(take, strict=False)
    return dict(nemo=str(path), tensors=len(take))


def main():
    p = loop.arguments(__doc__, DEFAULTS)
    p.add_argument('--nemo', type=Path, default=None, help='start the encoder from a NeMo FastConformer-CTC-Large .nemo file')
    args = p.parse_args()
    if args.nemo and args.init: p.error('--nemo and --init exclude each other')
    from balbal import models as M
    model = M.build_model('balbal-v2-manas', args.tokenizer_dir)
    report = load_init(model, 'balbal-v2-manas', args.init, args.cond)
    if args.nemo: report.update(from_nemo(model, args.nemo))
    print('INIT', report, flush=True)
    loop.run(args, 'balbal-v2-manas', model, head_prefix=M.LANG_HEAD['balbal-v2-manas'], separate_head=False, eval_bf16=True)


if __name__ == '__main__':
    main()
