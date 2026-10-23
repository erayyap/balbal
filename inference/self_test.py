"""Decoder sanity check with synthetic emissions (not an accuracy benchmark).
  python self_test.py --model-dir MODEL_DIR      # needs MODEL_DIR/lm/{model.binary, tokenizer.model}
"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from decoder import KenLMDecoder


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-dir', type=Path, default=Path(os.getenv('BALBAL_MODEL_DIR', ROOT.parent / 'models')))
    decoder = KenLMDecoder(p.parse_args().model_dir / 'lm')
    text = 'Yarın saat on beşte toplantı var mı?'
    ids = decoder.sp.encode(text)
    frames = [decoder.blank_id]
    for token in ids:
        frames.extend([token, token, decoder.blank_id])
    scores = np.full((len(frames), decoder.vocab_size + 1), -80., dtype=np.float32)
    scores[np.arange(len(frames)), frames] = 0.
    scores -= np.log(np.exp(scores).sum(axis=1, keepdims=True))
    first = decoder.decode(scores)
    assert first == decoder.sp.decode(ids), first
    assert decoder.decode(scores) == first  # beam state resets between calls
    print('PASS: native LM, tokenizer mapping, CTC collapse, calibration. Example:', first)


if __name__ == '__main__':
    main()
