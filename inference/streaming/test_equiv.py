"""Streaming exactness check on audio files.

  python inference/streaming/test_equiv.py --model-dir MODEL_DIR a.wav b.wav ...
For each file:
  window   with endpointing off, the streamed text equals the offline fast decoder on the frames the stream committed
  pieces   the streamed text does not depend on the piece size (0.05 s vs 0.16 s)
  segment  segment mode with endpointing off equals the offline fast decoder on the frames it decoded
  default  segment mode with default settings vs offline decoding of the whole file (reported; endpoints may change the text)
Exits nonzero if a check in the first three groups fails.
"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import streamer as T
from balbal.audio import read_audio
from decoder import KenLMDecoder


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('audio', type=Path, nargs='+')
    p.add_argument('--model-dir', type=Path, default=os.getenv('BALBAL_MODEL_DIR'), required=not os.getenv('BALBAL_MODEL_DIR'))
    p.add_argument('--model', choices=T.M.MODELS, default=None)
    p.add_argument('--runtime', choices=('fast', 'onnx'), default='fast')
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    eng = T.Engine(a.model_dir, a.model, a.runtime, a.threads)
    off = KenLMDecoder(a.model_dir / 'lm', eng.cfg, fast=True).decode
    off_kw = dict(endpoint=1e9, max_segment=1e9, trace=True)
    n = len(a.audio); ok = dict(window=0, pieces=0, segment=0, default=0)
    for path in a.audio:
        w = read_audio(path)
        txt, _, st = T.transcribe(eng, w, mode='window', first_right=.64, second_pass=False, **off_kw)
        ref = off(np.concatenate(st.trace)); ok['window'] += txt == ref
        txt2, _, _ = T.transcribe(eng, w, piece=.05, mode='window', first_right=.64, second_pass=False, **off_kw); ok['pieces'] += txt2 == txt
        txt3, _, st3 = T.transcribe(eng, w, mode='segment', **off_kw); ok['segment'] += txt3 == off(np.concatenate(st3.trace))
        txt4, _, _ = T.transcribe(eng, w, mode='segment'); full = off(eng.forward(w)); ok['default'] += txt4 == full
        print(f'{path.name}\n  stream  {txt4}\n  offline {full}', flush=True)
    print(f'{eng.model}: window == offline on committed frames {ok["window"]}/{n}; piece size invariant {ok["pieces"]}/{n}; '
          f'segment == offline on its frames {ok["segment"]}/{n}; segment default == offline on the whole file {ok["default"]}/{n}')
    sys.exit(0 if ok['window'] == ok['pieces'] == ok['segment'] == n else 1)


if __name__ == '__main__':
    main()
