"""Offline transcription with balbal-v2 / balbal-v2-manas: audio files -> one JSON line each (text, per-word timestamps, timings).

  python inference/transcribe.py --model-dir MODEL_DIR speech.wav                       # fast CPU runtime (custom kernels)
  python inference/transcribe.py --model-dir MODEL_DIR --runtime onnx speech.wav        # standard onnxruntime operators only
  python inference/transcribe.py --model-dir MODEL_DIR --runtime torch speech.wav       # PyTorch/NeMo (--device cuda for a GPU)
  python inference/transcribe.py --model-dir MODEL_DIR --hotwords "Eray,Hasan" call.wav
MODEL_DIR layout and runtimes: see runtime.py. --model overrides MODEL_DIR/model.json. WAV/FLAC/OGG are read directly, other
formats through FFmpeg; resampling to 16 kHz uses the training resampler (scipy polyphase).
"""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from runtime import Engine, RUNTIMES
from balbal.audio import read_audio
from balbal.models import MODELS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('audio', type=Path, nargs='+')
    p.add_argument('--model-dir', type=Path, default=os.getenv('BALBAL_MODEL_DIR'), required=not os.getenv('BALBAL_MODEL_DIR'))
    p.add_argument('--model', choices=MODELS, default=None, help='default: MODEL_DIR/model.json')
    p.add_argument('--runtime', choices=RUNTIMES, default='fast')
    p.add_argument('--decoder', choices=('fast', 'flashlight'), default='fast', help='beam search implementation (same best path)')
    p.add_argument('--lm-dir', type=Path, default=None, help='default: MODEL_DIR/lm')
    p.add_argument('--device', default='cpu', help='runtime torch only: cpu or cuda')
    p.add_argument('--threads', type=int, default=None, help='CPU threads (default: ASR_THREADS or 8)')
    p.add_argument('--hotwords', default='', help='comma-separated names/words to favour, e.g. "Eray,Hasan"')
    p.add_argument('--text-only', action='store_true')
    args = p.parse_args()
    hotwords = [w.strip() for w in args.hotwords.split(',') if w.strip()]
    with redirect_stdout(sys.stderr):   # keep stdout one line per file
        engine = Engine(args.model_dir, args.model, args.runtime, args.threads, args.device, args.decoder, lm_dir=args.lm_dir)
    for path in args.audio:
        with redirect_stdout(sys.stderr):
            result = engine.transcribe(read_audio(path), hotwords=hotwords)
        print(result['text'] if args.text_only else json.dumps({'file': str(path), **result}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
