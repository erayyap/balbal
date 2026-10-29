"""balbal command line.

  balbal run speech.wav                      text of each file (balbal-v2; downloaded from Hugging Face on first use)
  balbal run --model manas a.wav b.mp3       balbal-v2-manas (Turkish + English)
  balbal run --json speech.wav               one JSON line per file: text, per-word timestamps, timings
  balbal run --hotwords "Eray,Hasan" call.wav
  balbal download [--model manas]            fetch the model files now (e.g. while building an image)
"""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from . import __version__
from .engine import RUNTIMES, Model
from .hub import MODELS, download

COMMANDS = ('run', 'download')


def model_options(p):
    p.add_argument('--model', '-m', default='balbal-v2', help=f'{" or ".join(MODELS)} (short: v2, manas); default balbal-v2')
    p.add_argument('--runtime', choices=RUNTIMES, default='fast', help='fast (custom kernels, default) or onnx (standard operators)')
    p.add_argument('--revision', default=None, help='Hugging Face revision (default: the tested one; main = latest)')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith('-'): argv.insert(0, 'run')   # balbal speech.wav
    p = argparse.ArgumentParser(prog='balbal', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', action='version', version=f'balbal {__version__}')
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='transcribe audio files')
    r.add_argument('audio', type=Path, nargs='+', help='WAV / FLAC / OGG, other formats (MP3, M4A, video) through FFmpeg')
    model_options(r)
    r.add_argument('--model-dir', type=Path, default=None, help='a local model directory instead of the download')
    r.add_argument('--threads', '-t', type=int, default=None, help='CPU threads (default: BALBAL_THREADS, else min(8, CPU count))')
    r.add_argument('--hotwords', default='', help='comma-separated names / words to favour, e.g. "Eray,Hasan"')
    r.add_argument('--no-itn', action='store_true', help='keep numbers as spoken words (default: digits)')
    r.add_argument('--json', action='store_true', help='one JSON line per file (text, words with start/end seconds, timings)')
    d = sub.add_parser('download', help='download the model files without transcribing')
    model_options(d)
    args = p.parse_args(argv)

    if args.command == 'download':
        print(download(args.model, args.runtime, args.revision))
        return
    missing = [str(f) for f in args.audio if not f.is_file()]
    if missing: p.error('no such file: ' + ', '.join(missing))
    with redirect_stdout(sys.stderr):   # stdout carries only the results
        model = Model(args.model, args.runtime, args.threads, args.model_dir, args.revision)
    many, status = len(args.audio) > 1, 0
    for path in args.audio:
        try:
            with redirect_stdout(sys.stderr):
                result = model.transcribe(path, hotwords=args.hotwords, itn=not args.no_itn)
        except (OSError, ValueError) as error:
            print(f'balbal: {path}: {error}', file=sys.stderr); status = 1
            continue
        if args.json: print(json.dumps({'file': str(path), **result}, ensure_ascii=False), flush=True)
        else: print(f'{path}: {result["text"]}' if many else result['text'], flush=True)
    sys.exit(status)


if __name__ == '__main__':
    main()
