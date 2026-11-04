"""`balbal run`: transcribe files and folders, print the results or write one output file per input (txt / json / srt / vtt),
optionally in several processes at once (--jobs).
"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import json
import multiprocessing
from pathlib import Path
import sys
import time

from .engine import Model, default_threads
from .formats import render

AUDIO_SUFFIXES = {'.wav', '.flac', '.ogg', '.oga', '.opus', '.mp3', '.m4a', '.aac', '.wma', '.amr', '.3gp', '.webm', '.mp4', '.mkv',
                  '.mov', '.avi'}


def expand(paths):
    """Files as given, folders -> their audio / video files (recursive, sorted). Returns [(file, path relative to its folder)]."""
    out = []
    for p in paths:
        if p.is_dir():
            out += [(f, f.relative_to(p)) for f in sorted(p.rglob('*')) if f.is_file() and f.suffix.lower() in AUDIO_SUFFIXES]
        else:
            out.append((p, Path(p.name)))
    return out


_model, _options = None, None


def _init(model_args, options):
    global _model, _options
    with redirect_stdout(sys.stderr):
        _model = Model(**model_args)
    _options = options


def _work(path):
    try:
        with redirect_stdout(sys.stderr):
            return path, _model.transcribe(path, **_options), None
    except (OSError, ValueError) as error:
        return path, None, str(error)


def run(args):
    """args: the parsed `balbal run` options. Returns the exit status (1 if a file failed)."""
    fmt = 'json' if args.json else args.format
    files = expand(args.audio)
    if not files: print('balbal: no audio files found', file=sys.stderr); return 1
    if fmt in ('srt', 'vtt') and len(files) > 1 and args.output_dir is None:
        print(f'balbal: --format {fmt} for several files needs --output-dir', file=sys.stderr); return 2
    targets = {}
    if args.output_dir is not None:
        for f, rel in files: targets[f] = args.output_dir / rel.with_suffix('.' + fmt)
        if args.skip_existing:
            done = [f for f, _ in files if targets[f].exists()]
            if done: print(f'balbal: skipping {len(done)} file(s) with existing output', file=sys.stderr)
            files = [(f, rel) for f, rel in files if f not in set(done)]
    jobs = max(1, min(args.jobs, len(files) or 1))
    threads = max(1, (args.threads or default_threads()) // jobs) if jobs > 1 else args.threads
    model_args = dict(model=args.model, runtime=args.runtime, threads=threads, model_dir=args.model_dir, revision=args.revision,
                      settings=getattr(args, 'settings', None))
    options = dict(hotwords=args.hotwords or None, itn=not args.no_itn)
    many, status, t0, audio = len(files) > 1, 0, time.perf_counter(), 0.
    paths = [f for f, _ in files]
    if jobs == 1:
        _init(model_args, options)
        results = map(_work, paths)
    else:
        if args.model_dir is None:   # download once, before the workers start
            from .hub import download
            download(args.model, args.runtime, args.revision)
        pool = ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context('spawn'), initializer=_init,
                                   initargs=(model_args, options))
        results = pool.map(_work, paths)
        print(f'balbal: {len(paths)} files, {jobs} processes x {threads} thread(s)', file=sys.stderr, flush=True)
    for k, (path, result, error) in enumerate(results, 1):
        if error is not None:
            print(f'balbal: {path}: {error}', file=sys.stderr); status = 1
            continue
        audio += result['duration_seconds']
        if fmt == 'json': result = {'file': str(path), **result}
        if path in targets:
            targets[path].parent.mkdir(parents=True, exist_ok=True)
            targets[path].write_text(render(result, fmt), encoding='utf-8')
            print(f'balbal: [{k}/{len(paths)}] {targets[path]}', file=sys.stderr, flush=True)
        elif fmt == 'txt':
            print(f'{path}: {result["text"]}' if many else result['text'], flush=True)
        else:
            sys.stdout.write(render(result, fmt)); sys.stdout.flush()
    if jobs > 1: pool.shutdown()
    if args.output_dir is not None and paths:
        wall = time.perf_counter() - t0
        print(f'balbal: {audio / 60:.1f} min of audio in {wall:.1f} s ({audio / max(wall, 1e-9):.0f}x real time)', file=sys.stderr)
    return status
