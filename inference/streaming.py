"""Live (streaming) transcription of a growing recording, same method as the balbal Android app.

Audio arrives in pieces of any size. It is cut into the same fixed 30 s chunks as offline transcription; a chunk that is
complete is decoded once and kept, and the open (last, growing) chunk is decoded again on every update, so each partial
result only costs one decode of at most 30 s of audio. Partial text can still change inside the open chunk; the text of
closed chunks never changes. finish() returns exactly what the offline transcribe() gives for the whole recording.
balbal-v1 is a full-context model (not a chunk-wise streaming architecture): this is re-decoding, not incremental decoding.

Works with either runtime (anything with transcribe_chunk(wave, offset_seconds, hotwords)):

  from runtime_onnx import Engine          # or: from transcribe import Transcriber
  from streaming import Stream
  stream = Stream(Engine(MODEL_DIR))
  for piece in microphone():               # mono float32 at 16 kHz, any length
      update = stream.feed(piece)          # None until update_seconds of new audio arrived
      if update: print(update['text'])
  print(stream.finish()['text'])

Command line, raw 16-bit mono 16 kHz PCM on stdin (prints one JSON line per update, the last one has "final": true):
  arecord -q -f S16_LE -r 16000 -c 1 | python streaming.py --model-dir MODEL_DIR -           # microphone (Linux)
  python streaming.py --model-dir MODEL_DIR --simulate speech.wav                         # a file, paced in real time
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class Stream:
    """One live recording; one instance must not be fed from several threads at once."""

    def __init__(self, engine, *, chunk_seconds=30., update_seconds=0.6, hotwords=None):
        self.engine, self.hotwords = engine, hotwords
        self.step = int(chunk_seconds * 16000)
        self.update_samples = int(update_seconds * 16000)
        self.closed = []             # results of complete chunks, never decoded again
        self.open = np.zeros(0, dtype=np.float32)
        self.pending = 0             # samples received since the last partial decode

    @property
    def seconds(self):
        return (len(self.closed) * self.step + len(self.open)) / 16000

    def _result(self, tail, final):
        chunks = self.closed + ([tail] if tail else [])
        return dict(text=' '.join(c['text'] for c in chunks if c['text']), words=[w for c in chunks for w in c['words']],
                    stable_text=' '.join(c['text'] for c in self.closed if c['text']),
                    seconds=self.seconds, final=final)

    def feed(self, samples):
        """Append mono float32 16 kHz samples. Returns a partial result {'text', 'words', 'stable_text', 'seconds',
        'final'} once at least update_seconds of new audio arrived since the last one, else None."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not np.isfinite(samples).all():
            raise ValueError('Audio contains nonfinite samples')
        self.open = np.concatenate([self.open, samples]); self.pending += len(samples)
        while len(self.open) >= self.step:   # a chunk filled up: decode it for good
            offset = len(self.closed) * self.step / 16000
            self.closed.append(self.engine.transcribe_chunk(self.open[:self.step], offset, self.hotwords))
            self.open = self.open[self.step:]
        if self.pending < self.update_samples:
            return None
        return self.partial()

    def partial(self):
        """Decode now: closed chunks + the open chunk as it is."""
        self.pending = 0
        tail = self.engine.transcribe_chunk(self.open, len(self.closed) * self.step / 16000, self.hotwords) if len(self.open) else None
        return self._result(tail, final=False)

    def finish(self):
        """The final result for everything fed so far (= offline transcribe() on the whole recording)."""
        tail = self.engine.transcribe_chunk(self.open, len(self.closed) * self.step / 16000, self.hotwords) if len(self.open) else None
        return self._result(tail, final=True)


def pcm_pieces(source, piece_samples, pace):
    """Raw s16le pieces from a binary file object, or (pace=True) a decoded file sliced and paced in real time."""
    if isinstance(source, np.ndarray):
        for i in range(0, len(source), piece_samples):
            if pace:
                time.sleep(min(piece_samples, len(source) - i) / 16000)
            yield source[i:i + piece_samples]
        return
    while True:
        data = source.read(piece_samples * 2)
        if not data:
            return
        yield np.frombuffer(data[:len(data) // 2 * 2], dtype='<i2').astype(np.float32) / 32768


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('audio', help="'-' = raw 16-bit mono 16 kHz PCM on stdin, or an audio file with --simulate")
    p.add_argument('--simulate', action='store_true', help='read an audio file and feed it in real time')
    p.add_argument('--model-dir', type=Path, default=Path(os.getenv('BALBAL_MODEL_DIR', ROOT.parent / 'models')))
    p.add_argument('--runtime', choices=['onnx', 'torch'], default='onnx')
    p.add_argument('--threads', type=int, default=None)
    p.add_argument('--update-seconds', type=float, default=0.6, help='audio between partial results (default 0.6)')
    p.add_argument('--hotwords', default='', help='comma-separated names/words to favour, e.g. "Arslan,Zaynep"')
    args = p.parse_args()
    hotwords = [w.strip() for w in args.hotwords.split(',') if w.strip()] or None
    if args.runtime == 'onnx':
        from runtime_onnx import Engine
        engine = Engine(args.model_dir, args.threads)
    else:
        from contextlib import redirect_stdout
        from transcribe import Transcriber
        with redirect_stdout(sys.stderr):
            engine = Transcriber(args.model_dir, 'cpu', args.threads or 4)
    stream = Stream(engine, update_seconds=args.update_seconds, hotwords=hotwords)
    if args.simulate:
        from transcribe import read_audio
        source = read_audio(args.audio)
    elif args.audio == '-':
        source = sys.stdin.buffer
    else:
        p.error("pass '-' for PCM on stdin, or --simulate FILE")
    for piece in pcm_pieces(source, int(args.update_seconds * 16000), args.simulate):
        update = stream.feed(piece)
        if update:
            print(json.dumps({k: update[k] for k in ('seconds', 'text', 'final')}, ensure_ascii=False), flush=True)
    print(json.dumps(stream.finish(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
