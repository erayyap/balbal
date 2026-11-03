"""`balbal live`: live transcription from the microphone (pip install 'balbal[live]'), or a file played as if live.

Partials are shown on one line that keeps changing (on a terminal), finals are printed one segment per line; Ctrl+C ends a microphone
session and prints the whole transcript. --json prints every event as a JSON line instead.
"""
import json
import queue
import shutil
import sys
import time

import numpy as np


BLOCK = 1600   # 0.1 s pieces at 16 kHz


class Printer:
    def __init__(self, as_json=False, timestamps=False, stream=sys.stdout):
        self.json, self.timestamps, self.out = as_json, timestamps, stream
        self.tty = stream.isatty() and not as_json
        self.open = False   # a partial line is showing
        self.last = None    # (start, text) of the last final line

    def __call__(self, events):
        for e in events:
            if self.json:
                print(json.dumps(e, ensure_ascii=False), file=self.out, flush=True); continue
            if e['type'] == 'partial':
                if not self.tty: continue
                width = max(20, shutil.get_terminal_size().columns - 1)
                text = e['text'] if len(e['text']) < width else '…' + e['text'][-(width - 1):]
                self.out.write('\r\x1b[2K\x1b[2m' + text + '\x1b[0m'); self.out.flush(); self.open = True
            elif e['type'] == 'final':
                if self.open: self.out.write('\r\x1b[2K'); self.open = False
                if e['joins'] and self.tty and self.last:   # continues the previous segment's last word: rewrite that line
                    start, text = self.last
                    width = max(20, shutil.get_terminal_size().columns)
                    rows = max(1, -(-len(self.line(start, text, e['end'])) // width))
                    self.out.write('\x1b[1A\r\x1b[2K' * rows)
                    self.last = start, text + e['text']
                else:
                    self.last = e['start'], e['text']
                print(self.line(self.last[0], self.last[1], e['end']), file=self.out, flush=True)

    def line(self, start, text, end):
        return (f'[{start:7.2f} -> {end:7.2f}] ' if self.timestamps else '') + text

    def clear(self):
        if self.open: self.out.write('\r\x1b[2K'); self.out.flush(); self.open = False


class Decimator:
    """Device rate -> 16 kHz for an integer ratio (48 kHz, 32 kHz): a windowed-sinc low-pass FIR with its state kept between blocks,
    then every n-th sample, so block edges leave no clicks."""

    def __init__(self, rate):
        from scipy.signal import firwin
        self.n = rate // 16000
        self.taps = firwin(24 * self.n + 1, 1. / self.n, window=('kaiser', 5.0)) if self.n > 1 else None
        self.zi, self.phase = (np.zeros(len(self.taps) - 1) if self.taps is not None else None), 0

    def __call__(self, x):
        if self.taps is None: return x
        from scipy.signal import lfilter
        y, self.zi = lfilter(self.taps, 1., x.astype(np.float64), zi=self.zi)
        out = y[self.phase::self.n]
        self.phase = (self.phase - len(x)) % self.n
        return out.astype(np.float32)


def open_microphone(device, pieces):
    """A started sounddevice InputStream that puts float32 16 kHz pieces into `pieces`. Tries 16 kHz first, then the device's own rate
    when it is a multiple of 16 kHz."""
    try:
        import sounddevice as sd
    except (ImportError, OSError) as error:
        raise SystemExit("balbal live needs the live extra: pip install 'balbal[live]' (Linux also needs PortAudio: "
                         f"sudo apt install libportaudio2) [{error}]")
    info = sd.query_devices(device, 'input')
    rates = [16000] + [r for r in (int(info['default_samplerate']), 48000, 32000) if r % 16000 == 0 and r != 16000]
    last = None
    for rate in dict.fromkeys(rates):
        try:
            decimate = Decimator(rate)
            def callback(data, frames, timing, status, decimate=decimate):
                pieces.put(decimate(data[:, 0].copy()))
            stream = sd.InputStream(samplerate=rate, blocksize=BLOCK * rate // 16000, device=device, channels=1, dtype='float32',
                                    callback=callback)
            stream.start()
            return stream, info['name'], rate
        except Exception as error:   # PortAudioError: rate not supported by this device / host API
            last = error
    raise SystemExit(f'balbal live: could not open {info["name"]} at 16 / 32 / 48 kHz ({last}); pick another --device '
                     '(python -m sounddevice lists them)')


def run_microphone(stream, device=None, printer=None):
    pieces = queue.Queue()
    mic, name, rate = open_microphone(device, pieces)
    print(f'balbal: listening on {name} ({rate} Hz); Ctrl+C to stop', file=sys.stderr, flush=True)
    try:
        while True:
            chunk = [pieces.get()]
            while not pieces.empty(): chunk.append(pieces.get_nowait())   # catch up in one window if decoding fell behind
            printer(stream.accept(np.concatenate(chunk)))
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop(); mic.close()
    while not pieces.empty(): printer(stream.accept(pieces.get_nowait()))
    printer(stream.finish())


def run_file(stream, wave, speed=1., printer=None):
    """Feed a recording in 0.1 s pieces, at `speed` x real time (0 = as fast as possible)."""
    t0 = time.perf_counter()
    for i in range(0, len(wave), BLOCK):
        if speed > 0: time.sleep(max(0., t0 + min(i + BLOCK, len(wave)) / 16000 / speed - time.perf_counter()))
        printer(stream.accept(wave[i:i + BLOCK]))
    printer(stream.finish())


def main(args):
    from contextlib import redirect_stdout
    from .audio import read_audio
    from .engine import Model
    printer = Printer(args.json, args.timestamps)
    with redirect_stdout(sys.stderr):
        model = Model(args.model, args.runtime, args.threads, args.model_dir, args.revision)
    options = {k: v for k, v in dict(chunk=args.chunk, right=args.right, endpoint=args.endpoint, max_segment=args.max_segment).items()
               if v is not None}
    stream = model.stream(args.mode, hotwords=args.hotwords or None, itn=not args.no_itn, **options)
    if args.file:
        run_file(stream, read_audio(args.file), args.speed, printer)
    else:
        run_microphone(stream, args.device, printer)
    printer.clear()
    st = stream.stats
    if not args.json:
        print(f'\nbalbal: {st["audio_seconds"]:.1f} s of audio, compute {100 * st["compute_rtf"]:.1f}% of real time', file=sys.stderr)
    if args.transcript and not args.json:
        print('\n' + stream.transcript(), flush=True)
