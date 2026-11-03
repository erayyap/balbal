"""CPU speech recognition with balbal-v2 / balbal-v2-manas: ONNX acoustic model + KenLM beam decoder (same path as the
inference/ code of github.com/erayyap/balbal).

Runtimes:
  fast   lean batch-1 graph with the balbal onnxruntime custom operators (fused product-key memory read with an int8 value table,
         rel-pos attention score); the operator library ships in this wheel, built for x86-64-v2 / v3 / v4 (AVX-512), picked per CPU
  onnx   the same model with standard onnxruntime operators only (larger download, slower); also the fallback when the operator
         library does not load
Both give [frames, V + 1] natural-log CTC probabilities (40 ms frames, blank last) to decoder.KenLMDecoder.
Audio longer than 30 s: overlapping windows (LONG_WINDOWS: 30 s windows, 10 s overlap); each window keeps only
the frames up to the middle of its overlaps (its edge frames saw little context), and since CTC frames sit on a fixed 40 ms grid the
kept frames join into one emission matrix for the whole file.
The KenLM search then runs over that, so no word is split at a window edge and the LM context carries on. Very long files are decoded in
pieces of about SEGMENT_SECONDS, cut in the middle of the longest confident-blank (silent) run near the end of each piece, so memory stays
bounded. overlap_seconds=0 gives the old behaviour: fixed chunks, each decoded on its own.
The models write numbers as words; transcribe() turns spoken number runs into digits (ITN) unless itn=False.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np

from . import hub
from .audio import SAMPLE_RATE, canonical_audio, read_audio
from .decoder import KenLMDecoder
from .itn import itn_text, itn_words

HERE = Path(__file__).resolve().parent
RUNTIMES = ('fast', 'onnx')
MIN_SAMPLES = 4000   # the convolutional subsampler needs at least 0.25 s
HOP = 640            # samples per output frame (10 ms hop x 4 subsampling): frame j of a window at sample s = file frame s / HOP + j
SEGMENT_SECONDS = 300.
# (window, overlap) seconds per model, chosen on real long recordings (12 whole phone channels, 3.9 h; WER, fixed 30 s chunks ->
# stitched): balbal-v2 22.31 -> 21.63 (15 s / 3 s: 23.06; 20 s / 8 s: 22.04), manas 19.38 -> 19.11 (30 s / 5 s: 19.02).
SINGLE_PASS_SECONDS = 30.   # clips up to this length are never windowed
LONG_WINDOWS = {'balbal-v2': (30., 10.), 'balbal-v2-manas': (30., 10.)}


def cpu_level():
    """x86-64 micro-architecture level of this CPU for the operator library: 'v4' (AVX-512), 'v3' (AVX2 + FMA) or 'v2'."""
    try:
        flags = set(next(line for line in open('/proc/cpuinfo') if line.startswith('flags')).split(':', 1)[1].split())
    except (OSError, StopIteration):
        return 'v2'
    if {'avx512f', 'avx512bw', 'avx512cd', 'avx512dq', 'avx512vl'} <= flags: return 'v4'
    if {'avx', 'avx2', 'bmi1', 'bmi2', 'f16c', 'fma', 'movbe'} <= flags: return 'v3'
    return 'v2'


def ops_library():
    """The custom operator library for this CPU (BALBAL_OPS_LIB overrides)."""
    return Path(os.getenv('BALBAL_OPS_LIB') or HERE / '_native' / f'libbalbal_ops_{cpu_level()}.so')


def ort_session(path, threads, custom_ops=False):
    import onnxruntime as ort
    o = ort.SessionOptions(); o.intra_op_num_threads = threads; o.inter_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.enable_mem_pattern = False   # every utterance has another length
    o.log_severity_level = 3
    if custom_ops:
        try:
            o.register_custom_ops_library(str(ops_library()))
        except Exception as error:
            raise OSError(f'custom op library {ops_library().name}: {error}') from error
    return ort.InferenceSession(str(path), o, providers=['CPUExecutionProvider'])


def default_threads():
    return int(os.getenv('BALBAL_THREADS') or os.getenv('ASR_THREADS') or min(8, os.cpu_count() or 1))


class Model:
    """A loaded balbal model. Load once and reuse; one instance must not be called from two threads at once (make one per thread).

    model      'balbal-v2' (default, 45M, Turkish) or 'balbal-v2-manas' (313M, Turkish + English); short: 'v2', 'manas'
    runtime    'fast' (default) or 'onnx' (standard operators only)
    threads    CPU threads (default: BALBAL_THREADS, else min(8, CPU count))
    model_dir  a local model directory (model.json, onnx/, lm/) instead of the Hugging Face download
    revision   Hugging Face revision (default: the one this release was tested with; 'main' = latest)
    """

    def __init__(self, model='balbal-v2', runtime='fast', threads=None, model_dir=None, revision=None, cache_dir=None, settings=None):
        if runtime not in RUNTIMES: raise ValueError(f'runtime must be one of {RUNTIMES}')
        if model_dir is not None:
            model_dir = Path(model_dir)
            f = model_dir / 'model.json'
            model = json.loads(f.read_text())['model'] if f.exists() and model == 'balbal-v2' else model
        self.model = hub.model_name(model)
        self.threads = int(threads or default_threads())
        self.revision, self.cache_dir = revision, cache_dir
        self.model_dir = model_dir or hub.download(self.model, runtime, revision, cache_dir)
        self.settings = dict(json.loads((HERE / 'configs' / self.model / 'decoder.json').read_text()), **(settings or {}))
        self.runtime = runtime
        try:
            self.session = self._session(runtime)
        except Exception as error:   # the operator library did not load on this system: the standard-operator graph instead
            if runtime != 'fast' or not isinstance(error, OSError): raise
            warnings.warn(f'balbal: the fast runtime did not load ({error}); using the standard-operator graph (runtime="onnx")')
            if model_dir is None: self.model_dir = hub.download(self.model, 'onnx', revision, cache_dir)
            self.runtime, self.session = 'onnx', self._session('onnx')
        self.decoder = KenLMDecoder(self.model_dir / 'lm', self.settings)
        self.forward(np.zeros(SAMPLE_RATE, np.float32))   # warm-up

    def _session(self, runtime):
        graph = self.model_dir / 'onnx' / ('model.onnx' if runtime == 'fast' else 'model-plain.onnx')
        return ort_session(graph, self.threads, custom_ops=runtime == 'fast')

    def __repr__(self):
        return f'balbal.Model({self.model!r}, runtime={self.runtime!r}, threads={self.threads})'

    def forward(self, wave):
        """Mono float32 16 kHz waveform (<= 30 s) -> [frames, V + 1] log-probabilities."""
        wave = np.ascontiguousarray(wave, dtype=np.float32)
        if len(wave) < MIN_SAMPLES: wave = np.pad(wave, (0, MIN_SAMPLES - len(wave)))
        return self.session.run(None, {'audio': wave[None]})[0]

    def transcribe_chunk(self, wave, offset_seconds=0., hotwords=None):
        """One chunk (<= 30 s) -> {'text', 'words', 'acoustic_seconds', 'decoder_seconds'}; word times shifted by offset_seconds."""
        t0 = time.perf_counter(); lp = self.forward(wave); t1 = time.perf_counter()
        out = self.decoder.decode_words(lp, hotwords, offset_seconds)
        return dict(out, acoustic_seconds=t1 - t0, decoder_seconds=time.perf_counter() - t1)

    def windows(self, wave, chunk_seconds=30., overlap_seconds=5.):
        """Overlapping windows over a long waveform -> (first file frame, [frames, V + 1] log-probabilities) per window, kept frames only;
        the pieces are contiguous on the file's 40 ms frame grid. Also the acoustic seconds spent."""
        size = max(HOP, int(chunk_seconds * SAMPLE_RATE) // HOP * HOP)
        overlap = min(int(overlap_seconds * SAMPLE_RATE) // HOP * HOP, size // 2)
        starts = [0]
        while starts[-1] + size < len(wave): starts.append(starts[-1] + size - overlap)
        # Window k keeps file frames [cuts[k], cuts[k + 1]): the cut sits in the middle of the overlap with the next window.
        cuts = [0] + [(b + overlap // 2) // HOP for b in starts[1:]] + [None]
        spent = 0.
        for k, s in enumerate(starts):
            t = time.perf_counter(); lp = self.forward(wave[s:s + size]); spent += time.perf_counter() - t
            first = s // HOP
            end = len(lp) if cuts[k + 1] is None else cuts[k + 1] - first
            yield cuts[k], lp[cuts[k] - first:end], spent

    def transcribe_long(self, wave, hotwords=None, chunk_seconds=30., overlap_seconds=5., segment_seconds=SEGMENT_SECONDS):
        """A waveform of any length -> {'text', 'words', 'acoustic_seconds', 'decoder_seconds'} (see the module docstring)."""
        segment = max(1, int(segment_seconds / 0.04))
        texts, words, pending, start, acoustic, decoding = [], [], [], 0, 0., 0.

        def decode(lp, first):
            nonlocal decoding
            t = time.perf_counter(); out = self.decoder.decode_words(lp, hotwords, first * 0.04); decoding += time.perf_counter() - t
            if out['text']: texts.append(out['text'])
            words.extend(out['words'])

        for _, lp, acoustic in self.windows(wave, chunk_seconds, overlap_seconds):
            pending.append(lp)
            if sum(len(p) for p in pending) < segment: continue
            buf = np.concatenate(pending)
            cut = silence_cut(buf[-len(lp):, -1]) + len(buf) - len(lp)
            decode(buf[:cut], start); start += cut; pending = [buf[cut:]]
        buf = np.concatenate(pending)
        if len(buf): decode(buf, start)
        return dict(text=' '.join(texts), words=words, acoustic_seconds=acoustic, decoder_seconds=decoding)

    def transcribe(self, audio, sample_rate=None, *, hotwords=None, itn=True, chunk_seconds=None, overlap_seconds=None):
        """Speech -> {'text', 'words': [{'word', 'start', 'end'}], 'duration_seconds', 'inference_seconds', 'realtime_factor', ...}.

        audio        a file path (WAV / FLAC / OGG directly, other formats through FFmpeg) or a waveform array (mono or
                     [samples, channels]) at `sample_rate` (default 16000)
        hotwords     names / words expected in the audio, favoured while decoding, e.g. ['Eray', 'Hasan']
        itn          write spoken numbers as digits ('bin dokuz yüz doksan beş' -> '1995'); a number run is one timed word
        chunk_seconds, overlap_seconds   acoustic window length and overlap for audio longer than 30 s (default: the model's
                     LONG_WINDOWS; audio up to 30 s is one window); overlap_seconds=0 cuts fixed chunks (30 s unless chunk_seconds)
                     decoded one by one, as balbal 0.1.0 did"""
        if isinstance(audio, (str, os.PathLike)):
            wave = read_audio(audio)
        else:
            wave = canonical_audio(audio, sample_rate or SAMPLE_RATE)
        if isinstance(hotwords, str): hotwords = [w.strip() for w in hotwords.split(',') if w.strip()]
        window, overlap = LONG_WINDOWS.get(self.model, (30., 5.))
        if overlap_seconds is None: overlap_seconds = overlap
        if chunk_seconds is None: chunk_seconds = window if overlap_seconds > 0 else 30.
        if len(wave) <= SINGLE_PASS_SECONDS * SAMPLE_RATE:   # up to 30 s: one acoustic pass over the whole clip, as before
            chunk_seconds = SINGLE_PASS_SECONDS
        if overlap_seconds > 0:
            out = self.transcribe_long(wave, hotwords or None, chunk_seconds, overlap_seconds)
            elapsed, text, words = out['acoustic_seconds'] + out['decoder_seconds'], out['text'], out['words']
        else:
            step = int(chunk_seconds * SAMPLE_RATE)
            chunks = [self.transcribe_chunk(wave[i:i + step], i / SAMPLE_RATE, hotwords or None) for i in range(0, len(wave), step)]
            elapsed = sum(c['acoustic_seconds'] + c['decoder_seconds'] for c in chunks)
            text, words = ' '.join(c['text'] for c in chunks if c['text']), [w for c in chunks for w in c['words']]
        if itn: text, words = itn_text(text), itn_words(words)
        return dict(text=text, words=words, model=self.model, runtime=self.runtime, duration_seconds=len(wave) / SAMPLE_RATE,
                    inference_seconds=elapsed, realtime_factor=len(wave) / SAMPLE_RATE / elapsed if elapsed else None)


def silence_cut(blank_log_probs, confident=0.99):
    """Where to split a long emission matrix between two decoder runs: the middle of the longest run of confident blank frames
    (p(blank) > confident), else the most blank frame. Index into the given frames, at least 1."""
    quiet = np.concatenate([[False], blank_log_probs > np.log(confident), [False]])
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    if len(edges):
        runs = edges.reshape(-1, 2); a, b = runs[np.argmax(runs[:, 1] - runs[:, 0])]
        return max(1, int((a + b) // 2))
    return max(1, int(np.argmax(blank_log_probs)))


_loaded = {}


def load(model='balbal-v2', **kwargs):
    """balbal.Model(model, **kwargs); downloads the model from Hugging Face on first use."""
    return Model(model, **kwargs)


def transcribe(audio, sample_rate=None, *, model='balbal-v2', hotwords=None, itn=True, **kwargs):
    """One-call transcription; the model is loaded once per process (per model and options) and kept."""
    key = (hub.model_name(model), tuple(sorted(kwargs.items())))
    if key not in _loaded: _loaded[key] = Model(model, **kwargs)
    return _loaded[key].transcribe(audio, sample_rate, hotwords=hotwords, itn=itn)
