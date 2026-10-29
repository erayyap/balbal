"""CPU speech recognition with balbal-v2 / balbal-v2-manas: ONNX acoustic model + KenLM beam decoder (same path as the
inference/ code of github.com/erayyap/balbal).

Runtimes:
  fast   lean batch-1 graph with the balbal onnxruntime custom operators (fused product-key memory read with an int8 value table,
         rel-pos attention score); the operator library ships in this wheel, built for x86-64-v2 / v3 / v4 (AVX-512), picked per CPU
  onnx   the same model with standard onnxruntime operators only (larger download, slower); also the fallback when the operator
         library does not load
Both give [frames, V + 1] natural-log CTC probabilities (40 ms frames, blank last) to decoder.KenLMDecoder.
Audio longer than 30 s is cut into fixed 30 s chunks (no VAD); each chunk is decoded on its own.
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

    def transcribe(self, audio, sample_rate=None, *, hotwords=None, itn=True, chunk_seconds=30.):
        """Speech -> {'text', 'words': [{'word', 'start', 'end'}], 'duration_seconds', 'inference_seconds', 'realtime_factor', ...}.

        audio        a file path (WAV / FLAC / OGG directly, other formats through FFmpeg) or a waveform array (mono or
                     [samples, channels]) at `sample_rate` (default 16000)
        hotwords     names / words expected in the audio, favoured while decoding, e.g. ['Eray', 'Hasan']
        itn          write spoken numbers as digits ('bin dokuz yüz doksan beş' -> '1995'); a number run is one timed word"""
        if isinstance(audio, (str, os.PathLike)):
            wave = read_audio(audio)
        else:
            wave = canonical_audio(audio, sample_rate or SAMPLE_RATE)
        if isinstance(hotwords, str): hotwords = [w.strip() for w in hotwords.split(',') if w.strip()]
        step = int(chunk_seconds * SAMPLE_RATE)
        chunks = [self.transcribe_chunk(wave[i:i + step], i / SAMPLE_RATE, hotwords or None) for i in range(0, len(wave), step)]
        elapsed = sum(c['acoustic_seconds'] + c['decoder_seconds'] for c in chunks)
        text, words = ' '.join(c['text'] for c in chunks if c['text']), [w for c in chunks for w in c['words']]
        if itn: text, words = itn_text(text), itn_words(words)
        return dict(text=text, words=words, model=self.model, runtime=self.runtime, duration_seconds=len(wave) / SAMPLE_RATE,
                    inference_seconds=elapsed, realtime_factor=len(wave) / SAMPLE_RATE / elapsed if elapsed else None)


_loaded = {}


def load(model='balbal-v2', **kwargs):
    """balbal.Model(model, **kwargs); downloads the model from Hugging Face on first use."""
    return Model(model, **kwargs)


def transcribe(audio, sample_rate=None, *, model='balbal-v2', hotwords=None, itn=True, **kwargs):
    """One-call transcription; the model is loaded once per process (per model and options) and kept."""
    key = (hub.model_name(model), tuple(sorted(kwargs.items())))
    if key not in _loaded: _loaded[key] = Model(model, **kwargs)
    return _loaded[key].transcribe(audio, sample_rate, hotwords=hotwords, itn=itn)
