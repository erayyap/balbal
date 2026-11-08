"""CPU speech recognition with balbal-v2 / balbal-v2-manas, or a model trained on their architectures: ONNX acoustic model + KenLM
beam decoder (same path as the inference/ code of github.com/erayyap/balbal).

Runtimes:
  fast   lean batch-1 graph with the balbal onnxruntime custom operators (fused product-key memory read with an int8 value table,
         rel-pos attention score); the operator library ships in this wheel, built for x86-64-v2 / v3 / v4 (AVX-512), picked per CPU
  onnx   the same model with standard onnxruntime operators only (larger download, slower); also the fallback when the operator
         library does not load
Both give [frames, V + 1] natural-log CTC probabilities (40 ms frames, blank last) to decoder.KenLMDecoder.
Audio longer than 30 s: overlapping windows (30 s windows, 10 s overlap; model.json "long_windows"); each window keeps only
the frames up to the middle of its overlaps (its edge frames saw little context), and since CTC frames sit on a fixed 40 ms grid the
kept frames join into one emission matrix for the whole file.
The KenLM search then runs over that, so no word is split at a window edge and the LM context carries on. Very long files are decoded in
pieces of about SEGMENT_SECONDS, cut in the middle of the longest confident-blank (silent) run near the end of each piece, so memory stays
bounded. overlap_seconds=0 gives the old behaviour: fixed chunks, each decoded on its own.
The models write numbers as words; transcribe() turns spoken number runs into digits (ITN) for models with an ITN (Turkish) unless
itn=False.

A model folder's model.json: {"model": family, "name", "languages", "itn", "long_windows", "frame_samples", "license"...} (model_info);
decoder.json in the folder replaces the packaged decoder settings of its family. frame_samples: samples per output frame (640 = 40 ms, the
default; balbal-v2e-zellige has 1280 = 80 ms frames; live streaming needs 640). Models in ONNX_ONLY (balbal-v2e-zellige) are one plain
onnx/model.onnx graph: every runtime setting loads it with standard operators.
Live audio: Model.stream() (balbal.streaming).
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np

from . import hub
from .audio import SAMPLE_RATE, canonical_audio, read_audio
from . import lang
from .decoder import KenLMDecoder

HERE = Path(__file__).resolve().parent
RUNTIMES = ('fast', 'onnx')
MIN_SAMPLES = 4000   # the convolutional subsampler needs at least 0.25 s
HOP = 640            # samples per output frame (10 ms hop x 4 subsampling): frame j of a window at sample s = file frame s / HOP + j (default; model.json frame_samples)
SEGMENT_SECONDS = 300.
# (window, overlap) seconds per model, chosen on real long recordings (12 whole phone channels, 3.9 h; WER, fixed 30 s chunks ->
# stitched): balbal-v2 22.31 -> 21.63 (15 s / 3 s: 23.06; 20 s / 8 s: 22.04), manas 19.38 -> 19.11 (30 s / 5 s: 19.02).
SINGLE_PASS_SECONDS = 30.   # clips up to this length are never windowed
# No word at or above this confidence -> result['no_speech']. On 38 hand-checked phone clips + synthetic silence / noise / tones /
# clicks: every speech clip has a word >= 0.26, non-speech input stays <= 0.10 (the models can write a short word over digital silence).
NO_SPEECH_CONFIDENCE = 0.15
LONG_WINDOWS = {'balbal-v2': (30., 10.), 'balbal-v2-manas': (30., 10.), 'balbal-v2e-zellige': (30., 10.)}   # family defaults; model.json "long_windows" wins
FAMILY_LANGUAGES = {'balbal-v2': ['tr'], 'balbal-v2-manas': ['tr', 'en'], 'balbal-v2e-zellige': ['ary']}
FRAME_SAMPLES = {'balbal-v2e-zellige': 1280}   # 80 ms output frames (8x subsampling); the other models 640
ONNX_ONLY = tuple(hub.ONNX_ONLY)


def model_info(model_dir, family=None, name=None):
    """model.json of a model folder with the defaults filled in: {'model' (family), 'name', 'languages', 'itn', 'long_windows'}.
    family: used when model.json has no "model"; name: the display name when model.json has no "name" (default: the family)."""
    f = Path(model_dir) / 'model.json'
    info = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    family = hub.model_name(info.get('model') or family or 'balbal-v2')
    languages = info.get('languages') or FAMILY_LANGUAGES[family]
    if isinstance(languages, str): languages = [languages]
    window, overlap = info.get('long_windows') or LONG_WINDOWS[family]
    return dict(model=family, name=info.get('name') or name or family, languages=list(languages),
                itn=(info['itn'] if 'itn' in info else languages[0]) or None, long_windows=(float(window), float(overlap)),
                frame_samples=int(info.get('frame_samples') or FRAME_SAMPLES.get(family, HOP)))


def cpu_level():
    """x86-64 micro-architecture level of this CPU for the operator library: 'v4' (AVX-512), 'v3' (AVX2 + FMA) or 'v2'."""
    if sys.platform != 'linux': return _cpu_level_numpy()
    try:
        flags = set(next(line for line in open('/proc/cpuinfo', encoding='utf-8') if line.startswith('flags')).split(':', 1)[1].split())
    except (OSError, StopIteration):
        return 'v2'
    if {'avx512f', 'avx512bw', 'avx512cd', 'avx512dq', 'avx512vl'} <= flags: return 'v4'
    if {'avx', 'avx2', 'bmi1', 'bmi2', 'f16c', 'fma', 'movbe'} <= flags: return 'v3'
    return 'v2'


def _cpu_level_numpy():
    """Windows: numpy's run-time CPU feature table (CPUID, with the OS's AVX / AVX-512 state support checked)."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as has
    except ImportError:
        try:
            from numpy.core._multiarray_umath import __cpu_features__ as has
        except ImportError:
            return 'v2'
    if has.get('AVX512_SKX'): return 'v4'   # AVX512 F + CD + BW + DQ + VL
    if all(has.get(k) for k in ('AVX', 'AVX2', 'FMA3', 'F16C')): return 'v3'
    return 'v2'


def ops_library():
    """The custom operator library for this CPU (BALBAL_OPS_LIB overrides): libbalbal_ops_{v2,v3,v4}.so on Linux,
    balbal_ops_{v2,v3,v4}.dll on Windows, libbalbal_ops.dylib on macOS (arm64)."""
    if os.getenv('BALBAL_OPS_LIB'): return Path(os.environ['BALBAL_OPS_LIB'])
    if sys.platform == 'darwin': return HERE / '_native' / 'libbalbal_ops.dylib'
    if sys.platform == 'win32': return HERE / '_native' / f'balbal_ops_{cpu_level()}.dll'
    return HERE / '_native' / f'libbalbal_ops_{cpu_level()}.so'


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

    model      'balbal-v2' (default, 45M, Turkish), 'balbal-v2-manas' (313M, Turkish + English) or 'balbal-v2e-zellige' (Moroccan
               Darija, CC BY-NC-SA 4.0 non-commercial; offline only); short: 'v2', 'manas', 'zellige'. Also a
               local model folder or the Hugging Face repo id ('owner/name') of a model trained on a balbal architecture
    runtime    'fast' (default) or 'onnx' (standard operators only)
    threads    CPU threads (default: BALBAL_THREADS, else min(8, CPU count))
    model_dir  a local model folder (model.json, onnx/, lm/; same as passing the folder as model)
    revision   Hugging Face revision (default: the one this release was tested with, or main for other repos)
    settings   decoder settings over decoder.json, e.g. {'domain_lm': 'my-lm', 'domain_weight': 0.3}: a domain KenLM directory (built
               on your own text with this model's tokenizer) mixed into the general LM; also a model folder's own decoder.json
               (a relative domain_lm there is relative to the folder)
    Attributes: model (display name), family (architecture), languages (primary first), source (what to load it again from).
    """

    def __init__(self, model='balbal-v2', runtime='fast', threads=None, model_dir=None, revision=None, cache_dir=None, settings=None):
        if runtime not in RUNTIMES: raise ValueError(f'runtime must be one of {RUNTIMES}')
        if model_dir is not None:
            kind, family = 'local', (hub.model_name(model) if model != 'balbal-v2' else None)
            where, label = Path(model_dir), None
        else:
            kind, where = hub.resolve(model)
            family = where if kind == 'registry' else None
            label = where if kind == 'repo' else None
        self.source = str(where)
        self.threads = int(threads or default_threads())
        self.revision, self.cache_dir = revision, cache_dir
        self.model_dir = where if kind == 'local' else self._download(where, runtime)
        self.info = model_info(self.model_dir, family, label)
        if kind == 'registry': self.info['model'] = family   # the registry name is the family (its model.json says so too)
        self.model, self.family, self.languages = self.info['name'], self.info['model'], self.info['languages']
        self.language = lang.get(self.languages[0])
        self._itn = lang.itn_functions(self.info['itn'])
        self.hop = self.info['frame_samples']            # samples per output frame
        if kind == 'local': hub.license_notice(self.model_dir)
        own = self.model_dir / 'decoder.json'
        packaged = json.loads((HERE / 'configs' / self.family / 'decoder.json').read_text(encoding='utf-8'))
        mine = json.loads(own.read_text(encoding='utf-8')) if kind != 'registry' and own.exists() else {}
        if mine.get('domain_lm') and not Path(mine['domain_lm']).is_absolute(): mine['domain_lm'] = str(self.model_dir / mine['domain_lm'])
        self.settings = {**packaged, **mine, **(settings or {})}   # later wins (dict(a, **b, **c) fails on a shared key)
        self.runtime = runtime = 'onnx' if self.family in ONNX_ONLY else runtime   # one plain graph, no fast graph: no warning, no fallback
        try:
            self.session = self._session(runtime)
        except Exception as error:   # the operator library did not load, or the model has no fast graph: the standard-operator graph
            if runtime != 'fast' or not isinstance(error, OSError): raise
            warnings.warn(f'balbal: the fast runtime did not load ({error}); using the standard-operator graph (runtime="onnx")')
            if kind != 'local': self.model_dir = self._download(where, 'onnx')
            self.runtime, self.session = 'onnx', self._session('onnx')
        self.decoder = KenLMDecoder(self.model_dir / 'lm', self.settings, language=self.language, frame_seconds=self.hop / SAMPLE_RATE)
        self.forward(np.zeros(SAMPLE_RATE, np.float32))   # warm-up

    def _download(self, where, runtime):
        try:
            return hub.download(where, runtime, self.revision, self.cache_dir)
        except FileNotFoundError:   # a repo without the fast graph: the session step falls back to the standard-operator graph
            if runtime != 'fast': raise
            return hub.download(where, 'onnx', self.revision, self.cache_dir)

    def itn(self, text, words):
        """(text, words) with spoken numbers as digits, by the model's ITN (unchanged for a model without one)."""
        return (self._itn[0](text), self._itn[1](words)) if self._itn else (text, words)

    def _session(self, runtime):
        graph = self.model_dir / 'onnx' / ('model.onnx' if runtime == 'fast' or self.family in ONNX_ONLY else 'model-plain.onnx')
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
        HOP = self.hop
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
        frame = self.hop / SAMPLE_RATE   # seconds per output frame
        segment = max(1, int(segment_seconds / frame))
        texts, words, pending, start, acoustic, decoding = [], [], [], 0, 0., 0.

        def decode(lp, first):
            nonlocal decoding
            t = time.perf_counter(); out = self.decoder.decode_words(lp, hotwords, first * frame); decoding += time.perf_counter() - t
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
        """Speech -> {'text', 'words': [{'word', 'start', 'end', 'confidence'}], 'duration_seconds', 'inference_seconds',
        'realtime_factor', 'no_speech', ...}. confidence: 0-1 per word from the acoustic model (low = likely wrong, see word_timing).
        no_speech: no word reaches NO_SPEECH_CONFIDENCE (silence, noise, a short word made up over digital silence); the text is
        left as decoded.

        audio        a file path (WAV / FLAC / OGG directly, other formats through FFmpeg) or a waveform array (mono or
                     [samples, channels]) at `sample_rate` (default 16000)
        hotwords     names / words expected in the audio, favoured while decoding, e.g. ['Eray', 'Hasan']
        itn          write spoken numbers as digits ('bin dokuz yüz doksan beş' -> '1995') with the model's ITN (Turkish; other
                     languages have none, so it changes nothing there); a number run is one timed word
        chunk_seconds, overlap_seconds   acoustic window length and overlap for audio longer than 30 s (default: the model's
                     LONG_WINDOWS; audio up to 30 s is one window); overlap_seconds=0 cuts fixed chunks (30 s unless chunk_seconds)
                     decoded one by one, as balbal 0.1.0 did"""
        if isinstance(audio, (str, os.PathLike)):
            wave = read_audio(audio)
        else:
            wave = canonical_audio(audio, sample_rate or SAMPLE_RATE)
        if isinstance(hotwords, str): hotwords = [w.strip() for w in hotwords.split(',') if w.strip()]
        window, overlap = self.info['long_windows']
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
        if itn: text, words = self.itn(text, words)
        no_speech = not any(w.get('confidence', 1.) >= NO_SPEECH_CONFIDENCE for w in words)
        return dict(text=text, words=words, no_speech=no_speech, model=self.model, runtime=self.runtime, duration_seconds=len(wave) / SAMPLE_RATE,
                    inference_seconds=elapsed, realtime_factor=len(wave) / SAMPLE_RATE / elapsed if elapsed else None)

    def _require_streaming(self):
        if self.hop != HOP: raise NotImplementedError(f'{self.model} has {1000 * self.hop // SAMPLE_RATE} ms output frames; live streaming '
                                                      f'(balbal live, WebSocket /v1/realtime) is built for 40 ms frames (balbal-v2, balbal-v2-manas)')

    def stream(self, mode='segment', *, hotwords=None, itn=True, **options):
        """A live stream: feed audio pieces with .accept(piece), end with .finish(); both return partial / final text events
        (see balbal.streaming). mode 'segment' (default) or 'window'; options: chunk, right, endpoint, max_segment (seconds)."""
        self._require_streaming()
        from .streaming import LiveStream
        return LiveStream(self, mode, hotwords, itn, **options)

    def transcribe_stream(self, pieces, mode='segment', *, hotwords=None, itn=True, **options):
        """Generator of live events for an iterable of audio pieces (float32 or int16, mono, 16 kHz); see stream()."""
        self._require_streaming()
        from .streaming import transcribe_stream
        return transcribe_stream(self, pieces, mode=mode, hotwords=hotwords, itn=itn, **options)


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
    kind, where = hub.resolve(model)
    key = (kind, str(where), tuple(sorted(kwargs.items())))
    if key not in _loaded: _loaded[key] = Model(model, **kwargs)
    return _loaded[key].transcribe(audio, sample_rate, hotwords=hotwords, itn=itn)
