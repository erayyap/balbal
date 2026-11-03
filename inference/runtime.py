"""Acoustic model runtimes + KenLM decoder for balbal-v2, balbal-v2-manas and models trained on their architectures.

MODEL_DIR (the weights are published separately):
  model.json                         {"model": "balbal-v2"} or {"model": "balbal-v2-manas"}, optionally "languages", "itn", "name",
                                     "long_windows" (balbal.models.describe)
  decoder.json, refiner.json, model_config.yaml   optional: replace the family's configs/<family>/ files
  model.pt                           PyTorch state dict                                   (runtime torch, ONNX export)
  tokenizer/{tokenizer.model, tokenizer.vocab, vocab.txt}                                 (runtime torch, ONNX export)
  lm/{model.binary, tokenizer.model} 6-gram token KenLM
  onnx/model.onnx (+ .data)          fast graph: custom operators from inference/kernels  (runtime fast, the default)
  onnx/model-plain.onnx (+ .data)    the same model with standard operators only          (runtime onnx)

Runtimes (all CPU unless runtime torch gets device cuda):
  fast   lean batch-1 graph with the balbal custom operators (fused product-key memory read with an int8 value table, rel-pos
         attention score); needs inference/kernels/libbalbal_ops.so (or BALBAL_OPS_LIB)
  onnx   lean batch-1 graph with standard onnxruntime operators only (no build step besides the decoder)
  torch  the PyTorch/NeMo model (needs torch + NeMo)
All three give [frames, V + 1] natural-log CTC probabilities (40 ms frames, blank last) to decoder.KenLMDecoder.
Audio longer than 30 s: overlapping windows (30 s windows, 10 s overlap); each window keeps only
the frames up to the middle of its overlaps (its edge frames saw little context), and since CTC frames sit on a fixed 40 ms grid the
kept frames join into one emission matrix for the whole file.
The KenLM search then runs over that, so no word is split at a window edge and the LM context carries on. Very long files are decoded in
pieces of about SEGMENT_SECONDS, cut in the middle of the longest confident-blank (silent) run near the end of each piece, so memory stays
bounded. overlap_seconds=0 gives the old behaviour: fixed chunks, each decoded on its own.
The models write numbers as words; transcribe() turns spoken number runs into digits (ITN) for models whose primary language has an
ITN (Turkish), unless itn=False.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from decoder import KenLMDecoder
from balbal import lang as L
from balbal import models as M

RUNTIMES = ('fast', 'onnx', 'torch')
OPS_LIB = Path(os.getenv('BALBAL_OPS_LIB', HERE / 'kernels/libbalbal_ops.so'))
MIN_SAMPLES = 4000   # the convolutional subsampler needs at least 0.25 s
HOP = 640            # samples per output frame (10 ms hop x 4 subsampling): frame j of a window at sample s = file frame s / HOP + j
SEGMENT_SECONDS = 300.
# (window, overlap) seconds per model, chosen on real long recordings (12 whole phone channels, 3.9 h; WER, fixed 30 s chunks ->
# stitched): balbal-v2 22.31 -> 21.63 (15 s / 3 s: 23.06; 20 s / 8 s: 22.04), manas 19.38 -> 19.11 (30 s / 5 s: 19.02).
SINGLE_PASS_SECONDS = 30.   # clips up to this length are never windowed
# Both families use 30 s / 10 s (balbal.models.DEFAULT_LONG_WINDOWS); model.json "long_windows" overrides it.


def model_name(model_dir, name=None):
    """The architecture family: --model if given, else MODEL_DIR/model.json, else balbal-v2."""
    return M.describe(model_dir, name)['model']


def ort_session(path, threads, custom_ops=False):
    import onnxruntime as ort
    o = ort.SessionOptions(); o.intra_op_num_threads = threads; o.inter_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.enable_mem_pattern = False   # every utterance has another length
    if custom_ops:
        if not OPS_LIB.exists(): raise FileNotFoundError(f'{OPS_LIB} is missing: run inference/kernels/build.sh (or set BALBAL_OPS_LIB)')
        o.register_custom_ops_library(str(OPS_LIB))
    return ort.InferenceSession(str(path), o, providers=['CPUExecutionProvider'])


class Engine:
    """Load once and reuse; one instance must not be called concurrently (make one per thread)."""

    def __init__(self, model_dir, model=None, runtime='fast', threads=None, device='cpu', decoder='fast', settings=None, lm_dir=None):
        if runtime not in RUNTIMES: raise ValueError(f'runtime must be one of {RUNTIMES}')
        self.model_dir, self.runtime = Path(model_dir), runtime
        self.info = M.describe(model_dir, model)
        self.model, self.name, self.languages = self.info['model'], self.info['name'], self.info['languages']
        self.language = L.get(self.languages[0])
        self.itn = L.itn_functions(self.info['itn'])
        self.threads = int(threads or os.getenv('ASR_THREADS', '8'))
        self.settings = dict(M.decoder_config(self.model, self.model_dir), **(settings or {}))
        if runtime == 'torch':
            import torch
            torch.set_num_threads(self.threads); torch.set_float32_matmul_precision('highest')
            self.torch, self.device = torch, device
            self.net = M.build_model(self.model, self.model_dir / 'tokenizer', self.model_dir / 'model.pt', device=device,
                                     model_dir=self.model_dir)
        else:
            graph = self.model_dir / 'onnx' / ('model.onnx' if runtime == 'fast' else 'model-plain.onnx')
            self.session = ort_session(graph, self.threads, custom_ops=runtime == 'fast')
        self.decoder = KenLMDecoder(Path(lm_dir) if lm_dir else self.model_dir / 'lm', self.settings, fast=decoder == 'fast',
                                    language=self.language)
        self.forward(np.zeros(16000, np.float32))   # warm-up

    def forward(self, wave):
        """Mono float32 16 kHz waveform (<= 30 s) -> [frames, V + 1] log-probabilities."""
        wave = np.ascontiguousarray(wave, dtype=np.float32)
        if len(wave) < MIN_SAMPLES: wave = np.pad(wave, (0, MIN_SAMPLES - len(wave)))
        if self.runtime != 'torch':
            return self.session.run(None, {'audio': wave[None]})[0]
        torch = self.torch
        with torch.inference_mode():
            lp, n, _ = self.net(input_signal=torch.from_numpy(wave)[None].to(self.device),
                                input_signal_length=torch.tensor([len(wave)], device=self.device))
        return lp[0, :int(n[0])].float().cpu().numpy()

    def transcribe_chunk(self, wave, offset_seconds=0., hotwords=None):
        """One chunk (<= 30 s) -> {'text', 'words', 'acoustic_seconds', 'decoder_seconds'}; word times shifted by offset_seconds."""
        t0 = time.perf_counter(); lp = self.forward(wave); t1 = time.perf_counter()
        out = self.decoder.decode_words(lp, hotwords, offset_seconds)
        return dict(out, acoustic_seconds=t1 - t0, decoder_seconds=time.perf_counter() - t1)

    def windows(self, wave, chunk_seconds=30., overlap_seconds=5.):
        """Overlapping windows over a long waveform -> (first file frame, [frames, V + 1] log-probabilities) per window, kept frames only;
        the pieces are contiguous on the file's 40 ms frame grid. Also the acoustic seconds spent."""
        size = max(HOP, int(chunk_seconds * 16000) // HOP * HOP)
        overlap = min(int(overlap_seconds * 16000) // HOP * HOP, size // 2)
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

    def transcribe(self, wave, chunk_seconds=None, hotwords=None, itn=True, overlap_seconds=None):
        """Mono float32 waveform at 16 kHz -> {'text', 'words', 'duration_seconds', 'inference_seconds', 'realtime_factor'}.
        itn: write spoken numbers as digits ('bin dokuz yüz doksan beş' -> '1995'), with the model's ITN (model.json "itn", default
        the primary language's: Turkish has one, other languages none, so for them itn changes nothing); a number run becomes one
        timed word.
        chunk_seconds, overlap_seconds: acoustic window length and overlap for audio longer than 30 s (default: the model's long_windows);
        overlap_seconds=0 cuts fixed chunks (30 s unless chunk_seconds) decoded one by one, as before 2026-10-30."""
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
            raise ValueError('Expected a nonempty, finite mono 16 kHz waveform')
        window, overlap = self.info['long_windows']
        if overlap_seconds is None: overlap_seconds = overlap
        if chunk_seconds is None: chunk_seconds = window if overlap_seconds > 0 else 30.
        if len(wave) <= SINGLE_PASS_SECONDS * 16000:   # up to 30 s: one acoustic pass over the whole clip
            chunk_seconds = SINGLE_PASS_SECONDS
        if overlap_seconds > 0:
            out = self.transcribe_long(wave, hotwords or None, chunk_seconds, overlap_seconds)
            elapsed, text, words = out['acoustic_seconds'] + out['decoder_seconds'], out['text'], out['words']
        else:
            step = int(chunk_seconds * 16000)
            chunks = [self.transcribe_chunk(wave[i:i + step], i / 16000, hotwords) for i in range(0, len(wave), step)]
            elapsed = sum(c['acoustic_seconds'] + c['decoder_seconds'] for c in chunks)
            text, words = ' '.join(c['text'] for c in chunks if c['text']), [w for c in chunks for w in c['words']]
        if itn and self.itn: text, words = self.itn[0](text), self.itn[1](words)
        return dict(text=text, words=words,
                    model=self.name, runtime=self.runtime, duration_seconds=len(wave) / 16000, inference_seconds=elapsed,
                    realtime_factor=len(wave) / 16000 / elapsed if elapsed else None)



def silence_cut(blank_log_probs, confident=0.99):
    """Where to split a long emission matrix between two decoder runs: the middle of the longest run of confident blank frames
    (p(blank) > confident), else the most blank frame. Index into the given frames, at least 1."""
    quiet = np.concatenate([[False], blank_log_probs > np.log(confident), [False]])
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    if len(edges):
        runs = edges.reshape(-1, 2); a, b = runs[np.argmax(runs[:, 1] - runs[:, 0])]
        return max(1, int((a + b) // 2))
    return max(1, int(np.argmax(blank_log_probs)))