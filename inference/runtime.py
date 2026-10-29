"""Acoustic model runtimes + KenLM decoder for balbal-v2 and balbal-v2-manas.

MODEL_DIR (the weights are published separately):
  model.json                         {"model": "balbal-v2"} or {"model": "balbal-v2-manas"}
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
Audio longer than 30 s is cut into fixed 30 s chunks (no VAD); each chunk is decoded on its own.
The models write numbers as words; transcribe() turns spoken number runs into digits (ITN) unless itn=False.
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
from balbal import models as M
from balbal.normalizer.itn import itn_text, itn_words

RUNTIMES = ('fast', 'onnx', 'torch')
OPS_LIB = Path(os.getenv('BALBAL_OPS_LIB', HERE / 'kernels/libbalbal_ops.so'))
MIN_SAMPLES = 4000   # the convolutional subsampler needs at least 0.25 s


def model_name(model_dir, name=None):
    """--model if given, else MODEL_DIR/model.json, else balbal-v2."""
    if name: return M.check_name(name)
    f = Path(model_dir) / 'model.json'
    return M.check_name(json.loads(f.read_text())['model']) if f.exists() else 'balbal-v2'


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
        self.model = model_name(model_dir, model)
        self.threads = int(threads or os.getenv('ASR_THREADS', '8'))
        self.settings = dict(M.decoder_config(self.model), **(settings or {}))
        if runtime == 'torch':
            import torch
            torch.set_num_threads(self.threads); torch.set_float32_matmul_precision('highest')
            self.torch, self.device = torch, device
            self.net = M.build_model(self.model, self.model_dir / 'tokenizer', self.model_dir / 'model.pt', device=device)
        else:
            graph = self.model_dir / 'onnx' / ('model.onnx' if runtime == 'fast' else 'model-plain.onnx')
            self.session = ort_session(graph, self.threads, custom_ops=runtime == 'fast')
        self.decoder = KenLMDecoder(Path(lm_dir) if lm_dir else self.model_dir / 'lm', self.settings, fast=decoder == 'fast')
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

    def transcribe(self, wave, chunk_seconds=30., hotwords=None, itn=True):
        """Mono float32 waveform at 16 kHz -> {'text', 'words', 'duration_seconds', 'inference_seconds', 'realtime_factor'}.
        itn: write spoken numbers as digits ('bin dokuz yüz doksan beş' -> '1995'); a number run becomes one timed word."""
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
            raise ValueError('Expected a nonempty, finite mono 16 kHz waveform')
        step = int(chunk_seconds * 16000)
        chunks = [self.transcribe_chunk(wave[i:i + step], i / 16000, hotwords) for i in range(0, len(wave), step)]
        elapsed = sum(c['acoustic_seconds'] + c['decoder_seconds'] for c in chunks)
        text, words = ' '.join(c['text'] for c in chunks if c['text']), [w for c in chunks for w in c['words']]
        if itn: text, words = itn_text(text), itn_words(words)
        return dict(text=text, words=words,
                    model=self.model, runtime=self.runtime, duration_seconds=len(wave) / 16000, inference_seconds=elapsed,
                    realtime_factor=len(wave) / 16000 / elapsed if elapsed else None)
