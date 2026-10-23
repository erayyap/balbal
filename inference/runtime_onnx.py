"""balbal-v1 on onnxruntime CPU: no torch or NeMo at run time (this is the runtime behind the published CPU speed).

MODEL_DIR must contain model-ext.onnx + model-ext.data (from export_onnx.py; the weights file is memory-mapped)
and lm/{model.binary, tokenizer.model}. CPU arena, memory patterns and weight prepacking are off to keep RSS low.

  python runtime_onnx.py --model-dir MODEL_DIR speech.wav [more.wav ...]
  python runtime_onnx.py --model-dir MODEL_DIR --hotwords "Arslan,Zaynep" call.wav   # favour these names
  from runtime_onnx import Engine; Engine(MODEL_DIR).transcribe(wave)['text']   # mono float32, 16 kHz
The result also has 'words': [{'word', 'start', 'end'}] with times in seconds (see word_timing.py).
Audio longer than 30 s is cut into fixed 30 s chunks (no VAD), as in transcribe.py.
Live transcription of a growing recording: streaming.py.
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
from decoder import KenLMDecoder


class Engine:
    def __init__(self, model_dir, threads=None):
        import onnxruntime as ort
        model_dir = Path(model_dir)
        o = ort.SessionOptions()
        o.intra_op_num_threads = int(threads or os.getenv('ASR_THREADS', '6')); o.inter_op_num_threads = 1
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        o.enable_cpu_mem_arena = False; o.enable_mem_pattern = False
        o.add_session_config_entry('session.disable_prepacking', '1')
        self.session = ort.InferenceSession(str(model_dir / 'model-ext.onnx'), o, providers=['CPUExecutionProvider'])
        self.decoder = KenLMDecoder(model_dir / 'lm')
        self.transcribe(np.zeros(16000, dtype=np.float32))   # warm-up

    def transcribe_chunk(self, wave, offset_seconds=0., hotwords=None):
        """One chunk (<= 30 s) -> {'text', 'words'}; word times are shifted by offset_seconds. Used by streaming.py."""
        if len(wave) < 4000:   # the convolutional subsampler needs at least 0.25 s
            wave = np.pad(wave, (0, 4000 - len(wave)))
        a = self.session.run(None, {'audio': np.ascontiguousarray(wave, dtype=np.float32)[None]})[0]
        return self.decoder.decode_words(a, hotwords, offset_seconds)

    def transcribe(self, wave, chunk_seconds=30., hotwords=None):
        wave = np.asarray(wave, dtype=np.float32)
        start = time.perf_counter(); step = int(chunk_seconds * 16000)
        chunks = [self.transcribe_chunk(wave[i:i + step], i / 16000, hotwords) for i in range(0, len(wave), step)]
        elapsed = time.perf_counter() - start
        return {'text': ' '.join(c['text'] for c in chunks if c['text']), 'words': [w for c in chunks for w in c['words']],
                'duration_seconds': len(wave) / 16000,
                'inference_seconds': elapsed, 'realtime_factor': len(wave) / 16000 / elapsed}


def main():
    from transcribe import read_audio
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('audio', type=Path, nargs='+')
    p.add_argument('--model-dir', type=Path, default=Path(os.getenv('BALBAL_MODEL_DIR', ROOT.parent / 'models')))
    p.add_argument('--threads', type=int, default=None)
    p.add_argument('--hotwords', default='', help='comma-separated names/words to favour, e.g. "Arslan,Zaynep"')
    args = p.parse_args()
    hotwords = [w.strip() for w in args.hotwords.split(',') if w.strip()]
    engine = Engine(args.model_dir, args.threads)
    for path in args.audio:
        print(json.dumps({'file': str(path), **engine.transcribe(read_audio(path), hotwords=hotwords)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
