"""Turkish speech-to-text with PyTorch/NeMo: audio -> Conformer (3 refinement passes + acoustic memory) -> KenLM -> text.

The weights are not part of this repository. MODEL_DIR must contain:
  conformer.pt                         PyTorch state dict
  tokenizer/{tokenizer.model, tokenizer.vocab, vocab.txt}   SentencePiece (1024 pieces)
  lm/{model.binary, tokenizer.model}   6-gram token KenLM
Architecture and decoder settings come from ../configs.

  python transcribe.py --model-dir MODEL_DIR command.wav --device cpu
  python transcribe.py --model-dir MODEL_DIR a.flac b.mp3 --device cuda --text-only
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CONFIGS = ROOT.parent / 'configs'
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

import numpy as np
from decoder import KenLMDecoder


def canonical_audio(audio, rate):
    """Mono float32 at 16 kHz with the same resampler as training (scipy polyphase); differs slightly from FFmpeg's."""
    from math import gcd
    from scipy.signal import resample_poly
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != 16000:
        d = gcd(int(rate), 16000)
        audio = resample_poly(audio, 16000 // d, int(rate) // d)
    return np.asarray(audio, dtype=np.float32)


def read_audio(path):
    """WAV/FLAC/OGG via soundfile at the native rate; anything else (MP3/M4A/video...) via FFmpeg at 48 kHz.
    Either way the final resampling to 16 kHz is the training one."""
    path = Path(path).expanduser().resolve(strict=True)
    try:
        import soundfile as sf
        audio, rate = sf.read(str(path), dtype='float32')
    except Exception:
        result = subprocess.run([
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-protocol_whitelist', 'file,pipe', '-i', str(path), '-map', '0:a:0',
            '-vn', '-ac', '1', '-ar', '48000', '-f', 'f32le', 'pipe:1',
        ], capture_output=True, timeout=600)
        if result.returncode:
            raise ValueError('Audio conversion failed: ' + result.stderr.decode(errors='replace')[-1000:])
        audio, rate = np.frombuffer(result.stdout, dtype='<f4').copy(), 48000
    wave = canonical_audio(audio, rate)
    if not wave.size or not np.isfinite(wave).all():
        raise ValueError('Audio is empty or contains nonfinite samples')
    return wave


def build_model(model_dir, *, weights=True):
    """NeMo Conformer-CTC from configs/model_config.yaml + loop-refinement/memory head from configs/refiner.json.
    With weights=True, loads MODEL_DIR/conformer.pt strictly."""
    import torch
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import EncDecCTCModelBPE
    from nemo.utils import logging
    from balbal_model.conformer_loop_memory import attach_loop_memory
    logging.set_verbosity(logging.ERROR)
    model_dir = Path(model_dir)
    cfg = OmegaConf.load(CONFIGS / 'model_config.yaml')
    for key, value in list(cfg.tokenizer.items()):
        if isinstance(value, str) and (value == 'tokenizer' or value.startswith('tokenizer/')):
            cfg.tokenizer[key] = str(model_dir / value)
    model = EncDecCTCModelBPE(cfg=cfg)
    refiner = json.loads((CONFIGS / 'refiner.json').read_text())
    attach_loop_memory(model, refiner['config'], memory_config=refiner['memory'])
    if weights:
        state = torch.load(model_dir / 'conformer.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    model.ctc_extra.inference_loops = refiner['inference_loops']
    return model


class Transcriber:
    """Load once and reuse; one instance must not be called concurrently."""

    def __init__(self, model_dir, device='auto', threads=4):
        import torch
        from nemo.core.classes.mixins import AccessMixin
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.torch = torch
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but this PyTorch environment has no usable GPU')
        self.model = build_model(model_dir).to(self.device).float().eval()
        AccessMixin.set_access_enabled(False, guid=self.model.model_guid)
        self.decoder = KenLMDecoder(Path(model_dir) / 'lm')
        acoustic = self.model.tokenizer.ids_to_tokens(list(range(self.decoder.vocab_size)))
        if acoustic != [self.decoder.sp.id_to_piece(i) for i in range(self.decoder.vocab_size)]:
            raise RuntimeError('Acoustic and LM tokenizer IDs differ')

    def _sync(self):
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()

    def transcribe_waveform(self, wave, *, chunk_seconds=30., hotwords=None):
        """Mono float32 waveform at 16 kHz. Audio longer than 30 s is cut into fixed 30 s chunks (no VAD).
        hotwords: optional names/words to favour (see decoder.KenLMDecoder.decode)."""
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
            raise ValueError('Expected a nonempty, finite mono 16 kHz waveform')
        step = int(chunk_seconds * 16000)
        segments = []
        for start in range(0, len(wave), step):
            samples = wave[start:start + step].copy()
            original = len(samples)
            if len(samples) < 4000:  # the convolutional subsampler needs at least 0.25 s
                samples = np.pad(samples, (0, 4000 - len(samples)))
            self._sync(); tick = time.perf_counter()
            with self.torch.inference_mode():
                x = self.torch.from_numpy(samples)[None].to(self.device)
                lengths = self.torch.tensor([len(samples)], device=self.device)
                log_probs, encoded, *_ = self.model(input_signal=x, input_signal_length=lengths)
                emissions = log_probs[0, :int(encoded[0])].float().cpu().numpy()
            self.model.ctc_extra.clear()
            self._sync(); acoustic = time.perf_counter() - tick
            tick = time.perf_counter(); out = self.decoder.decode_words(emissions, hotwords, start / 16000)
            decoding = time.perf_counter() - tick
            segments.append(dict(start_seconds=start / 16000, end_seconds=(start + original) / 16000, text=out['text'],
                                 words=out['words'],
                                 acoustic_seconds=acoustic, decoder_seconds=decoding))
        elapsed = sum(s['acoustic_seconds'] + s['decoder_seconds'] for s in segments)
        return dict(text=' '.join(s['text'] for s in segments if s['text']), words=[w for s in segments for w in s['words']],
                    duration_seconds=len(wave) / 16000,
                    device=self.device, segments=segments, inference_seconds=elapsed,
                    realtime_speed=(len(wave) / 16000 / elapsed) if elapsed else None)

    def transcribe(self, path, *, chunk_seconds=30., hotwords=None):
        return self.transcribe_waveform(read_audio(path), chunk_seconds=chunk_seconds, hotwords=hotwords)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('audio', type=Path, nargs='+')
    p.add_argument('--model-dir', type=Path, default=Path(os.getenv('BALBAL_MODEL_DIR', ROOT.parent / 'models')))
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--text-only', action='store_true')
    p.add_argument('--hotwords', default='', help='comma-separated names/words to favour, e.g. "Arslan,Zaynep"')
    args = p.parse_args()
    hotwords = [w.strip() for w in args.hotwords.split(',') if w.strip()]
    with redirect_stdout(sys.stderr):  # keep stdout one JSON object per file
        engine = Transcriber(args.model_dir, args.device, args.threads)
    for path in args.audio:
        with redirect_stdout(sys.stderr):
            result = engine.transcribe(path, hotwords=hotwords)
        print(result['text'] if args.text_only else json.dumps({'file': str(path), **result}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
