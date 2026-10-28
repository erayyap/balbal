"""Audio input shared by inference and training: mono float32 at 16 kHz, resampled with the training resampler (scipy polyphase).

read_audio(path)   WAV/FLAC/OGG via soundfile at the native rate; any other format (MP3, M4A, video, ...) via FFmpeg at 48 kHz.
pcm_hash(audio)    sha256 of the canonical PCM16 samples: the audio identity of a manifest row (stable across lossless containers).
"""
import functools
import hashlib
import math
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


@functools.lru_cache(None)
def _fir(up, down, dtype):
    from scipy.signal import firwin
    # resample_poly's default filter (Kaiser 5.0, half length 10*max(up, down)), designed once instead of per call
    return firwin(20 * max(up, down) + 1, 1. / max(up, down), window=('kaiser', 5.0)).astype(dtype)


def resample(x, up, down):
    """resample_poly(x, up, down) with its default filter designed once per (up, down, dtype); bit-identical."""
    from scipy.signal import resample_poly
    g = math.gcd(up, down); up, down = up // g, down // g
    if up == down or x.dtype.kind not in 'fc': return resample_poly(x, up, down)
    return resample_poly(x, up, down, window=_fir(up, down, x.dtype))


def canonical_audio(audio, rate):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all() or rate <= 0:
        raise ValueError('empty, malformed, or nonfinite audio')
    if rate != SAMPLE_RATE:
        audio = resample(audio, SAMPLE_RATE, int(rate))
    return np.asarray(audio, dtype=np.float32)


def read_audio(path):
    path = Path(path).expanduser()
    try:
        import soundfile as sf
        audio, rate = sf.read(str(path), dtype='float32')
    except Exception:
        result = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-protocol_whitelist', 'file,pipe',
                                 '-i', str(path.resolve(strict=True)), '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '48000',
                                 '-f', 'f32le', 'pipe:1'], capture_output=True, timeout=600)
        if result.returncode:
            raise ValueError('Audio conversion failed: ' + result.stderr.decode(errors='replace')[-1000:])
        audio, rate = np.frombuffer(result.stdout, dtype='<f4').copy(), 48000
    return canonical_audio(audio, rate)


def pcm_hash(audio):
    audio = canonical_audio(audio, SAMPLE_RATE)
    pcm = np.rint(np.clip(audio, -1, 32767 / 32768) * 32768).astype('<i2')
    return hashlib.sha256(pcm.tobytes()).hexdigest()
