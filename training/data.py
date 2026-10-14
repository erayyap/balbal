"""Training data path of balbal-v1: audio reading, virtual (assembled) rows, augmentation, batching.

Manifest rows (JSON lines):
  {"id", "audio_path", "pcm_sha256", "duration_seconds", "text", "source"}                       one audio file
  {"id", "parts": [{"audio_path", "pcm_sha256", "start", "end", "gap_before"}, ...], "duration_seconds", "text", "source"}
      a virtual row: successive segments of one recording joined on the fly (start/end in 16 kHz samples,
      gap_before = seconds of zeros inserted before the part); no audio is written.
pcm_sha256 = sha256 of the canonical PCM16 samples (pcm_hash below); a mismatch is an error (the audio changed).

Augmentation of TRAIN clips (rows with a non-empty label; evaluation rows are never touched), per clip:
  1. telephone band with probability TEL_P (default 0.5): 16 kHz -> 8 kHz -> 300 Hz high-pass (p .5) -> G.711 mu-law
     round trip (p .5) -> 16 kHz;
  2. edge format with probability EDGE_P (default 0.5): trim silent edges (10 ms frames within 35 dB of the clip's p95
     level) and pad U(0, EDGE_MAX=0.6) s of exact digital zeros on each side; the untrimmed clip is kept when trimming
     would make the label CTC-infeasible.
SpecAugment (2 freq masks of 27, 5 time masks of 5 %) runs inside the NeMo model in training mode.
"""
import hashlib
import math
import os
import zlib

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
TEL_P = float(os.getenv('TEL_P', '0.5'))
EDGE_P = float(os.getenv('EDGE_P', '0.5')); EDGE_MAX = float(os.getenv('EDGE_MAX', '0.6'))
_calls = [0]


def canonical_audio(audio, rate):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all() or rate <= 0:
        raise ValueError('empty, malformed, or nonfinite audio')
    if rate != SAMPLE_RATE:
        divisor = math.gcd(int(rate), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, int(rate) // divisor)
    return np.asarray(audio, dtype=np.float32)


def read_audio(path):
    audio, rate = sf.read(path, dtype='float32')
    return canonical_audio(audio, rate)


def pcm_hash(audio):
    audio = canonical_audio(audio, SAMPLE_RATE)
    # Canonical PCM16 identity, stable across lossless WAV/FLAC containers.
    pcm = np.rint(np.clip(audio, -1, 32767 / 32768) * 32768).astype('<i2')
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def mulaw(x, mu=255.):
    y = np.sign(x)*np.log1p(mu*np.abs(x))/np.log1p(mu); q = np.round((y+1)/2*255)/255*2-1
    return np.sign(q)*np.expm1(np.abs(q)*np.log1p(mu))/mu


def telephone(x, rng):
    from scipy.signal import butter, sosfilt
    y = np.clip(resample_poly(x, 1, 2), -1, 1)
    if rng.random() < .5: y = sosfilt(butter(4, 300, 'highpass', fs=8000, output='sos'), y)
    if rng.random() < .5: y = mulaw(np.clip(y, -1, 1))
    return np.asarray(resample_poly(y, 2, 1)[:len(x)], dtype=np.float32)


def tight(x, db=35., f=160):
    n = len(x)//f
    if n < 3: return x
    e = 10*np.log10((x[:n*f].astype(np.float64).reshape(n, f)**2).mean(1) + 1e-12)
    sp = np.nonzero(e > np.percentile(e, 95) - db)[0]
    return x[int(sp[0])*f:(int(sp[-1])+1)*f] if len(sp) else x


def ctc_frames_needed(label):
    """CTC frames (40 ms each) a label needs: one per token plus a blank between repeated tokens."""
    return len(label) + sum(a == b for a, b in zip(label, label[1:]))


def _checked(path, digest, row_id):
    wave = read_audio(path)
    if pcm_hash(wave) != digest:
        raise ValueError('Audio identity changed: ' + row_id)
    return wave


class Rows(Dataset):
    """(index, waveform, label ids). Labels are token-id lists; an empty label marks an evaluation row."""

    def __init__(self, rows, labels):
        self.rows, self.labels = rows, labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]; label = torch.tensor(self.labels[i], dtype=torch.long)
        if 'parts' in row:
            chunks = []
            for p in row['parts']:
                w = _checked(p['audio_path'], p['pcm_sha256'], row['id'])
                if p.get('gap_before'): chunks.append(np.zeros(int(p['gap_before']*16000), np.float32))
                chunks.append(w[int(p.get('start') or 0):p.get('end')])
            wave = np.concatenate(chunks).astype(np.float32)
        else:
            wave = _checked(row['audio_path'], row['pcm_sha256'], row['id'])
        if len(label):
            # One generator per call: telephone band first, then the edge format.
            _calls[0] += 1; rng = np.random.default_rng([i, _calls[0], os.getpid(), zlib.crc32(str(id(self)).encode())])
            if rng.random() < TEL_P: wave = telephone(wave, rng)
            if rng.random() < EDGE_P:
                x = tight(wave)
                if ctc_frames_needed(label.tolist()) > len(x)//640 - 2: x = wave   # keep CTC-feasible
                z = lambda: np.zeros(int(rng.uniform(0, EDGE_MAX)*16000), np.float32)
                wave = np.concatenate([z(), x, z()]).astype(np.float32)
        return i, torch.from_numpy(np.ascontiguousarray(wave)), label


def collate(items):
    ids, waves, labels = zip(*items)
    return (list(ids), torch.nn.utils.rnn.pad_sequence(waves, batch_first=True),
            torch.tensor([len(x) for x in waves]),
            torch.nn.utils.rnn.pad_sequence(labels, batch_first=True),
            torch.tensor([len(x) for x in labels]))
