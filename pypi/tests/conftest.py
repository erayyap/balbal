"""Tests: `pytest` runs the unit tests (no model needed). `BALBAL_TEST_MODELS=1 pytest` also runs the model tests, which download
balbal-v2 (~590 MB, once; BALBAL_TEST_MODEL picks another model). BALBAL_TEST_AUDIO=a folder of speech clips adds checks on real
speech (no audio is kept in this repository)."""
import os
from pathlib import Path

import numpy as np
import pytest


def pytest_collection_modifyitems(config, items):
    if os.getenv('BALBAL_TEST_MODELS') == '1': return
    skip = pytest.mark.skip(reason='model test: set BALBAL_TEST_MODELS=1')
    for item in items:
        if 'model' in item.keywords: item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line('markers', 'model: needs the downloaded model')


@pytest.fixture(scope='session')
def model():
    import balbal
    return balbal.load(os.getenv('BALBAL_TEST_MODEL', 'balbal-v2'), threads=int(os.getenv('BALBAL_THREADS', '4')))


@pytest.fixture(scope='session')
def speech():
    """Real speech clips from BALBAL_TEST_AUDIO (up to 8), else a synthetic signal: tones and noise bursts between silences
    (the decoder checks below hold for any input)."""
    from balbal.audio import read_audio
    folder = os.getenv('BALBAL_TEST_AUDIO')
    if folder:
        files = sorted(f for f in Path(folder).iterdir() if f.suffix.lower() in ('.wav', '.flac', '.ogg', '.mp3', '.m4a'))[:8]
        return [read_audio(f) for f in files]
    rng = np.random.default_rng(0); t = np.arange(16000) / 16000
    parts = []
    for k in range(6):
        parts += [np.zeros(4000), 0.3 * np.sin(2 * np.pi * (200 + 90 * k) * t) * np.hanning(16000), np.zeros(3000),
                  0.1 * rng.standard_normal(6000)]
    return [np.concatenate(parts).astype(np.float32)]
