"""Model files from Hugging Face: erayyapagci/balbal-v2 and erayyapagci/balbal-v2-manas, downloaded once into
HF_HOME/balbal/<model>/<revision> (default ~/.cache/huggingface/balbal; BALBAL_CACHE or cache_dir= moves it). Only what the CPU runtime
needs is fetched: the ONNX graph of the chosen runtime and the KenLM. The files are plain copies, not hub-cache symlinks: onnxruntime
refuses external weight data that resolves outside the graph's directory.

The revisions are pinned to the weights this release was tested with (decoder settings in configs/ belong to them); revision='main'
takes the latest upload instead.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

MODELS = ('balbal-v2', 'balbal-v2-manas')
ALIASES = {'v2': 'balbal-v2', 'balbal': 'balbal-v2', 'manas': 'balbal-v2-manas', 'v2-manas': 'balbal-v2-manas'}
REPOS = {'balbal-v2': 'erayyapagci/balbal-v2', 'balbal-v2-manas': 'erayyapagci/balbal-v2-manas'}
REVISIONS = {'balbal-v2': '2729f924ea12e2cc236eb7350d11c0c83ee3fa6a', 'balbal-v2-manas': 'c651a680a9147b0a7eb3fd7f5f4f914f3f4ddc25'}
COMMON = ['model.json', 'lm/model.binary', 'lm/tokenizer.model']
GRAPHS = {'fast': ['onnx/model.onnx', 'onnx/model.onnx.data'], 'onnx': ['onnx/model-plain.onnx', 'onnx/model-plain.onnx.data']}
SIZES = {'balbal-v2': {'fast': '590 MB', 'onnx': '680 MB'}, 'balbal-v2-manas': {'fast': '1.1 GB', 'onnx': '1.6 GB'}}


def model_name(name):
    """'balbal-v2' / 'balbal-v2-manas' or a short alias ('v2', 'manas')."""
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    if key not in MODELS:
        raise ValueError(f'unknown model {name!r} (one of {", ".join(MODELS)}; short: v2, manas)')
    return key


def cache_root(cache_dir=None) -> Path:
    if cache_dir or os.getenv('BALBAL_CACHE'): return Path(cache_dir or os.getenv('BALBAL_CACHE')).expanduser()
    from huggingface_hub import constants
    return Path(constants.HF_HOME) / 'balbal'


def download(model='balbal-v2', runtime='fast', revision=None, cache_dir=None, quiet=False) -> Path:
    """Local directory with the model's files for `runtime` ('fast' or 'onnx'); downloads what is missing.
    A pinned revision already on disk is used without contacting Hugging Face."""
    from huggingface_hub import snapshot_download
    name = model_name(model)
    if runtime not in GRAPHS: raise ValueError(f'runtime must be one of {tuple(GRAPHS)}')
    revision = revision or REVISIONS[name]
    files = COMMON + GRAPHS[runtime]
    local = cache_root(cache_dir) / name / revision
    pinned = len(revision) == 40 and all(c in '0123456789abcdef' for c in revision)
    if pinned and all((local / f).is_file() for f in files): return local
    if not quiet and not all((local / f).is_file() for f in files):
        print(f'balbal: downloading {name} ({runtime} runtime, ~{SIZES[name][runtime]}, once) from huggingface.co/{REPOS[name]} '
              f'to {local}', file=sys.stderr, flush=True)
    return Path(snapshot_download(repo_id=REPOS[name], revision=revision, allow_patterns=files, local_dir=local))
