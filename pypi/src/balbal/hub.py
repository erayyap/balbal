"""Model files from Hugging Face, downloaded once into HF_HOME/balbal/<model>/<revision> (default ~/.cache/huggingface/balbal;
BALBAL_CACHE or cache_dir= moves it). Only what the CPU runtime needs is fetched: the ONNX graph of the chosen runtime and the KenLM
(plus model.json / decoder.json). The files are plain copies, not hub-cache symlinks: onnxruntime refuses external weight data that
resolves outside the graph's directory.

A model is one of:
  a registry name   balbal-v2 / balbal-v2-manas (short: v2, manas): erayyapagci/balbal-v2(-manas), revisions pinned to the weights this
                    release was tested with (decoder settings in configs/ belong to them); revision='main' takes the latest upload
  a local folder    model.json, onnx/, lm/ (and optionally decoder.json), used as it is
  a Hugging Face repo id   'owner/name' of a model trained on a balbal architecture, same layout as a local folder (revision default main)
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
EXTRA = ['decoder.json']   # fetched for other repos when present: their own decoder settings
GRAPHS = {'fast': ['onnx/model.onnx', 'onnx/model.onnx.data'], 'onnx': ['onnx/model-plain.onnx', 'onnx/model-plain.onnx.data']}
SIZES = {'balbal-v2': {'fast': '590 MB', 'onnx': '680 MB'}, 'balbal-v2-manas': {'fast': '1.1 GB', 'onnx': '1.6 GB'}}


def model_name(name):
    """'balbal-v2' / 'balbal-v2-manas' or a short alias ('v2', 'manas')."""
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    if key not in MODELS:
        raise ValueError(f'unknown model {name!r} (one of {", ".join(MODELS)}; short: v2, manas; or a model folder or a '
                         f'Hugging Face repo id owner/name)')
    return key


def resolve(model):
    """('registry', name) | ('local', Path) | ('repo', 'owner/name') for a model given as a registry name, a folder or a repo id."""
    text = str(model).strip()
    try:
        return 'registry', model_name(text)
    except ValueError:
        pass
    path = Path(text).expanduser()
    if path.is_dir(): return 'local', path
    parts = text.split('/')
    if len(parts) == 2 and all(parts) and not text.startswith(('.', '~')) and all(c.isalnum() or c in '-_.' for p in parts for c in p):
        return 'repo', text
    raise ValueError(f'unknown model {model!r}: not one of {", ".join(MODELS)} (short: v2, manas), not a folder, not an owner/name repo id')


def cache_root(cache_dir=None) -> Path:
    if cache_dir or os.getenv('BALBAL_CACHE'): return Path(cache_dir or os.getenv('BALBAL_CACHE')).expanduser()
    from huggingface_hub import constants
    return Path(constants.HF_HOME) / 'balbal'


def download(model='balbal-v2', runtime='fast', revision=None, cache_dir=None, quiet=False) -> Path:
    """Local directory with the model's files for `runtime` ('fast' or 'onnx'); downloads what is missing.
    model: a registry name or a Hugging Face repo id (see resolve). A pinned revision already on disk is used without contacting
    Hugging Face; another revision ('main') is refreshed, and its copy on disk is used when Hugging Face cannot be reached."""
    from huggingface_hub import snapshot_download
    if runtime not in GRAPHS: raise ValueError(f'runtime must be one of {tuple(GRAPHS)}')
    kind, name = resolve(model)
    if kind == 'local': return name
    if kind == 'repo': return download_repo(name, runtime, revision, cache_dir, quiet)
    revision = revision or REVISIONS[name]
    files = COMMON + GRAPHS[runtime]
    local = cache_root(cache_dir) / name / revision
    pinned = len(revision) == 40 and all(c in '0123456789abcdef' for c in revision)
    if pinned and all((local / f).is_file() for f in files): return local
    if not quiet and not all((local / f).is_file() for f in files):
        print(f'balbal: downloading {name} ({runtime} runtime, ~{SIZES[name][runtime]}, once) from huggingface.co/{REPOS[name]} '
              f'to {local}', file=sys.stderr, flush=True)
    path = Path(snapshot_download(repo_id=REPOS[name], revision=revision, allow_patterns=files, local_dir=local))
    count_download(REPOS[name])
    return path


def download_repo(repo, runtime='fast', revision=None, cache_dir=None, quiet=False) -> Path:
    """The files of another Hugging Face model repo (model.json, onnx/ graph, lm/, decoder.json when present)."""
    from huggingface_hub import snapshot_download
    revision = revision or 'main'
    files = COMMON + GRAPHS[runtime]
    local = cache_root(cache_dir) / repo.replace('/', '--') / revision.replace('/', '--')
    have = all((local / f).is_file() for f in files)
    try:
        from huggingface_hub import list_repo_files
        listed = set(list_repo_files(repo, revision=revision))
    except Exception:   # offline: the copy on disk, if complete
        if have: return local
        raise
    missing = [f for f in files if f not in listed]
    if missing:   # checked before downloading anything (the LM alone can be a gigabyte)
        graph = GRAPHS[runtime][0]
        hint = (f' {repo} has no {graph}: balbal runs ONNX graphs; export the model with inference/export_onnx.py from '
                'https://github.com/erayyap/balbal and upload the onnx/ folder.') if graph in missing else ''
        raise FileNotFoundError(f'{repo}@{revision} lacks {", ".join(missing)}.{hint}')
    if not quiet and not have:
        print(f'balbal: downloading {repo} ({runtime} runtime, once) from huggingface.co/{repo} to {local}', file=sys.stderr, flush=True)
    snapshot_download(repo_id=repo, revision=revision, allow_patterns=files + [f for f in EXTRA if f in listed], local_dir=local)
    return local


def count_download(repo):
    """One HEAD request for config.json on main: the Hub counts model downloads by requests to that file (no data is sent)."""
    try:
        from huggingface_hub import file_exists
        file_exists(repo, 'config.json')
    except Exception:
        pass
