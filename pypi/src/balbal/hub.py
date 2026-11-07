"""Model files from Hugging Face, downloaded once into HF_HOME/balbal/<model>/<revision> (default ~/.cache/huggingface/balbal;
BALBAL_CACHE or cache_dir= moves it). Only what the CPU runtime needs is fetched: the ONNX graph of the chosen runtime and the KenLM
(plus model.json / decoder.json). The files are plain copies, not hub-cache symlinks: onnxruntime refuses external weight data that
resolves outside the graph's directory.

A model is one of:
  a registry name   balbal-v2 / balbal-v2-manas / balbal-v2.1-zellige (short: v2, manas, zellige): erayyapagci/<name>, revisions pinned
                    to the weights this release was tested with (decoder settings in configs/ belong to them); revision='main' takes
                    the latest upload. balbal-v2.1-zellige (Moroccan Darija) is released under CC BY-NC-SA 4.0 (non-commercial);
                    the others under the balbal Community License 1.1. Its model.json says so ("license"), and the first download
                    / load prints a notice (license_notice)
  a local folder    model.json, onnx/, lm/ (and optionally decoder.json), used as it is
  a Hugging Face repo id   'owner/name' of a model trained on a balbal architecture, same layout as a local folder (revision default main)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

MODELS = ('balbal-v2', 'balbal-v2-manas', 'balbal-v2.1-zellige')
ALIASES = {'v2': 'balbal-v2', 'balbal': 'balbal-v2', 'manas': 'balbal-v2-manas', 'v2-manas': 'balbal-v2-manas',
           'zellige': 'balbal-v2.1-zellige', 'v2.1': 'balbal-v2.1-zellige', 'v2.1-zellige': 'balbal-v2.1-zellige'}
REPOS = {'balbal-v2': 'erayyapagci/balbal-v2', 'balbal-v2-manas': 'erayyapagci/balbal-v2-manas',
         'balbal-v2.1-zellige': 'erayyapagci/balbal-v2.1-zellige'}
REVISIONS = {'balbal-v2': '2729f924ea12e2cc236eb7350d11c0c83ee3fa6a', 'balbal-v2-manas': 'c651a680a9147b0a7eb3fd7f5f4f914f3f4ddc25',
             'balbal-v2.1-zellige': '938a21998c56216386b53280cde0bf6037995cc1'}
# Licence of each model's weights (shown by `balbal --help` and in the first-use notice); the models not listed use the balbal Community License 1.1.
LICENSES = {'balbal-v2': 'balbal Community License 1.1', 'balbal-v2-manas': 'balbal Community License 1.1',
            'balbal-v2.1-zellige': 'CC BY-NC-SA 4.0 (non-commercial)'}
LANGUAGE_NAMES = {'balbal-v2': 'Turkish', 'balbal-v2-manas': 'Turkish, English', 'balbal-v2.1-zellige': 'Moroccan Darija'}
COMMON = ['model.json', 'lm/model.binary', 'lm/tokenizer.model']
EXTRA = ['decoder.json']   # fetched for other repos when present: their own decoder settings
GRAPHS = {'fast': ['onnx/model.onnx', 'onnx/model.onnx.data'], 'onnx': ['onnx/model-plain.onnx', 'onnx/model-plain.onnx.data']}
# Models released as one plain-operator graph only (no custom-operator "fast" graph): every runtime loads onnx/model.onnx (engine.ONNX_ONLY).
ONNX_ONLY = {'balbal-v2.1-zellige': ['onnx/model.onnx']}
SIZES = {'balbal-v2': {'fast': '590 MB', 'onnx': '680 MB'}, 'balbal-v2-manas': {'fast': '1.1 GB', 'onnx': '1.6 GB'},
         'balbal-v2.1-zellige': {'fast': '1.4 GB', 'onnx': '1.4 GB'}}


def graph_files(name, runtime):
    """Graph files of a registry model for a runtime (ONNX-only models have the one plain graph for both)."""
    return ONNX_ONLY.get(name) or GRAPHS[runtime]


def license_notice(model_dir, name=None, stream=None, remember=False):
    """Once per model download (a marker file in the cache folder; once per process for a local folder): if the model's model.json declares
    a non-commercial licence ("license": "cc-by-nc-sa-4.0", "license_name": ..., "license_url": ...), print one notice naming it to stderr.
    remember: a downloaded copy (marker file); False for a local folder. Never blocks and never raises."""
    import json
    try:
        info = json.loads((Path(model_dir) / 'model.json').read_text(encoding='utf-8'))
        spdx = str(info.get('license') or '').lower()
        if not (info.get('non_commercial') or '-nc' in spdx): return False
        marker = Path(model_dir) / '.license-notice'
        if str(Path(model_dir)) in _noticed or marker.exists(): return False
        _noticed.add(str(Path(model_dir)))
        label = info.get('name') or name or Path(model_dir).name
        print(f'balbal: NOTICE: {label} is released under {info.get("license_name") or spdx.upper()}'
              f'{" (" + info["license_url"] + ")" if info.get("license_url") else ""} - non-commercial use only, with attribution and '
              f'share-alike. This is not the balbal Community License 1.1 of balbal-v2 / balbal-v2-manas.', file=stream or sys.stderr, flush=True)
        if remember:
            try: marker.write_text('shown\n', encoding='utf-8')
            except OSError: pass
        return True
    except Exception:
        return False


_noticed = set()


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
    for registry, repo in REPOS.items():   # the registry model's own Hugging Face repo id is the registry model
        if text == repo: return 'registry', registry
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
    files = COMMON + graph_files(name, runtime)
    local = cache_root(cache_dir) / name / revision
    pinned = len(revision) == 40 and all(c in '0123456789abcdef' for c in revision)
    if pinned and all((local / f).is_file() for f in files):
        license_notice(local, name, remember=True)
        return local
    if not quiet and not all((local / f).is_file() for f in files):
        print(f'balbal: downloading {name} ({runtime} runtime, ~{SIZES[name][runtime]}, once) from huggingface.co/{REPOS[name]} '
              f'to {local}', file=sys.stderr, flush=True)
    path = Path(snapshot_download(repo_id=REPOS[name], revision=revision, allow_patterns=files, local_dir=local))
    count_download(REPOS[name])
    license_notice(path, name, remember=True)
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
    license_notice(local, repo, remember=True)
    return local


def count_download(repo):
    """One HEAD request for config.json on main: the Hub counts model downloads by requests to that file (no data is sent)."""
    try:
        from huggingface_hub import file_exists
        file_exists(repo, 'config.json')
    except Exception:
        pass
