# Training on Windows

The training scripts run natively on Windows with one NVIDIA GPU, no WSL needed. Tested with `training/manas/train.py` and
`training/soup.py` on Windows with an RTX 4090, Python 3.12 and the pinned environment below.

## Environment

```powershell
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-deps -r requirements/full.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.10.0+cu128 True
```

`kenlm==0.3.0` and `texterrors==0.5.1` have no Windows wheels on PyPI and build from source, so install the Microsoft C++
Build Tools (the "Desktop development with C++" workload) before installing `requirements/full.txt`.

## Text encoding

Manifests, metrics and LM text are UTF-8. The scripts open them as UTF-8, but output piped to a file or over SSH uses the
console code page. Set both variables before a run so Turkish characters print correctly:

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
```

## Notes

- Multi-GPU (`torchrun`) uses NCCL, which PyTorch does not ship for Windows. Use one GPU, or train on Linux / WSL2.
- `training/build_lm.py` needs KenLM's `lmplz` and `build_binary`. KenLM has no official Windows binaries: build it yourself, or
  run that step in WSL2.
- In Windows PowerShell 5.1 an empty argument such as `--soup ''` is dropped before Python sees it. Write `--soup=` instead.
- `LOADER_WORKERS=4` worked (Windows starts DataLoader workers with spawn; each epoch starts them again).
