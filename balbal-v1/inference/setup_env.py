"""Install the pinned PyTorch/NeMo inference environment into the active Python 3.12 venv, build the decoder, run checks.
  python setup_env.py --device cpu --model-dir MODEL_DIR
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--model-dir', type=Path, default=Path(os.getenv('BALBAL_MODEL_DIR', ROOT.parent / 'models')))
    args = p.parse_args()
    if sys.prefix == sys.base_prefix:
        raise SystemExit('Create and activate a Python 3.12 virtual environment first.')
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('Use Python 3.12 for the pinned inference environment.')
    pip = [sys.executable, '-m', 'pip']
    index = 'https://download.pytorch.org/whl/' + ('cu128' if args.device == 'cuda' else 'cpu')
    subprocess.run([*pip, 'install', 'torch==2.10.0', 'torchaudio==2.10.0', '--index-url', index], check=True)
    # Exact known-good runtime, without unrelated NeMo training extras.
    subprocess.run([*pip, 'install', '--no-deps', '-r', str(ROOT / 'requirements-runtime.txt')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'build_native.py')], check=True)
    model_dir = str(args.model_dir.resolve())
    if not (args.model_dir / 'conformer.pt').exists():
        print('Environment ready. Weights are not included; put them in MODEL_DIR (see README) and rerun the checks.')
        return
    subprocess.run([sys.executable, str(ROOT / 'self_test.py'), '--model-dir', model_dir], check=True)
    subprocess.run([sys.executable, '-c', 'import sys; from transcribe import Transcriber; '
                    f'Transcriber(sys.argv[1], {args.device!r}); print("Acoustic model, refiner and tokenizer verified")',
                    model_dir], cwd=ROOT, check=True)
    print(f'Ready: python transcribe.py --model-dir {model_dir} your_audio.wav --device {args.device}')


if __name__ == '__main__':
    main()
