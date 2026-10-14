"""Build the tiny token bonus wrapper for the pinned Flashlight wheel."""
from pathlib import Path
import importlib.metadata
import os
import shlex
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent


def main():
    import pybind11
    if sys.platform != 'linux':
        raise SystemExit('This build is tested on Linux x86-64; use Linux/WSL2.')
    if importlib.metadata.version('flashlight-text') != '0.0.7' or pybind11.__version__ != '2.10.4':
        raise SystemExit('Install the exact versions in requirements.txt first (binding ABI matters).')
    output = ROOT / 'native' / ('ctc_lm_bonus' + sysconfig.get_config_var('EXT_SUFFIX'))
    temp = output.with_name(output.name + f'.tmp-{os.getpid()}')
    command = [*shlex.split(os.environ.get('CXX', 'c++')), '-O3', '-shared', '-std=c++14', '-fPIC',
               '-DPYBIND11_BUILD_ABI="_cxxabi1014"',
               '-I' + pybind11.get_include(), '-I' + sysconfig.get_path('include'),
               '-I' + str(ROOT / 'native/include'), str(ROOT / 'native/ctc_lm_bonus.cpp'), '-o', str(temp)]
    try:
        subprocess.run(command, check=True)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    # Check in a new interpreter: a previously imported extension cannot be replaced in memory.
    code = '''import sys
sys.path.insert(0, sys.argv[1])
from flashlight.lib.text.decoder import ZeroLM
from ctc_lm_bonus import TokenBonusLM
lm = TokenBonusLM(ZeroLM(), 3.0)
state = lm.start(False)
state, score = lm.score(state, 0)
assert abs(score - 3.0) < 1e-6
assert abs(lm.finish(state)[1]) < 1e-6
'''
    subprocess.run([sys.executable, '-c', code, str(ROOT / 'native')], check=True)
    print('Native decoder binding built and verified:', output.name)


if __name__ == '__main__':
    main()
