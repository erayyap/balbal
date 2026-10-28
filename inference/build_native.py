"""Build the native CTC decoder extensions against the pinned flashlight-text wheel, then check them.

  ctc_lm_bonus        TokenBonusLM (per-token length reward) and HotwordLM (hotword biasing) around flashlight's KenLM
  ctc_fast_decoder    FastCTCDecoder: the exact best path of flashlight's LexiconFreeDecoder (CTC) with fewer LM queries
  ctc_stream_decoder  StreamCTCDecoder: the same search, incremental (streaming)

Needs a C++17 compiler, flashlight-text==0.0.7 and pybind11==2.10.4 (the binding ABI must match the wheel's).
CXXFLAGS (default "-O3 -march=native") sets the optimisation / target CPU.
  python inference/build_native.py
"""
from pathlib import Path
import importlib.metadata
import os
import shlex
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent
MODULES = ('ctc_lm_bonus', 'ctc_fast_decoder', 'ctc_stream_decoder')

CHECK = '''import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from flashlight.lib.text.decoder import ZeroLM, LexiconFreeDecoder, LexiconFreeDecoderOptions, CriterionType
from ctc_lm_bonus import TokenBonusLM, HotwordLM
from ctc_fast_decoder import FastCTCDecoder
from ctc_stream_decoder import StreamCTCDecoder
lm = TokenBonusLM(ZeroLM(), 3.0)
state, score = lm.score(lm.start(False), 0)
assert abs(score - 3.0) < 1e-6 and abs(lm.finish(state)[1]) < 1e-6
hot = HotwordLM(ZeroLM(), [[5, 6]], [[1.0, 2.0]])
s = hot.start(False); s, a = hot.score(s, 5); s, b = hot.score(s, 6)
assert (a, b, hot.finish(s)[1]) == (1.0, 2.0, 0.0)   # completed hotword keeps its bonus
s = hot.start(False); s, a = hot.score(s, 5)
assert hot.finish(s)[1] == -1.0                      # unfinished hotword: bonus taken back
# fast and streaming decoders == flashlight on random emissions (same best path)
rng = np.random.default_rng(0); V = 40
for trial in range(20):
    x = rng.standard_normal((int(rng.integers(5, 80)), V + 1)).astype(np.float32) * 3
    x -= np.log(np.exp(x).sum(1, keepdims=True)); x = np.ascontiguousarray(x)
    bonus = TokenBonusLM(ZeroLM(), 0.5)
    ref = LexiconFreeDecoder(LexiconFreeDecoderOptions(16, 8, 6.0, 1.0, 0.0, True, CriterionType.CTC), bonus, -1, V, []).decode(x.ctypes.data, *x.shape)[0].tokens
    fast = FastCTCDecoder(16, 8, 6.0, 1.0, bonus, V, 0.5).decode(x.ctypes.data, *x.shape)
    assert list(ref) == list(fast), trial
    sd = StreamCTCDecoder(16, 8, 6.0, 1.0, bonus, V, 0.5); h = len(x) // 2
    a, b = np.ascontiguousarray(x[:h]), np.ascontiguousarray(x[h:])
    sd.step(a.ctypes.data, *a.shape); sd.step(b.ctypes.data, *b.shape)
    collapse = lambda p: [t for i, t in enumerate(p) if 0 <= t < V and (i == 0 or t != p[i - 1])]
    assert sd.finish() == collapse(list(fast)[1:]), trial
'''


def main():
    import pybind11
    if sys.platform != 'linux':
        raise SystemExit('This build is tested on Linux x86-64; use Linux/WSL2.')
    if importlib.metadata.version('flashlight-text') != '0.0.7' or pybind11.__version__ != '2.10.4':
        raise SystemExit('Install flashlight-text==0.0.7 and pybind11==2.10.4 first (the binding ABI must match the wheel).')
    flags = shlex.split(os.environ.get('CXXFLAGS', '-O3 -march=native'))
    for name in MODULES:
        output = ROOT / 'native' / (name + sysconfig.get_config_var('EXT_SUFFIX'))
        temp = output.with_name(output.name + f'.tmp-{os.getpid()}')
        command = [*shlex.split(os.environ.get('CXX', 'c++')), *flags, '-shared', '-std=c++17', '-fPIC',
                   '-DPYBIND11_BUILD_ABI="_cxxabi1014"', '-I' + pybind11.get_include(), '-I' + sysconfig.get_path('include'),
                   '-I' + str(ROOT / 'native/include'), str(ROOT / 'native' / (name + '.cpp')), '-o', str(temp)]
        try:
            subprocess.run(command, check=True)
            temp.replace(output)
        finally:
            temp.unlink(missing_ok=True)
    # Check in a new interpreter: a previously imported extension cannot be replaced in memory.
    subprocess.run([sys.executable, '-c', CHECK, str(ROOT / 'native')], check=True)
    print('Native decoder extensions built and verified:', ', '.join(MODULES))


if __name__ == '__main__':
    main()
