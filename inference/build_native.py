"""Build the native CTC decoder extensions against the pinned flashlight-text wheel, then check them.

  ctc_lm_bonus        TokenBonusLM (per-token length reward), HotwordLM (hotword biasing) and MixLM (general + domain LM mix)
                      around flashlight's KenLM
  ctc_fast_decoder    FastCTCDecoder: the exact best path of flashlight's LexiconFreeDecoder (CTC) with fewer LM queries
  ctc_stream_decoder  StreamCTCDecoder: the same search, incremental (streaming)

Needs a C++17 compiler, flashlight-text==0.0.7 and pybind11==2.10.4 (the binding ABI must match the wheel's). On Windows the
compiler is MSVC (Visual Studio Build Tools, "Desktop development with C++"), found by setuptools, as flashlight-text's Windows
wheel is an MSVC build. Windows also gets native/kenlm.dll, which flashlight-text's Windows KenLM module needs but its wheel lacks
(built from KenLM; inference/decoder.py loads it first).
CXXFLAGS (default "-O3 -march=native"; ignored on Windows) sets the optimisation / target CPU.
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
from ctc_lm_bonus import TokenBonusLM, HotwordLM, MixLM
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
mix = MixLM(ZeroLM(), ZeroLM(), 0.3)
s, a = mix.score(mix.start(False), 7)
assert abs(a) < 1e-6 and abs(mix.finish(s)[1]) < 1e-6   # (1 - w) * 1 + w * 1 = 1
assert abs(mix.mix(-1.0, -3.0) - np.log10(.7e-1 + .3e-3)) < 1e-6 and abs(mix.mix(-3.0, -1.0) - np.log10(.7e-3 + .3e-1)) < 1e-6
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


KENLM_COMMIT = '4cb443e60b7bf2c0ddf3c745378f76cb59e254e5'
KENLM_SHA256 = '11df4f929b175f0b3bd26be7a5a83b3c17cdefa83baecab0b6e69f77430184ea'   # github.com/kpu/kenlm/archive/COMMIT.tar.gz
KENLM_SOURCES = (['util/' + n for n in ('bit_packing', 'ersatz_progress', 'exception', 'file', 'file_piece', 'float_to_string',
                                        'integer_to_string', 'mmap', 'murmur_hash', 'parallel_read', 'pool', 'read_compressed',
                                        'scoped', 'spaces', 'string_piece', 'usage')]
                 + ['util/double-conversion/' + n for n in ('bignum-dtoa', 'bignum', 'cached-powers', 'double-to-string', 'fast-dtoa',
                                                            'fixed-dtoa', 'string-to-double', 'strtod')]
                 + ['lm/' + n for n in ('bhiksha', 'binary_format', 'config', 'lm_exception', 'model', 'quantize', 'read_arpa',
                                        'search_hashed', 'search_trie', 'sizes', 'trie', 'trie_sort', 'value_build',
                                        'virtual_interface', 'vocab')])
# What flashlight-text-kenlm.dll imports from kenlm.dll: lm::ngram::Config::Config() and lm::ngram::LoadVirtual(path, config, type)
KENLM_EXPORTS = ['??0Config@ngram@lm@@QEAA@XZ', '?LoadVirtual@ngram@lm@@YAPEAVModel@base@2@PEBDAEBUConfig@12@W4ModelType@12@@Z']

CHECK_KENLM = """import ctypes, os, sys
ctypes.WinDLL(os.path.join(sys.argv[1], 'kenlm.dll'))   # loaded first, as balbal does: flashlight's module then finds it
from flashlight.lib.text.dictionary import Dictionary
from flashlight.lib.text.decoder.kenlm import KenLM
d = Dictionary()
for w in ['<unk>', 'a', 'the', 'looking', 'on']: d.add_entry(w)
lm = KenLM(sys.argv[2], d)
s, a = lm.score(lm.start(False), d.get_index('looking')); s, b = lm.score(s, d.get_index('on'))
got, want = (a, b, lm.finish(s)[1]), (-0.48465219140052795, -0.3488368093967438, -2.4608566761016846)   # Linux flashlight-text
assert all(abs(x - y) < 1e-6 for x, y in zip(got, want)), got
print('kenlm.dll: flashlight-text KenLM loads and scores', got)
"""


def build_kenlm(out_dir):
    """Windows: flashlight-text 0.0.7's Windows wheel links its KenLM module against a kenlm.dll it does not ship. Builds out_dir/kenlm.dll
    from KenLM (pinned commit, MSVC with /MD like flashlight-text), then checks flashlight's KenLM through it on KenLM's test ARPA."""
    import hashlib
    import io
    import tarfile
    import tempfile
    import urllib.request
    from setuptools._distutils.ccompiler import new_compiler
    data = urllib.request.urlopen(f'https://github.com/kpu/kenlm/archive/{KENLM_COMMIT}.tar.gz', timeout=120).read()
    if hashlib.sha256(data).hexdigest() != KENLM_SHA256: raise SystemExit('KenLM archive: unexpected SHA-256')
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive: archive.extractall(tmp, filter='data')
        source = Path(tmp) / f'kenlm-{KENLM_COMMIT}'
        compiler = new_compiler(); compiler.initialize()
        objects = compiler.compile([str(source / (f + '.cc')) for f in KENLM_SOURCES], output_dir=str(Path(tmp) / 'objects'),
                                   macros=[('NDEBUG', None), ('KENLM_MAX_ORDER', '6'), ('_CRT_SECURE_NO_WARNINGS', None)],
                                   include_dirs=[str(source)], extra_postargs=['/std:c++14', '/utf-8'])   # KenLM uses std::binary_function
        compiler.link_shared_object(objects, str(Path(out_dir) / 'kenlm.dll'), export_symbols=KENLM_EXPORTS)   # /DLL (link_shared_lib is not)
        subprocess.run([sys.executable, '-c', CHECK_KENLM, str(out_dir), str(source / 'lm' / 'test.arpa')], check=True)


def build(out_dir, flags):
    """Compile MODULES into out_dir: $CXX (default c++) with flags on Linux / macOS, MSVC through setuptools on Windows."""
    import pybind11
    out_dir = Path(out_dir)
    includes = [pybind11.get_include(), str(ROOT / 'native/include')]
    if sys.platform == 'win32':
        return build_msvc(out_dir, includes)
    # The binding ABI tag must match the flashlight-text wheel: set explicitly for its Linux (gcc) build, pybind11's
    # default for its macOS (clang/libc++) build.
    abi = [] if sys.platform == 'darwin' else ['-DPYBIND11_BUILD_ABI="_cxxabi1014"']
    for name in MODULES:
        output = out_dir / (name + sysconfig.get_config_var('EXT_SUFFIX'))
        temp = output.with_name(output.name + f'.tmp-{os.getpid()}')
        command = [*shlex.split(os.environ.get('CXX', 'c++')), *flags, '-shared', '-std=c++17', '-fPIC',
                   *abi, *('-I' + i for i in includes), '-I' + sysconfig.get_path('include'),
                   str(ROOT / 'native' / (name + '.cpp')), '-o', str(temp)]
        if sys.platform == 'darwin':
            command += ['-undefined', 'dynamic_lookup']   # Python symbols resolve at import, as in any macOS extension
        try:
            subprocess.run(command, check=True)
            temp.replace(output)
        finally:
            temp.unlink(missing_ok=True)


def build_msvc(out_dir, includes):
    """Windows: setuptools' build_ext finds MSVC and links against the Python import library (/MD, like flashlight-text)."""
    import shutil
    import tempfile
    from setuptools import Distribution, Extension
    from setuptools.command.build_ext import build_ext
    with tempfile.TemporaryDirectory() as tmp:
        modules = [Extension(name, [str(ROOT / 'native' / (name + '.cpp'))], include_dirs=includes, language='c++',
                             extra_compile_args=['/O2', '/std:c++17', '/EHsc', '/utf-8']) for name in MODULES]
        command = build_ext(Distribution({'name': 'balbal-native', 'ext_modules': modules}))
        command.build_lib, command.build_temp = tmp, str(Path(tmp) / 'objects')
        command.ensure_finalized(); command.run()
        for name in MODULES:
            built = Path(tmp) / (name + sysconfig.get_config_var('EXT_SUFFIX'))
            shutil.copyfile(built, out_dir / built.name)


def check(folder):
    """Check the extensions in folder in a new interpreter: a previously imported extension cannot be replaced in memory."""
    subprocess.run([sys.executable, '-c', CHECK, str(folder)], check=True)


def main():
    import pybind11
    if sys.platform not in ('linux', 'darwin', 'win32'):
        raise SystemExit('This build is tested on Linux x86-64, macOS arm64 and Windows x86-64.')
    if importlib.metadata.version('flashlight-text') != '0.0.7' or pybind11.__version__ != '2.10.4':
        raise SystemExit('Install flashlight-text==0.0.7 and pybind11==2.10.4 first (the binding ABI must match the wheel).')
    if sys.platform == 'win32': build_kenlm(ROOT / 'native')
    build(ROOT / 'native', shlex.split(os.environ.get('CXXFLAGS', '-O3 -march=native')))
    check(ROOT / 'native')
    print('Native decoder extensions built and verified:', ', '.join(MODULES))


if __name__ == '__main__':
    main()
