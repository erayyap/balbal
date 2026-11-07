"""macOS arm64 and Windows x86-64 wheels with the native parts compiled in (Linux: build_wheels.sh in the manylinux image).
Run it with each Python the wheel is for (3.11, 3.12), from the repository root:
  python pypi/build_wheel.py      -> pypi/wheelhouse/balbal-VERSION-cpXY-cpXY-{macosx_14_0_arm64,win_amd64}.whl
Needs pybind11==2.10.4, flashlight-text==0.0.7, setuptools, build==1.3.0 and wheel==0.45.1 in that Python, and
  macOS:   the Xcode Command Line Tools (clang++). The wheel targets macOS 14+, as onnxruntime 1.30+ does.
  Windows: the Visual Studio Build Tools (MSVC, for the decoder extensions: the binding ABI must match flashlight-text's MSVC
           build) and LLVM's clang++ (for the operator library, which uses GCC builtins and the -march levels).
Decoder extensions: inference/build_native.py's build, checked against flashlight-text. Operator library: one arm64 build on macOS;
x86-64-v2 / v3 (AVX2) / v4 (AVX-512) on Windows, picked at run time as on Linux, with the static C runtime.
Windows also gets kenlm.dll: flashlight-text 0.0.7's Windows wheel links its KenLM module against a kenlm.dll it does not ship, so
the KenLM library is built here (pinned commit, MSVC, /MD like flashlight-text) and balbal._native loads it before that module."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
PYPI, KERNELS = ROOT / 'pypi', ROOT / 'inference' / 'kernels'
NATIVE = PYPI / 'src' / 'balbal' / '_native'
sys.path.insert(0, str(ROOT / 'inference'))
import build_native   # noqa: E402


def run(command, **kw):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run([str(c) for c in command], check=True, **kw)


def operator_libraries():
    """pkm_read.cc with -ffast-math, relpos.cc without it (RelPosScore must divide exactly like onnxruntime's Div)."""
    cxx = os.environ.get('CXX', 'clang++')
    if sys.platform == 'darwin':
        builds = [('libbalbal_ops.dylib', [], ['-fPIC'], ['-Wl,-exported_symbol,_RegisterCustomOps'])]
    else:
        builds = [(f'balbal_ops_{level}.dll', [f'-march=x86-64-{level}'], ['-fms-runtime-lib=static'], ['-fms-runtime-lib=static'])
                  for level in ('v2', 'v3', 'v4')]
    for name, march, compile_flags, link_flags in builds:
        with tempfile.TemporaryDirectory() as tmp:
            flags = ['-O3', *march, *compile_flags, '-std=c++17', f'-I{KERNELS / "third_party" / "onnxruntime" / "include"}']
            run([cxx, *flags, '-ffast-math', '-c', KERNELS / 'pkm_read.cc', '-o', Path(tmp) / 'pkm_read.o'])
            run([cxx, *flags, '-fno-fast-math', '-c', KERNELS / 'relpos.cc', '-o', Path(tmp) / 'relpos.o'])
            run([cxx, '-shared', *link_flags, Path(tmp) / 'pkm_read.o', Path(tmp) / 'relpos.o', '-o', NATIVE / name])


def main():
    if sys.platform == 'darwin':
        os.environ.setdefault('MACOSX_DEPLOYMENT_TARGET', '14.0')
        platform_tag = 'macosx_' + os.environ['MACOSX_DEPLOYMENT_TARGET'].replace('.', '_') + '_arm64'
    elif sys.platform == 'win32':
        platform_tag = 'win_amd64'
    else:
        raise SystemExit('Linux wheels: pypi/build_wheels.sh in the manylinux image.')
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    for f in NATIVE.iterdir():
        if f.suffix in ('.so', '.pyd', '.dll', '.dylib'): f.unlink()
    operator_libraries()
    for f in NATIVE.iterdir():
        if f.suffix in ('.lib', '.exp'): f.unlink()   # the linker's import library of each DLL
    if sys.platform == 'win32': build_native.build_kenlm(NATIVE)
    build_native.build(NATIVE, ['-O3', '-fvisibility=hidden'])
    build_native.check(NATIVE)
    dist = PYPI / 'dist'
    shutil.rmtree(dist, ignore_errors=True); shutil.rmtree(PYPI / 'build', ignore_errors=True)
    run([sys.executable, '-m', 'build', '--wheel', '--outdir', dist, PYPI])
    run([sys.executable, '-m', 'wheel', 'tags', '--remove', '--python-tag', tag, '--abi-tag', tag, '--platform-tag', platform_tag,
         *dist.glob('*.whl')])
    (PYPI / 'wheelhouse').mkdir(exist_ok=True)
    for wheel in dist.glob(f'*-{tag}-{tag}-{platform_tag}.whl'):
        shutil.copy(wheel, PYPI / 'wheelhouse' / wheel.name); print('built', PYPI / 'wheelhouse' / wheel.name)
    shutil.rmtree(dist, ignore_errors=True); shutil.rmtree(PYPI / 'build', ignore_errors=True)


if __name__ == '__main__':
    main()
