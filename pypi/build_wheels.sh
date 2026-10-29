#!/bin/bash
# Linux x86-64 wheels with the native parts compiled in, built in the manylinux_2_28 image (glibc 2.28+):
#   docker run --rm -v "$PWD":/io -w /io/pypi quay.io/pypa/manylinux_2_28_x86_64 ./build_wheels.sh     (from the repository root)
# -> wheelhouse/balbal-VERSION-cpXY-cpXY-manylinux_2_28_x86_64.whl for Python 3.11-3.12 (onnxruntime 1.30+ starts at 3.11, flashlight-text 0.0.7 stops at 3.12).
# Decoder extensions: x86-64-v2 (any 64-bit x86 CPU since ~2009). Operator library: v2 / v3 (AVX2) / v4 (AVX-512), picked at run time.
set -euo pipefail
NAT=src/balbal/_native K=../inference/kernels D=../inference/native   # the C++ sources of the repository's inference code
rm -rf build dist wheelhouse "$NAT"/*.so src/*.egg-info; mkdir -p wheelhouse
for LEVEL in v2 v3 v4; do
  F="-O3 -march=x86-64-$LEVEL -fPIC -std=c++17 -I$K/third_party/onnxruntime/include"
  tmp=$(mktemp -d)
  g++ $F -ffast-math -c $K/pkm_read.cc -o $tmp/pkm_read.o
  g++ $F -fno-fast-math -c $K/relpos.cc -o $tmp/relpos.o
  g++ -shared -s -Wl,--version-script=$K/exports.map $tmp/pkm_read.o $tmp/relpos.o -o $NAT/libbalbal_ops_$LEVEL.so
  rm -rf $tmp
done
for PY in cp311-cp311 cp312-cp312; do
  P=/opt/python/$PY/bin/python
  $P -m pip install -q --no-warn-script-location pybind11==2.10.4 flashlight-text==0.0.7 build==1.3.0 wheel==0.45.1 auditwheel
  rm -f "$NAT"/ctc_*.so
  EXT=$($P -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
  for M in ctc_lm_bonus ctc_fast_decoder ctc_stream_decoder; do
    g++ -O3 -march=x86-64-v2 -shared -s -std=c++17 -fPIC -fvisibility=hidden -DPYBIND11_BUILD_ABI='"_cxxabi1014"' \
      -I$($P -c 'import pybind11; print(pybind11.get_include())') -I$($P -c 'import sysconfig; print(sysconfig.get_path("include"))') \
      -I$D/include $D/$M.cpp -o $NAT/$M$EXT
  done
  rm -rf build dist src/*.egg-info
  $P -m build --wheel --outdir dist . > /dev/null
  $P -m wheel tags --remove --python-tag ${PY%%-*} --abi-tag ${PY##*-} --platform-tag linux_x86_64 dist/*.whl > /dev/null
  $P -m auditwheel repair --plat manylinux_2_28_x86_64 -w wheelhouse dist/*-linux_x86_64.whl
done
rm -rf build dist "$NAT"/*.so src/*.egg-info
ls -la wheelhouse
