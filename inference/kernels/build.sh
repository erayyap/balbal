#!/bin/bash
# Build libbalbal_ops.so: the onnxruntime CPU custom operators of the fast graphs (domain "balbal": PkmRead, PkmReadQ8, RelPosScore).
# pkm_read.cc is built with -ffast-math, relpos.cc WITHOUT it (RelPosScore must divide exactly like onnxruntime's Div).
# MARCH (default native) selects the target CPU; the AVX-512 paths are used when the target has AVX-512, else the scalar paths
# (same results). Needs g++ (C++17). Usage: inference/kernels/build.sh   [MARCH=x86-64-v3 inference/kernels/build.sh]
set -euo pipefail
cd "$(dirname "$0")"
CXX=${CXX:-g++}; MARCH=${MARCH:-native}
F="-O3 -march=$MARCH -fPIC -std=c++17 -Ithird_party/onnxruntime/include"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
$CXX $F -ffast-math -c pkm_read.cc -o "$tmp/pkm_read.o"
$CXX $F -fno-fast-math -c relpos.cc -o "$tmp/relpos.o"
$CXX -shared -Wl,--version-script=exports.map "$tmp/pkm_read.o" "$tmp/relpos.o" -o "$tmp/libbalbal_ops.so"
mv "$tmp/libbalbal_ops.so" libbalbal_ops.so
echo "built $(pwd)/libbalbal_ops.so (-march=$MARCH)"
