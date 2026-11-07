#!/bin/bash
# Build libbalbal_ops.so: the onnxruntime CPU custom operators of the fast graphs (domain "balbal": PkmRead, PkmReadQ8, RelPosScore).
# pkm_read.cc is built with -ffast-math, relpos.cc WITHOUT it (RelPosScore must divide exactly like onnxruntime's Div).
# MARCH (default native) selects the target CPU; the AVX-512 paths are used when the target has AVX-512, else the scalar paths
# (same results). Needs g++ (C++17); on macOS clang++ (Xcode Command Line Tools), which builds libbalbal_ops.dylib; on Windows
# (Git Bash) LLVM's clang++ with the Visual Studio Build Tools, which builds balbal_ops.dll (static C runtime, no extra DLLs).
# Usage: inference/kernels/build.sh   [MARCH=x86-64-v3 inference/kernels/build.sh]
set -euo pipefail
cd "$(dirname "$0")"
PIC=-fPIC
case "$(uname)" in
  Darwin) CXX=${CXX:-clang++}; LIB=libbalbal_ops.dylib; EXPORTS="-Wl,-exported_symbol,_RegisterCustomOps" ;;
  MINGW*|MSYS*|CYGWIN*) CXX=${CXX:-clang++}; LIB=balbal_ops.dll; PIC="-fms-runtime-lib=static"; EXPORTS="$PIC" ;;   # dllexport in pkm_read.cc
  *) CXX=${CXX:-g++}; LIB=libbalbal_ops.so; EXPORTS="-Wl,--version-script=exports.map" ;;
esac
MARCH=${MARCH:-native}
F="-O3 -march=$MARCH $PIC -std=c++17 -Ithird_party/onnxruntime/include"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
$CXX $F -ffast-math -c pkm_read.cc -o "$tmp/pkm_read.o"
$CXX $F -fno-fast-math -c relpos.cc -o "$tmp/relpos.o"
$CXX -shared $EXPORTS "$tmp/pkm_read.o" "$tmp/relpos.o" -o "$tmp/$LIB"
mv "$tmp/$LIB" "$LIB"
echo "built $(pwd)/$LIB (-march=$MARCH)"
