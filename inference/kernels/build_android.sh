#!/bin/bash
# libbalbal_ops.so for Android arm64 (API 26+) with the Android NDK; load it into onnxruntime-android with
# SessionOptions.registerCustomOpLibrary. The AVX-512 paths are compiled out; ARM uses the scalar top-k (same results).
#   ANDROID_NDK=/path/to/ndk inference/kernels/build_android.sh   ->  inference/kernels/android-arm64/libbalbal_ops.so
set -euo pipefail
cd "$(dirname "$0")"
: "${ANDROID_NDK:?set ANDROID_NDK to the NDK directory}"
CXX=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang++
F="-O3 -march=armv8.2-a+dotprod -fPIC -std=c++17 -Ithird_party/onnxruntime/include"
mkdir -p android-arm64
$CXX $F -ffast-math -fno-finite-math-only -c pkm_read.cc -o android-arm64/pkm_read.o
$CXX $F -fno-fast-math -c relpos.cc -o android-arm64/relpos.o
$CXX -shared -static-libstdc++ -Wl,-z,max-page-size=16384 -Wl,--gc-sections -Wl,--version-script=exports.map \
  -Wl,-soname,libbalbal_ops.so -s android-arm64/pkm_read.o android-arm64/relpos.o -o android-arm64/libbalbal_ops.so
rm android-arm64/*.o
ls -la android-arm64/
