"""Prebuilt native parts: the beam-search decoder extensions (ctc_lm_bonus, ctc_fast_decoder, ctc_stream_decoder; pybind11 over
flashlight-text's KenLM) and the onnxruntime custom operator library (libbalbal_ops_{v2,v3,v4}.so on Linux, balbal_ops_{v2,v3,v4}.dll
on Windows, libbalbal_ops.dylib on macOS).

Windows: flashlight-text's KenLM module needs a kenlm.dll that its wheel does not include; the Windows wheel of this package carries
one (built from KenLM). Importing this module loads it by full path, so flashlight's module finds it already loaded."""
import os
from pathlib import Path
import sys

if sys.platform == 'win32' and (Path(__file__).resolve().parent / 'kenlm.dll').exists():
    import ctypes
    _dll_directory = os.add_dll_directory(str(Path(__file__).resolve().parent))
    _kenlm = ctypes.WinDLL(str(Path(__file__).resolve().parent / 'kenlm.dll'))
