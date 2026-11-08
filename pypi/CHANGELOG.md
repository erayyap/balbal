# Changelog

## 0.6.2

- `balbal.__version__` (and `balbal --version`, the server's `/health`) said 0.6.0 in the 0.6.1 package; it now matches the
  package version. No other change.

## 0.6.1

- The Moroccan Darija model is renamed **balbal-v2e-zellige** (the "e" stands for experimental); it was balbal-v2.1-zellige in 0.6.0.
  Hugging Face repo: [erayyapagci/balbal-v2e-zellige](https://huggingface.co/erayyapagci/balbal-v2e-zellige) (the old repo id
  redirects). `--model zellige`, `v2e` and `v2e-zellige` load it; the old names (`balbal-v2.1-zellige`, `v2.1`, `v2.1-zellige`, the old
  repo id) still work. Same weights and output. The model is downloaded once more into a new cache folder; the old folder
  `<HF_HOME>/balbal/balbal-v2.1-zellige` can be deleted. balbal-v2 and balbal-v2-manas are unchanged.

## 0.6.0

- New model: **balbal-v2.1-zellige** (`--model zellige`, `balbal.load('zellige')`), Moroccan Darija speech recognition (language `ary`, Arabic
  script, output as decoded: no casing, no number writing). Sabeq-ASR FastConformer encoder (80 ms output frames) with the balbal
  refiner and product-key memory, 6-gram piece KenLM. ONNX only: one plain-operator graph (`onnx/model.onnx`, ~1.2 GB fp32); `--runtime
  fast` loads the same graph without a warning. Offline transcription only: `balbal live` and the WebSocket stream
  raise a clear error for models with 80 ms frames (word timestamps are scaled to the frame length but not checked against forced alignment for this model).
- Licence: zellige is CC BY-NC-SA 4.0 (non-commercial), unlike balbal-v2 / balbal-v2-manas (balbal Community License 1.1). A model whose
  `model.json` declares a non-commercial `"license"` prints one notice naming it when first downloaded (a local folder: once per process);
  `balbal --help` lists each model with its language and licence.
- Model folders may set `"frame_samples"` (samples per output frame, default 640 = 40 ms) in `model.json`: long-audio windows, word times
  and the segment cuts follow it. balbal-v2 and balbal-v2-manas output is unchanged.

## 0.5.0

- Wheels for macOS (Apple Silicon, macOS 14+) and Windows x86-64, Python 3.11 and 3.12, with the native beam-search decoder and the
  custom onnxruntime operators compiled in (Windows: x86-64-v2, AVX2 and AVX-512 builds, chosen at run time as on Linux). macOS build
  support by Ali Kösekahya (#3). The Windows wheel also carries `kenlm.dll` (KenLM, LGPL), which flashlight-text's Windows KenLM
  module needs but its own wheel does not include.
- Windows fixes: model.json, decoder.json and the demo page are read as UTF-8; `balbal serve` decodes MP3 / M4A / video uploads
  through a closed temporary file; `balbal run` output redirected to a file or another program is written as UTF-8.
- Linux: the native code is unchanged from 0.4.0, and so is the output.

## 0.4.0

- Domain LM: `--domain-lm DIR` / `--domain-weight W` (`balbal run`, `live`, `serve`), `settings={'domain_lm': ..., 'domain_weight': ...}`
  or the same keys in a model folder's `decoder.json` mix a KenLM built on your own text (`training/build_lm.py --text`) into the
  general one: `p = (1 - w) p_general + w p_domain`, default weight 0.3. Without a domain LM the output is unchanged.
- `training/build_lm.py` (GitHub repository) builds an LM from `--text` alone.

## 0.3.1

- The README points to the models' new license, balbal Community License 1.1: small businesses may also build and sell speech
  products on balbal-v2 and balbal-v2-manas, with a visible "Powered by balbal" credit. No code changes; model files and output are
  unchanged from 0.3.0.

## 0.3.0

- Models trained on the balbal architectures, in any language, load by Hugging Face repo id or folder:
  `balbal.load('owner/name')`, `balbal run --model owner/name`, `balbal serve --model ./my-model`. They need `model.json`, `lm/`
  and `onnx/` (exported with `inference/export_onnx.py` from the GitHub repo); a `decoder.json` in the model replaces the default
  decoder settings.
- `model.json` may name the model's `languages` (primary first), its `itn` and `long_windows`. Number writing (ITN) and hotword casing
  follow the primary language: Turkish keeps its ITN, other languages keep numbers as the model writes them.
- The servers take model names, folders and repo ids; Wyoming routes a request's language to a served model with that language.
- Subtitles end segments at `؟ 。 ？ ！ ।` too; word timing treats `« » „ ¿ ¡ ، ؟ ؛ 。 、` as punctuation.
- The output of balbal-v2 and balbal-v2-manas is unchanged from 0.2.0 (text, timings, confidence).

## 0.2.0

- Live transcription: `model.stream()` / `model.transcribe_stream()` give partial and final text while audio arrives, using the
  native incremental decoder. `balbal live` transcribes the microphone (`pip install 'balbal[live]'`) or plays a file as if live.
- `balbal serve` (`pip install 'balbal[server]'`): an OpenAI-compatible `POST /v1/audio/transcriptions` (json, text,
  verbose_json, srt, vtt), a live WebSocket at `/v1/stream`, and a browser microphone demo at `/`.
- `balbal wyoming` (`pip install 'balbal[wyoming]'`): speech-to-text for Home Assistant over the Wyoming protocol.
- Every word has a `confidence` (0-1). The result has `no_speech` when no word is confident: silence, noise, or a short word
  written over digital silence.
- `balbal run` takes folders, writes txt / json / srt / vtt files with `--format` and `--output-dir`, runs several processes
  with `--jobs` and resumes with `--skip-existing`.
- The transcribed text and word timings are the same as in 0.1.2.

## 0.1.2

- Long audio: balbal-v2 windows are now 30 s with 10 s overlap (as manas already had).

## 0.1.1

- Long audio: overlapping windows stitched on the CTC frame grid, decoded in one KenLM pass.

## 0.1.0

- First release: balbal-v2 and balbal-v2-manas on CPU, KenLM beam search, hotwords, digits for spoken numbers, word timestamps.
