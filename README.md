# balbal

Turkish speech recognition: model, CPU inference (offline and streaming) and training code.

| Model | Size | Encoder | Tokenizer | Languages |
|---|---|---|---|---|
| **balbal-v2** (default) | 45.4M parameters (30.0M of them a product-key memory table) | Conformer-CTC, 16 layers, d_model 176, 4x subsampling | SentencePiece 1024 | Turkish |
| **balbal-v2-manas** (large) | 313.3M parameters (192.0M of them a product-key memory table) | FastConformer-CTC, 18 layers, d_model 512, 4x subsampling | SentencePiece 2048 | Turkish, English |

Both models add a 3-pass loop refiner with a product-key acoustic memory on top of the CTC encoder and decode with a 6-gram token
KenLM beam search. Output: cased, punctuated text with per-word timestamps.

**Weights: coming soon on Hugging Face.** This repository holds the code only. The previous model's code is in [`balbal-v1/`](balbal-v1/).

## Contents

```
balbal/              shared Python package: model definitions, model builder, audio reading, text normalizer and scorer
configs/             balbal-v2/ and balbal-v2-manas/: NeMo model config, refiner + memory config, decoder settings
inference/
  transcribe.py      offline transcription (runtimes fast / onnx / torch)
  runtime.py         the three runtimes + model directory layout
  decoder.py         KenLM beam decoder (+ hotword biasing, word timestamps)
  export_onnx.py     PyTorch weights -> the two CPU ONNX graphs (+ onnx_tools.py)
  build_native.py    builds the native decoders in native/ (C++ / pybind11)
  kernels/           onnxruntime custom operators of the fast graph (C++ sources + build scripts, x86-64 and Android arm64)
  streaming/         streaming recognizer, WebSocket server, file client, browser microphone demo, exactness test
training/
  balbal_v2/train.py, manas/train.py   trainers (shared loop and data code in common/)
  soup.py            weight soups
  build_lm.py        6-gram token KenLM for the decoder
requirements/        cpu.txt (inference without torch), full.txt (export, torch runtime, training)
balbal-v1/           balbal-v1 code, kept for reproducibility
```

## Install

Python 3.12 on Linux x86-64.

```bash
# CPU inference only (runtimes fast + onnx, streaming)
pip install -r requirements/cpu.txt
# or everything (ONNX export, runtime torch, training): torch 2.10.0 + NeMo 2.5.0
pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128   # or a ROCm / CPU build
pip install --no-deps -r requirements/full.txt
```

Build the native decoders (needs a C++17 compiler; `flashlight-text==0.0.7` and `pybind11==2.10.4` from the requirements):

```bash
python inference/build_native.py              # builds inference/native/*.so, then checks them against flashlight-text
```

Build the custom operators of the fast graph (needs g++; the onnxruntime 1.30 headers are vendored in `inference/kernels/third_party/`):

```bash
inference/kernels/build.sh                    # -> inference/kernels/libbalbal_ops.so (MARCH=x86-64-v3 for a portable build)
python inference/kernels/test_kernels.py      # PkmRead / PkmReadQ8 vs numpy, RelPosScore vs the stock operators (bit-identical)
ANDROID_NDK=/path/to/ndk inference/kernels/build_android.sh   # Android arm64 -> inference/kernels/android-arm64/
```

`BALBAL_OPS_LIB` points the runtime at another build of the library.

## Model directory

Every tool takes `--model-dir MODEL_DIR` (or the environment variable `BALBAL_MODEL_DIR`):

```
MODEL_DIR/
  model.json                         {"model": "balbal-v2"} or {"model": "balbal-v2-manas"}
  model.pt                           PyTorch weights                    (runtime torch, ONNX export)
  tokenizer/                         tokenizer.model, tokenizer.vocab, vocab.txt
  lm/                                model.binary (KenLM), tokenizer.model
  onnx/model.onnx (+ .data)          fast graph, custom operators       (runtime fast, the default)
  onnx/model-plain.onnx (+ .data)    standard operators only            (runtime onnx)
```

The ONNX graphs are made from `model.pt` (any 16 kHz-readable clips serve as the parity check; a few seconds each):

```bash
python inference/export_onnx.py --model-dir MODEL_DIR clip1.wav clip2.wav clip3.wav
```

It writes both graphs and `onnx/export.json` (log-probability and frame-argmax agreement with PyTorch). balbal-v2-manas stores its
matrix weights in int8 by default (`--int8 on|off`); both fast graphs store the memory table in int8.

## Offline transcription

```bash
python inference/transcribe.py --model-dir MODEL_DIR speech.wav                  # fast CPU runtime
python inference/transcribe.py --model-dir MODEL_DIR --runtime onnx speech.wav   # standard onnxruntime operators only
python inference/transcribe.py --model-dir MODEL_DIR --runtime torch speech.wav  # PyTorch / NeMo (--device cuda for a GPU)
python inference/transcribe.py --model-dir MODEL_DIR --hotwords "Eray,Hasan" call.wav
```

One JSON line per file: `text`, `words` (`word`, `start`, `end` in seconds), `duration_seconds`, `inference_seconds`,
`realtime_factor`. `--text-only` prints the text alone, `--threads` sets the CPU threads (default `ASR_THREADS` or 8),
`--decoder flashlight` runs flashlight-text's own beam search (same best path as the default native decoder). Audio longer than
30 s is cut into 30 s chunks. WAV / FLAC / OGG are read directly, other formats through FFmpeg.

In Python:

```python
import sys; sys.path[:0] = ['inference', '.']
from runtime import Engine
from balbal.audio import read_audio
engine = Engine('MODEL_DIR')                       # runtime='fast' | 'onnx' | 'torch'
print(engine.transcribe(read_audio('speech.wav'))['text'])
```

## Streaming

The models are full-context, so streaming re-runs the graph on recent audio. Two modes:

- `segment` (default): every 0.32 s the graph runs once over the open segment and the whole segment is re-decoded. Partials equal
  the final text except for the last ~0.6 s. A segment ends after 0.5 s of quiet, blank audio, or at 5 s (cut at a short pause first).
- `window`: a sliding window (8 s left context, 0.32 s chunk, 0.64 s lookahead) commits each chunk once to an incremental beam
  search; a closed segment is re-decoded with full context for its final text.

The streaming decoder (`inference/native/ctc_stream_decoder.cpp`) returns the same best path as the offline decoder on the same frames.

```bash
python inference/streaming/server.py --model-dir MODEL_DIR [--mode segment|window] [--port 8791] [--workers 2 --threads 4]
python inference/streaming/client.py speech.wav --url ws://127.0.0.1:8791/ws      # streams a file in real time
# browser microphone demo: http://127.0.0.1:8791/
python inference/streaming/test_equiv.py --model-dir MODEL_DIR a.wav b.wav       # streaming == offline checks
```

WebSocket protocol (`ws://HOST:PORT/ws`):

| Direction | Message | Content |
|---|---|---|
| client → server | binary | little-endian int16 PCM, mono, 16 kHz, any piece size |
| client → server | text `{"type": "end"}` | no more audio |
| server → client | `ready` | `model`, `config`, `worker` |
| server → client | `partial` | `text` (open segment incl. the provisional tail), `stable`, `start`, `joins`, `audio_end` |
| server → client | `final` | `text`, `start`, `end`, `joins` (the segment continues the previous word) |
| server → client | `done` | `transcript`, `stats` |

In Python:

```python
import sys; sys.path[:0] = ['inference/streaming']
from streamer import Engine, SegmentStream
engine = Engine('MODEL_DIR')                       # one per process
stream = SegmentStream(engine)                     # one per live stream (Stream = window mode)
for piece in pieces: events = stream.accept(piece) # float32, 16 kHz
events += stream.finish(); print(stream.transcript())
```

## Training

Manifests are JSON lines, one row per clip:

```json
{"id": "a1", "audio_path": "clips/a1.wav", "duration_seconds": 4.2, "text": "Merhaba, nasılsınız?", "source": "my-set"}
```

Optional fields: `lang` (`tr`, `en` or `ns` for non-speech; default `--default-lang`, `ns` for an empty text), `lang_spans`
(`[[start_s, end_s, "en"], ...]` for code-switching), `pcm_sha256` (checked when present), `parts` (a row joined from segments of
other files). See `training/common/data.py`.

```bash
# balbal-v2: from balbal-v2 (or balbal-v1) weights
python training/balbal_v2/train.py runs/my-balbal --tokenizer-dir MODEL_DIR/tokenizer --init MODEL_DIR/model.pt \
    --train train.jsonl --dev dev.jsonl
# balbal-v2-manas: from balbal-v2-manas weights, here with the fine-tune settings
python training/manas/train.py runs/my-manas --tokenizer-dir MODEL_DIR/tokenizer --init MODEL_DIR/model.pt \
    --train train.jsonl --dev dev.jsonl --lr 5e-5 --epochs 2 --decay .5 --soup ""
# balbal-v2-manas from NVIDIA's English FastConformer-CTC-Large (encoder only; head, refiner and memory start fresh)
python training/manas/train.py runs/my-manas --tokenizer-dir MY_TOKENIZER --nemo stt_en_fastconformer_ctc_large.nemo --step-offset 0 \
    --train train.jsonl --dev dev.jsonl
# multi-GPU
torchrun --nproc_per_node 4 training/balbal_v2/train.py runs/my-balbal ...
```

The defaults are the release recipes:

| | balbal-v2 | balbal-v2-manas |
|---|---|---|
| Update | 64 clips, micro-batch 16 | 64 clips, micro-batch 8 |
| Length | 5 epochs | 4.5 epochs, then a 2-epoch fine-tune |
| LR | 1.4e-4, 500 warmup updates, flat, linear decay over the last 20 % to 10 % | 1.4e-4, 500 warmup updates, flat to epoch 2.5, linear decay to 10 %; fine-tune 5e-5, decay over the last 50 % |
| Optimizer | Muon + AdamW, weight decay .001 | Muon (bf16 Newton-Schulz) + AdamW, weight decay .01 |
| Memory table LR | 5x | 5x (and the learned read temperature) |
| Objective | 0.3 x base CTC + 0.7 x mean CTC of the 3 refiner passes + 0.1 x frame language CE | same |
| Released weights | final checkpoint of the 5 epochs | uniform soup of the fine-tune's 7 evaluations, epochs 0.5 to 2.0 |

Each run writes `config.json`, `metrics.jsonl` (a progress line every 100 updates, an evaluation every quarter epoch with greedy WER /
CER on `--dev`, overall and per `source`), the best checkpoints by dev WER in `checkpoints/`, `state.pt` (`--resume` continues the
run) and `final.pt`. `--cond` also trains an utterance-level language condition (tr / en / auto). All options: `--help`.

```bash
python training/soup.py soup.pt runs/my-manas/checkpoints/step*.pt          # uniform weight soup
python training/build_lm.py --tokenizer MODEL_DIR/tokenizer/tokenizer.model --train train.jsonl --heldout dev.jsonl \
    --out MODEL_DIR/lm --quantize                                              # the decoder's 6-gram token KenLM
```

## Checks

The commands used to test this release on CPU (`MODEL_DIR` with `model.pt`, `tokenizer/` and `lm/`; `clip.wav` a few seconds of Turkish speech):

```bash
python inference/build_native.py
inference/kernels/build.sh && python inference/kernels/test_kernels.py
python inference/export_onnx.py --model-dir MODEL_DIR clip.wav other1.wav other2.wav
for r in fast onnx torch; do python inference/transcribe.py --model-dir MODEL_DIR --runtime $r --text-only clip.wav; done   # same text
python inference/streaming/test_equiv.py --model-dir MODEL_DIR clip.wav other1.wav other2.wav
python inference/streaming/server.py --model-dir MODEL_DIR --port 8791 &   # then:
python inference/streaming/client.py clip.wav --url ws://127.0.0.1:8791/ws
LOADER_WORKERS=0 python training/balbal_v2/train.py runs/check --tokenizer-dir MODEL_DIR/tokenizer --init MODEL_DIR/model.pt \
    --train toy.jsonl --dev toy-dev.jsonl --batch 4 --micro 2 --updates 3 --eval-epochs 1 --progress-every 1
```

## License and citation

Code: Apache License 2.0 (`LICENSE`, `NOTICE`). The balbal-v2-manas model configuration derives from NVIDIA's
`stt_en_fastconformer_ctc_large` (CC-BY-4.0). Weights: see their model cards (coming soon on Hugging Face). How to cite: `CITATION.cff`.
