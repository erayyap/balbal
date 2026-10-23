# balbal-v1: Turkish speech recognition (code)

This repository holds the inference, text normalization and training code of **balbal-v1**, a small Turkish speech
recognition model.

> **Weights are on Hugging Face:** [erayyapagci/balbal-v1](https://huggingface.co/erayyapagci/balbal-v1) (model, tokenizer,
> KenLM), under the [balbal Community License 1.0](https://huggingface.co/erayyapagci/balbal-v1/blob/main/LICENSE): free for
> research and non-commercial use, and for small businesses' own commercial use; selling speech recognition products built on
> it needs a commercial license. The code loads the weights from a folder (`MODEL_DIR`).

## Architecture

| Part | Details |
|---|---|
| Acoustic model | Conformer-CTC (NeMo `EncDecCTCModelBPE`), 16 layers, d_model 176, 80 mel, 4x subsampling (40 ms frames) |
| Loop refiner | 2-layer Transformer refiner on top of the encoder output, run for **3 passes** (each pass takes the previous CTC distribution as input) |
| Acoustic memory | Product-key memory: 500×500 sub-keys, 4 heads, top-16, value size 120, 3-frame context |
| Vocabulary | SentencePiece 1024 pieces + CTC blank (1025 classes) |
| Language model | 6-gram **token** KenLM (over SentencePiece pieces) |
| Decoder | flashlight-text lexicon-free beam search: alpha 0.75, beta 1.5, temperature 1.2, beam 64, top-12 tokens, threshold 8 |
| Parameters | **45.4M stored / 15.4M active** (most of the memory table is not read on any given frame) |

Long audio is cut into fixed 30 s chunks (no VAD); each chunk is decoded on its own.

## Results

| Evaluation | WER |
|---|---|
| FLEURS tr (test), balbal-normalizer, KenLM | **6.54** |
| Mu-Bench tr-TR, official scoring | **11.93** |

balbal-normalizer: `normalizer/metrics_v2.py` (see below).

## Repository layout

```
steering/       steering: push the memory reads toward a given sentence (auto / weak / strong)
balbal_model/   model definition: loop CTC refiner + acoustic memory (attached to the NeMo Conformer)
configs/        model_config.yaml (NeMo), refiner.json (refiner/memory), decoder.json (decoder settings)
inference/      PyTorch/NeMo inference, ONNX CPU runtime, live streaming, KenLM decoder, ONNX export
normalizer/     Turkish normalization, balbal-normalizer scoring, digits→words (training labels), ITN (words→digits)
training/       training loop, Muon+AdamW, data/augmentation, weight soups, KenLM build, calibration
```

## MODEL_DIR layout

```
MODEL_DIR/
  conformer.pt                     PyTorch state dict (balbal-v1)
  tokenizer/tokenizer.model        SentencePiece
  tokenizer/tokenizer.vocab
  tokenizer/vocab.txt
  lm/model.binary                  6-gram token KenLM (trie)
  lm/tokenizer.model               the same SentencePiece
  model-ext.onnx, model-ext.data   (ONNX runtime only; written by export_onnx.py)
```

Pass it with `--model-dir MODEL_DIR` or the `BALBAL_MODEL_DIR` environment variable. To download:

```bash
huggingface-cli download erayyapagci/balbal-v1 --local-dir MODEL_DIR
```

## Inference

Linux x86-64, Python 3.12.

**PyTorch/NeMo (CPU or GPU):**

```bash
python -m venv .venv && . .venv/bin/activate
python inference/setup_env.py --device cpu --model-dir MODEL_DIR   # torch 2.10 + pinned NeMo, builds the decoder
python inference/transcribe.py --model-dir MODEL_DIR speech.wav --device cpu
```

The output is one JSON line per file (`text`, chunks, timings); `--text-only` prints only the text.
WAV/FLAC/OGG are read directly, other formats through FFmpeg; resampling to 16 kHz uses the same resampler as training
(scipy polyphase).

**ONNX (CPU only, no torch/NeMo needed):**

```bash
python inference/export_onnx.py --model-dir MODEL_DIR sample1.wav sample2.wav   # once, in the PyTorch environment
pip install -r inference/requirements-onnx.txt && python inference/build_native.py
python inference/runtime_onnx.py --model-dir MODEL_DIR speech.wav                  # threads via ASR_THREADS
```

`export_onnx.py` rewrites the mel front end as a DFT convolution and the refiner attention and memory reads as
shape-independent ops; it measures the difference to the NeMo model on the clips you give it (`export.json`) and moves the
weights to a separate 64 KiB-aligned file (onnxruntime memory-maps it).

Decoder self-test (synthetic input): `python inference/self_test.py --model-dir MODEL_DIR`.

From Python:

```python
import sys; sys.path.insert(0, 'inference')
from transcribe import Transcriber
asr = Transcriber('MODEL_DIR', device='cpu')
print(asr.transcribe('speech.wav')['text'])
```

**Hotwords:** give the names or words you expect in the recording and the decoder favours them. The model and the language
model do not change.

```bash
python inference/runtime_onnx.py --model-dir MODEL_DIR --hotwords "Arslan,Zaynep" speech.wav   # same flag in transcribe.py
```

From Python: `asr.transcribe('speech.wav', hotwords=['Arslan', 'Zaynep'])`. Every token that extends a hotword earns
`hotword_bonus` (`configs/decoder.json`, 1.0) per character; the bonus is taken back if the word is not completed. Chosen on
the development set: given each clip's names plus 10 unrelated names, the share of names spelled correctly rose from 66% to
85%; with only unrelated names, WER rose by about 0.2 points.

**Word timestamps:** every result also has a `words` field: `[{"word": "Yani", "start": 0.0, "end": 0.28}, ...]` (seconds;
30 s chunks are shifted by their start time). The times come from the CTC frames (40 ms) of the best beam-search path, the
same method as the balbal Android app's captions (`inference/word_timing.py`). On 100 development clips, 94.5% of word starts
are within 100 ms of MMS forced alignment. From Python: `asr.transcribe('speech.wav')['words']`.

### Streaming (live transcription)

`inference/streaming.py` transcribes a recording while it grows (a microphone, a call), with the same method as the balbal
Android app: the audio is cut into the same 30 s chunks as offline transcription, a complete chunk is decoded once and
kept, and the open (last, growing) chunk is decoded again on every update (default: every 0.6 s of new audio). So each
update costs one decode of at most 30 s of audio. Text inside the open chunk can still change as more audio arrives; text of
closed chunks (`stable_text`) never changes, and `finish()` returns exactly what offline transcription gives for the whole
recording (words and timestamps included). balbal-v1 is a full-context model, so this is re-decoding rather than an
incremental streaming architecture.

```bash
arecord -q -f S16_LE -r 16000 -c 1 | python inference/streaming.py --model-dir MODEL_DIR -     # microphone (Linux)
python inference/streaming.py --model-dir MODEL_DIR --simulate speech.wav                     # a file, paced in real time
```

It prints one JSON line per update (`seconds`, `text`, `final`); the last line is the final result with `words`. Input on
stdin is raw 16-bit mono 16 kHz PCM. `--runtime torch` uses the PyTorch model instead of ONNX; `--hotwords` works as above.

From Python (either runtime):

```python
from runtime_onnx import Engine          # or: from transcribe import Transcriber
from streaming import Stream
stream = Stream(Engine('MODEL_DIR'), update_seconds=0.6, hotwords=None)
for piece in audio_pieces:               # mono float32 at 16 kHz, any length
    update = stream.feed(piece)          # None until update_seconds of new audio arrived
    if update:
        print(update['text'])
print(stream.finish()['text'])
```

Cost: on 4 CPU threads (ONNX), a 65 s recording fed in 0.25 s pieces (86 updates) took 21.5 s of compute in total, about
a third of real time.

## Steering

balbal-v1 can be steered toward a sentence from another source (for example the text of a larger ASR system). On every
refiner pass a small offset is added to the memory reads and optimised for a few Adam steps on CTC(sentence), only on the
frames where the model is unsure (top probability < 0.9). The weights stay frozen and the offset is thrown away after each
clip. The model only changes where the audio allows it: acoustically mismatched sentences are mostly rejected, and a sentence
much shorter than what balbal hears is ignored.

With Whisper-large-v3-turbo text (WER, balbal-normalizer, KenLM):

| Test set | balbal-v1 | Whisper-large-v3-turbo alone | balbal-v1 steered by Turbo |
|---|---|---|---|
| FLEURS tr test | 6.53 | 4.11 | **3.97** |
| General development set (Turkish) | 10.03 | 8.36 | **7.20** |
| Mu-Bench tr-TR | 11.79 | 20.73 | **11.08** |
| LibriSpeech test-clean (English) | 22.91 | **3.85** | 4.75 |

- Where both models are good (FLEURS, general development set), steering beats both.
- Where balbal-v1 is weak (English; balbal is a Turkish model), steering brings it close to the steering model: 22.91 → 4.75,
  Whisper alone 3.85.
- Where the steering model is weak (Mu-Bench phone calls; Whisper makes almost twice balbal's errors), steering does not hurt:
  balbal keeps what it hears clearly. This gain is not statistically significant (paired bootstrap 95% interval
  [-1.58, +0.06]).

| Mode | For | Settings (`steering/steer.json`) |
|---|---|---|
| `weak` | a sentence that may be wrong (another ASR, a guess) | 3 steps, lr 0.03, unsure frames only |
| `strong` | a sentence from a trusted source | 6 steps, lr 0.03 |
| `auto` | runs Whisper-large-v3-turbo itself and steers toward its text with `weak` | needs `transformers`; GPU recommended |

On 200 held-out clips (13.16 WER unsteered): `strong` with the correct sentence goes down to 1.74 (`weak` 4.04); with a
sentence where 15% of the words are wrong, `strong` gives 8.02 and `weak` 7.03. Steering takes about 0.25 s per clip on CPU;
Whisper takes tens of seconds per clip on CPU.

```bash
python steering/steer_transcribe.py --model-dir MODEL_DIR --mode auto speech.wav          # pip install transformers
python steering/steer_transcribe.py --model-dir MODEL_DIR --mode weak --text "Yarın saat 15'te toplantı var mı?" speech.wav
python steering/steer_transcribe.py --model-dir MODEL_DIR --mode strong --text "..." speech.wav
```

Every output line is a JSON object: `text` (steered), `plain_text` (unsteered), `target` (the sentence with digits turned
into words), `guarded` (true = the sentence was rejected) and timings. Keep casing and punctuation in the sentence
(normalized sentences steer about 0.6 WER worse). Audio longer than 30 s is cut into 30 s chunks.

## Normalizer

- `normalizer/turkish.py`: `normalize_turkish()`: lowercasing (Turkish İ/ı), punctuation, numbers/symbols/e-mail words; v1
  scoring and an exact-text key for excluding held-out sentences from the language model.
- `normalizer/metrics_v2.py`: **balbal-normalizer** scoring: the `turkish.py` normalization + fillers (ee, hı), "ile" fusion,
  circumflex, ordinals, units, apostrophes, joined/split spelling variants, "at/et" in e-mail addresses, letter-digit codes,
  and **numbers compared digit by digit** (`doksan` = `9 0` = `dokuz sıfır`). `corpus_metrics_v2(refs, hyps)` returns
  WER/CER.
- `normalizer/numbers.py`: `verbalize()`: turns digits in training labels into words (`180'e` → `yüz seksene`); returns
  `None` for ambiguous forms and the row is dropped. balbal-v1 was trained on **words-only** labels and does not write digits.
- `normalizer/label_clean.py`: cleans real-speech labels (stage directions, stutters, symbols/URLs, more than 25 characters
  per second, text repeated more than 5 times).
- `normalizer/itn.py`: inverse normalization: `itn_text('sıfır beş yüz otuz iki')` → `0 532`; `itn_digits()` gives the
  digit-by-digit form used in scoring.

## Training

balbal-v1 is a continued training run that starts from the weights of the previous stage (this stage: 9000 updates;
the released weights are the raw step-6000 weights):

```bash
TEL_P=0.5 EDGE_P=0.5 LOADER_WORKERS=8 python training/train.py runs/balbal-v1 \
    --init PREVIOUS_STAGE.pt --tokenizer-dir MODEL_DIR/tokenizer \
    --train train.jsonl --dev dev.jsonl \
    --updates 9000 --lr 1e-4 --eval-every 3000 --soup 0.5 --micro 16
```

- **Data** (`training/data.py`): JSONL rows (`audio_path`, `pcm_sha256`, `duration_seconds`, `text`, `source`), or virtual
  rows (`parts`) that join consecutive pieces of the same recording on the fly. Clips that do not fit CTC are dropped.
- **Augmentation**: telephone band (`TEL_P`: 8 kHz, 300 Hz high-pass, mu-law), edge shape (`EDGE_P`: trim silent edges, add
  0–0.6 s of digital zeros), NeMo SpecAugment.
- **Objective**: 0.3 × CTC(encoder) + 0.7 × mean CTC(3 refiner passes), bf16.
- **Optimization**: Muon (hidden linear layers, per head) + AdamW (the rest; memory value table at 5× learning rate), 10-step
  warm-up + cosine (1 → 0.1), gradient clipping 1.0, 32 clips per update.
- **Evaluation**: greedy WER (balbal-normalizer) on `--dev` every `--eval-every` steps, raw and as a `--soup` average; the best
  5 checkpoints are kept. (The original run also ranked checkpoints with KenLM on an internal DEV set; that part is not in
  this repository.)
- **Language model**: `training/build_lm.py`: 6-gram token KenLM over all training text the model saw; held-out sentences
  are excluded by exact match.
- **Calibration**: `training/calibrate.py`: a 3×3 grid of alpha × temperature on DEV; the default changes only if the paired
  bootstrap 95% confidence interval is entirely below 0 (it did not change for balbal-v1).
- **Soup**: `training/soup.py OUT.pt 0.5:a.pt 0.5:b.pt`.

On ROCm, MIOpen is off in training steps (about 41% faster) and on during evaluation.

## Citation

If you use balbal-v1 in your work, you can cite it as:

```bibtex
@misc{yapagci2026balbal,
  author       = {Eray Yapa{\u{g}}c{\i}},
  title        = {balbal-v1: A Small Turkish Speech Recognition Model},
  year         = {2026},
  month        = oct,
  howpublished = {\url{https://erayy.com/balbal-blog}},
  note         = {Code: \url{https://github.com/erayyap/balbal}}
}
```

## License

The code is licensed under the [Apache License 2.0](LICENSE) (Copyright 2026 Eray Yapağcı). The license covers only the
code in this repository. The model weights, tokenizer and language model are released separately under the
[balbal Community License 1.0](https://huggingface.co/erayyapagci/balbal-v1/blob/main/LICENSE) on
[Hugging Face](https://huggingface.co/erayyapagci/balbal-v1): free for research and non-commercial use, and for small
businesses' own commercial use; selling speech recognition products built on them needs a commercial license. Third-party
licenses: `inference/licenses/` (flashlight-text, KenLM).

---

## Türkçe özet

**balbal-v1** Türkçe konuşma tanıma modelinin kodu: Conformer-CTC (16 katman, d_model 176, SentencePiece 1024), 3 geçişli
2 katmanlı düzeltici ve ürün-anahtar akustik bellek (45.4M saklanan / 15.4M aktif parametre), 6-gram token KenLM ile çözülür.
Sonuçlar: FLEURS tr test **6.54 WER** (balbal-normalizer), Mu-Bench tr-TR **11.93 WER** (resmi puanlama).

Ağırlıklar Hugging Face'te: [erayyapagci/balbal-v1](https://huggingface.co/erayyapagci/balbal-v1)
([balbal Community License 1.0](https://huggingface.co/erayyapagci/balbal-v1/blob/main/LICENSE)). `inference/` çıkarım
(PyTorch ve ONNX), kelime zaman damgaları, kelime önceliği (`--hotwords`) ve canlı yazıya dökme (`streaming.py`) içerir;
`steering/` yönlendirme, `normalizer/` Türkçe normalizasyon ve puanlama, `training/` eğitim kodudur. Kullanım ayrıntıları
yukarıdaki İngilizce bölümlerde. Kod Apache 2.0 lisanslıdır.
