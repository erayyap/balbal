# balbal

Open Turkish speech recognition that runs on a CPU, plus an experimental Moroccan Darija model.

```bash
pip install balbal
balbal run speech.wav
```

The first run downloads the model from Hugging Face (balbal-v2: ~590 MB, once); later runs start from the local cache.

| Model | Parameters | Languages | Mu-Bench WER / sigWER | Speed (8 CPU threads) |
|---|---|---|---|---|
| **balbal-v2** (default) | 45.4M | Turkish | 9.46% / 18.44% | 236x real time |
| **balbal-v2-manas** | 313.3M | Turkish, English | 8.20% / 14.89% | 123x real time |

**balbal-v2.1-zellige** (`--model zellige`) is an experimental Moroccan Darija (Arabic script) model, ~1.4 GB download, offline transcription only (no
`balbal live`, no WebSocket streaming). Its weights are released under **CC BY-NC-SA 4.0: non-commercial use only**, not the balbal
Community License 1.1 of the two models above; balbal prints a notice naming the license the first time it downloads it. It writes the
decoded Arabic text as is (no casing, no number writing).

Output: cased, punctuated text with per-word timestamps and confidences; spoken numbers come out as digits. Files, folders,
subtitles, live microphone input and an OpenAI-compatible server are covered below.

## Command line

```bash
balbal run speech.wav                         # text
balbal run --model manas a.wav b.mp3          # balbal-v2-manas; MP3 / M4A / video through FFmpeg
balbal run --json speech.wav                  # one JSON line per file: text, words with start / end / confidence, timings
balbal run --hotwords "Eray,Hasan" call.wav   # favour names / words you expect in the audio
balbal run --domain-lm my-lm/ notes.wav       # mix your own domain KenLM into the general one (see Domain LM below)
balbal run --no-itn speech.wav                # keep numbers as spoken words
balbal run -f srt -o subs/ videos/            # one subtitle file per audio / video file in a folder (txt, json, srt, vtt)
balbal run -j 4 -o out/ calls/                # four processes at once; --skip-existing resumes a stopped batch
balbal download --model manas                 # fetch the model files now
```

`--threads` sets the CPU threads (default `BALBAL_THREADS`, else up to 8).

## Python

```python
import balbal

print(balbal.transcribe('speech.wav')['text'])

model = balbal.load('balbal-v2-manas')          # load once, reuse
result = model.transcribe('call.wav', hotwords=['Eray'])
for w in result['words']:
    print(w['start'], w['end'], w['word'])

model.transcribe(samples, sample_rate=8000)      # a NumPy waveform at any sample rate
```

`transcribe` returns `text`, `words` (`word`, `start`, `end` in seconds, `confidence`), `duration_seconds`,
`inference_seconds` and `realtime_factor`. `confidence` (0-1) comes from the acoustic model: on our hand-checked phone
recordings, wrong words average about 0.4 and correct words about 0.85. `no_speech` is true when no word reaches 0.15
(silence, noise, or a short word written over digital silence); the text is left as decoded. Audio longer than 30 s is read in overlapping windows whose
CTC frames are stitched into one sequence and decoded in one KenLM pass, so no word is cut at a window edge. One `Model` per thread.

## Domain LM

For words the general KenLM rarely sees (medicine, law, product names), build a small KenLM on your own text with
`training/build_lm.py` from the [GitHub repository](https://github.com/erayyap/balbal) and mix it in:

```bash
python training/build_lm.py --tokenizer MODEL_DIR/lm/tokenizer.model --text my-sentences.txt --out my-lm
balbal run --domain-lm my-lm --domain-weight 0.3 dictation.wav      # also balbal live / balbal serve
```

```python
model = balbal.load('balbal-v2', settings={'domain_lm': 'my-lm', 'domain_weight': 0.3})
```

The two LMs are mixed linearly, `p = (1 - w) p_general + w p_domain`; tune the weight (default 0.3) on recordings whose text is
not in the domain LM. `--tokenizer` must be the model's own (`lm/tokenizer.model` in its folder; `balbal download` prints the
folder). A model folder's `decoder.json` can hold the same keys (`"domain_lm"` relative to the folder).

## Models in other languages

Models trained on the balbal architectures with the [GitHub code](https://github.com/erayyap/balbal#other-languages) load the same
way, by Hugging Face repo id or local folder:

```python
model = balbal.load('owner/name')               # or balbal.load('path/to/model-folder')
```

```bash
balbal run --model owner/name speech.wav
```

The repo or folder needs `model.json`, `lm/` and `onnx/` (exported with `inference/export_onnx.py`); `decoder.json` there replaces the
default decoder settings. `model.json` names the architecture and the languages, e.g.
`{"model": "balbal-v2-manas", "name": "my-model", "languages": ["az"]}`. Spoken numbers become digits only for languages with a
number writer (Turkish); other languages keep the words the model writes.

## Live transcription

```bash
pip install 'balbal[live]'                     # microphone access (Linux also needs PortAudio: sudo apt install libportaudio2)
balbal live                                    # speak; Ctrl+C ends and prints the transcript
balbal live --model manas --timestamps
balbal live --file speech.wav                  # a recording played as if live (no extra needed)
```

```python
stream = model.stream()                          # one per live audio stream
for piece in pieces:                             # float32 or int16, mono, 16 kHz, any size
    for event in stream.accept(piece):
        print(event['type'], event['text'])      # 'partial' (still changing) or 'final' (a closed segment)
stream.finish()
print(stream.transcript())
```

The models see the whole input at once, so live mode re-runs the model on the open segment every 0.32 s. Text older than about
0.64 s no longer changes; a segment closes after a 0.5 s pause or at 5 s. A partial's `stable` field holds its words that will
not change. `model.stream('window')` uses a sliding window instead.

## OpenAI-compatible server

```bash
pip install 'balbal[server]'
balbal serve --model manas --port 8000            # http://127.0.0.1:8000/v1, add --api-key KEY to require a key
```

Any OpenAI client works by pointing it at the server; OpenAI model names (`whisper-1`, `gpt-4o-transcribe`) map to the
served model:

```python
from openai import OpenAI

client = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='unused')
print(client.audio.transcriptions.create(model='whisper-1', file=open('call.mp3', 'rb')).text)
```

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions -F file=@call.wav -F response_format=srt
```

`POST /v1/audio/transcriptions` takes `file` (WAV / FLAC / OGG, other formats through FFmpeg, up to `--max-mb` 25),
`model`, `response_format` (`json`, `text`, `verbose_json`, `srt`, `vtt`) and `timestamp_granularities[]` (`word`,
`segment`); `prompt`, `temperature` and `language` are accepted and do not change the result. balbal extras: `hotwords`
(`"Eray,Hasan"`) and `itn` (`false` keeps numbers as words). `GET /v1/models` lists the served models (`--model` is
repeatable; the first one is the default). `--workers N` runs N transcriptions at once, each with its own model copy and
`--threads` CPU threads. With `verbose_json`, words carry `confidence` and segments an `avg_logprob`.

Live audio goes over a WebSocket at `ws://HOST:PORT/v1/stream`. Send binary little-endian int16 PCM (mono, 16 kHz, any piece
size), then the text `{"type": "end"}`. The server answers `ready`, then `partial` / `final` events as above, then `done` with
the transcript. Query parameters: `model`, `mode`, `hotwords`, `itn`; with an API key, send `Authorization: Bearer KEY` or
`api_key=KEY`. `http://HOST:PORT/` is a browser microphone demo of it (`/#key=KEY` when a key is set).

## Home Assistant

```bash
pip install 'balbal[wyoming]'
balbal wyoming                                   # Wyoming protocol on port 10300; --model manas for Turkish + English
```

In Home Assistant: Settings → Devices & services → Add integration → Wyoming Protocol, enter this machine's address and port
10300, then pick balbal as the speech-to-text engine of a voice assistant. Commands without confident speech return an empty
transcript. `--hotwords "salon,mutfak"` favours words in every command. `--use-ha-names` also favours the names Home Assistant
sends with a request.

## Docker

```bash
docker build -t balbal .                                   # --build-arg PRELOAD=balbal-v2 puts the model in the image
docker run -p 8000:8000 -v balbal-models:/models balbal     # balbal serve; options after the image: serve --model manas --api-key KEY
docker run -p 10300:10300 -v balbal-models:/models balbal wyoming
```

## Platform

Linux x86-64, macOS (Apple Silicon, macOS 14+) and Windows x86-64, Python 3.11 or 3.12. The wheel carries the native beam-search
decoder and the custom onnxruntime operators of the fast graph (on x86-64 built for x86-64-v2, AVX2 and AVX-512 CPUs, chosen at run
time). `runtime='onnx'` (`--runtime onnx`) runs the standard-operator graph instead.

On Windows, onnxruntime and the decoder need the Microsoft Visual C++ Redistributable (x64), which most machines already have; if
`import balbal` fails with a DLL error, install it from https://aka.ms/vs/17/release/vc_redist.x64.exe. MP3, M4A and video files
are read through FFmpeg on every platform (`ffmpeg` on the PATH); WAV, FLAC and OGG need nothing else.

Files go to `~/.cache/huggingface/balbal/` (`HF_HOME`, `BALBAL_CACHE` or `cache_dir=` move it). The model revisions are pinned to the ones this release was tested
with; `revision='main'` takes the latest upload.

## Links

- Model page: https://erayy.com/balbal
- Weights: [erayyapagci/balbal-v2](https://huggingface.co/erayyapagci/balbal-v2), [erayyapagci/balbal-v2-manas](https://huggingface.co/erayyapagci/balbal-v2-manas), [erayyapagci/balbal-v2.1-zellige](https://huggingface.co/erayyapagci/balbal-v2.1-zellige)
- Code (inference, streaming, training): https://github.com/erayyap/balbal

The code is Apache-2.0. The model weights have their own license (balbal Community License 1.1 for balbal-v2 and balbal-v2-manas, CC BY-NC-SA 4.0 for balbal-v2.1-zellige; on the model pages).
