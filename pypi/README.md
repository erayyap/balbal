# balbal

Open Turkish speech recognition that runs on a CPU.

```bash
pip install balbal
balbal run speech.wav
```

The first run downloads the model from Hugging Face (balbal-v2: ~590 MB, once); later runs start from the local cache.

| Model | Parameters | Languages | Mu-Bench WER / sigWER | Speed (8 CPU threads) |
|---|---|---|---|---|
| **balbal-v2** (default) | 45.4M | Turkish | 9.46% / 18.44% | 236x real time |
| **balbal-v2-manas** | 313.3M | Turkish, English | 8.20% / 14.89% | 123x real time |

Output: cased, punctuated text with per-word timestamps; spoken numbers come out as digits.

## Command line

```bash
balbal run speech.wav                         # text
balbal run --model manas a.wav b.mp3          # balbal-v2-manas; MP3 / M4A / video through FFmpeg
balbal run --json speech.wav                  # one JSON line per file: text, words with start / end seconds, timings
balbal run --hotwords "Eray,Hasan" call.wav   # favour names / words you expect in the audio
balbal run --no-itn speech.wav                # keep numbers as spoken words
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

`transcribe` returns `text`, `words` (`word`, `start`, `end` in seconds), `duration_seconds`, `inference_seconds` and
`realtime_factor`. Audio longer than 30 s is cut into 30 s chunks. One `Model` per thread.

## Platform

Linux x86-64, Python 3.11 or 3.12. The wheel carries the native beam-search decoder and the custom onnxruntime operators of the
fast graph (built for x86-64-v2, AVX2 and AVX-512 CPUs, chosen at run time). `runtime='onnx'` (`--runtime onnx`) runs the
standard-operator graph instead.

Files go to `~/.cache/huggingface/balbal/` (`HF_HOME`, `BALBAL_CACHE` or `cache_dir=` move it). The model revisions are pinned to the ones this release was tested
with; `revision='main'` takes the latest upload.

## Links

- Model page: https://erayy.com/balbal
- Weights: [erayyapagci/balbal-v2](https://huggingface.co/erayyapagci/balbal-v2), [erayyapagci/balbal-v2-manas](https://huggingface.co/erayyapagci/balbal-v2-manas)
- Code (inference, streaming, training): https://github.com/erayyap/balbal

The code is Apache-2.0. The model weights have their own license (balbal Community License 1.0, on the model pages).
