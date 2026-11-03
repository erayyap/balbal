# Changelog

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
