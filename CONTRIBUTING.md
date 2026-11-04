# Contributing to balbal

Thanks for helping. Issues and pull requests are welcome in English or Turkish.

## Before you start

For anything bigger than a small fix, open an issue first and describe the change.

Welcome:
- Platform support: build scripts and docs for macOS, Windows, ARM Linux
- Bug fixes in inference, `balbal serve`, streaming, the decoder and the training scripts
- Docs, examples and clear bug reports

Not accepted:
- Model weights, datasets, audio or transcripts
- Changes to model licensing
- Data generation and augmentation pipelines (out of scope for this repository)

## Pull requests

- One topic per PR, small and self-contained.
- Put new files next to the code they belong to (native builds under `inference/`, wheel changes under `pypi/`). If unsure, ask in the issue.
- Don't break the Linux x86_64 build or the published wheels.
- Run the tests that cover your change:
  - `pytest pypi/tests` (unit tests; `BALBAL_TEST_MODELS=1` also runs the model tests)
  - `python inference/kernels/test_kernels.py` if you touch the custom ONNX ops
  - `python inference/streaming/test_equiv.py --model-dir MODEL_DIR a.wav ...` if you touch streaming
- If you report accuracy, give WER and CER, the test set size and where it came from, and say whether hotwords or LM text overlap the test set.
- Never commit personal data, keys or private recordings.

## Sign-off

Sign each commit with `git commit -s`. The `Signed-off-by:` line certifies you wrote the change
or have the right to submit it ([Developer Certificate of Origin](https://developercertificate.org/)).

## License

Code in this repository is Apache License 2.0. By submitting a contribution you agree it is licensed
under Apache 2.0 (Section 5 of `LICENSE`). Model weights have their own license (see the model cards).
Contributing code gives no extra rights to the weights. If you add third-party code, keep its license
file and add a line to `NOTICE`.
