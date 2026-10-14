"""KenLM beam decoder for balbal-v1's CTC log-probabilities.

Input: natural-log CTC probabilities [frames, 1025], blank at ID 1024. Settings come from configs/decoder.json
(6-gram token LM, alpha 0.75, beta 1.5, temperature 1.2, beam 64, top tokens 12, threshold 8).
The LM files (model.binary + tokenizer.model) are not part of this repository; pass their directory as `lm_dir`.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent
CONFIGS = ROOT.parent / 'configs'


def calibrate(array, temperature, blank_bias):
    """Temperature/blank-bias calibration, identical to the training-side decoder."""
    a = np.ascontiguousarray(array, dtype=np.float32)
    if temperature <= 0:
        raise ValueError('Positive temperature required')
    if temperature != 1 or blank_bias != 0:
        a = a.copy() / temperature
        a[:, -1] += blank_bias
        maximum = a.max(axis=1, keepdims=True)
        a -= maximum + np.log(np.exp(a - maximum).sum(axis=1, keepdims=True))
    if not np.isfinite(a).all():
        raise FloatingPointError('Nonfinite decoder input')
    return np.ascontiguousarray(a, dtype=np.float32)


class KenLMDecoder:
    """CPU decoder; reuse one instance per thread/process."""

    def __init__(self, lm_dir, settings=None):
        from flashlight.lib.text.decoder import KenLM, LexiconFreeDecoder, LexiconFreeDecoderOptions, CriterionType, ZeroLM
        from flashlight.lib.text.dictionary import Dictionary
        sys.path.insert(0, str(ROOT / 'native'))
        try:
            from ctc_lm_bonus import TokenBonusLM
        except ImportError as error:
            raise ImportError('Run python build_native.py in this environment first.') from error
        lm_dir = Path(lm_dir)
        self.settings = dict(json.loads((CONFIGS / 'decoder.json').read_text()), **(settings or {}))
        s = self.settings
        self.sp = spm.SentencePieceProcessor(model_file=str(lm_dir / 'tokenizer.model'))
        self.vocab_size = self.sp.vocab_size()
        self.blank_id = self.vocab_size
        self.dictionary = Dictionary([chr(i + 100) for i in range(self.vocab_size)] + ['<unk>'])
        self.lm = KenLM(str(lm_dir / 'model.binary'), self.dictionary) if s['alpha'] else ZeroLM()
        scale = s['alpha'] if s['alpha'] else 1.0
        self.wrapped_lm = TokenBonusLM(self.lm, s['beta'] / scale)
        options = LexiconFreeDecoderOptions(int(s['beam']), int(s['top_tokens']), float(s['threshold']),
                                            scale, 0.0, True, CriterionType.CTC)
        self.decoder = LexiconFreeDecoder(options, self.wrapped_lm, -1, self.blank_id, [])

    def decode(self, log_probs):
        """Return the best transcript for one utterance's [frames, 1025] log-probabilities."""
        a = np.asarray(log_probs)
        if a.ndim != 2 or a.shape[1] != self.vocab_size + 1 or not a.shape[0]:
            raise ValueError('Expected nonempty [frames, 1025] log-probabilities, blank last')
        a = calibrate(a, self.settings['temperature'], self.settings['blank_bias'])
        hypotheses = self.decoder.decode(a.ctypes.data, *a.shape)
        if not hypotheses:
            raise RuntimeError('Beam search returned no hypothesis')
        # Collapse adjacent repeats before removing blanks: a blank separates repeated tokens.
        ids = [int(t) for t, _ in itertools.groupby(hypotheses[0].tokens) if 0 <= t < self.vocab_size]
        return self.sp.decode(ids)
