"""KenLM beam decoder for balbal-v1's CTC log-probabilities.

Input: natural-log CTC probabilities [frames, 1025], blank at ID 1024. Settings come from configs/decoder.json
(6-gram token LM, alpha 0.75, beta 1.5, temperature 1.2, beam 64, top tokens 12, threshold 8).
Optional hotwords (names/words expected in the audio) get a per-character bonus while decoding; see decode().
The LM files (model.binary + tokenizer.model) are not part of this repository; pass their directory as `lm_dir`.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from word_timing import token_spans, words_from_spans
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
            from ctc_lm_bonus import HotwordLM, TokenBonusLM
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
        self._options, self._scale, self._HotwordLM, self._Decoder = options, scale, HotwordLM, LexiconFreeDecoder
        self._pieces = {self.sp.id_to_piece(i): i for i in range(self.vocab_size)}

    def hotword_sequences(self, words, bonus_per_char):
        """Token sequences for hotword biasing: every vocabulary segmentation (up to 64) of each word as written,
        lowercased and capitalized (Turkish dotted/dotless i), so a word earns its bonus however the model spells it.
        Each token's bonus is bonus_per_char x its characters (word-start marker included); phrases are split into words."""
        lower = lambda w: w.replace('I', 'ı').replace('İ', 'i').lower()
        upper1 = lambda w: (w[0].replace('i', 'İ').replace('ı', 'I').upper() + w[1:]) if w else w
        seqs = set()
        for phrase in words:
            for word in str(phrase).split():
                for variant in {word, lower(word), upper1(lower(word))}:
                    text, found = '\u2581' + variant, []
                    def walk(i, path):
                        if len(found) >= 64:
                            return
                        if i == len(text):
                            found.append(tuple(path)); return
                        for j in range(len(text), i, -1):
                            piece = self._pieces.get(text[i:j])
                            if piece is not None:
                                walk(j, path + [piece])
                    walk(0, [])
                    seqs.update(found)
        seqs = sorted(seqs)
        bonuses = [[bonus_per_char * len(self.sp.id_to_piece(t)) / self._scale for t in q] for q in seqs]
        return [list(q) for q in seqs], bonuses

    def decode(self, log_probs, hotwords=None):
        """Return the best transcript for one utterance's [frames, 1025] log-probabilities (see decode_words)."""
        return self.decode_words(log_probs, hotwords)['text']

    def decode_words(self, log_probs, hotwords=None, offset_seconds=0.):
        """Best transcript plus per-word timestamps: {'text', 'words': [{'word', 'start', 'end'}]} (seconds, shifted
        by offset_seconds; see word_timing.py).

        hotwords: optional list of words/names expected in the audio (e.g. ['Arslan', 'Zaynep']). Tokens that extend a
        hotword earn `hotword_bonus` (configs/decoder.json) per character; the bonus is taken back if the word is not
        completed, so only whole hotwords are rewarded. Chosen on the development set: with each clip's names plus 10
        unrelated names, name recall rose from 66% to 85%; unrelated names alone cost about +0.2 WER."""
        a = np.asarray(log_probs)
        if a.ndim != 2 or a.shape[1] != self.vocab_size + 1 or not a.shape[0]:
            raise ValueError('Expected nonempty [frames, 1025] log-probabilities, blank last')
        a = calibrate(a, self.settings['temperature'], self.settings['blank_bias'])
        decoder = self.decoder
        if hotwords:
            # Per-call decoder over the shared KenLM; the hotword trie lives only for this utterance.
            sequences = self.hotword_sequences(hotwords, float(self.settings.get('hotword_bonus', 1.0)))
            decoder = self._Decoder(self._options, self._HotwordLM(self.wrapped_lm, *sequences), -1, self.blank_id, [])
        hypotheses = decoder.decode(a.ctypes.data, *a.shape)
        if not hypotheses:
            raise RuntimeError('Beam search returned no hypothesis')
        # Collapse adjacent repeats before removing blanks: a blank separates repeated tokens.
        spans = token_spans(hypotheses[0].tokens, self.vocab_size)
        return dict(text=self.sp.decode([t for t, _, _ in spans]), words=words_from_spans(spans, self.sp, offset_seconds))
