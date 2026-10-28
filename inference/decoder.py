"""KenLM beam decoder for the CTC log-probabilities of balbal-v2 / balbal-v2-manas.

Input: natural-log CTC probabilities [frames, V + 1], blank last. Settings: configs/<model>/decoder.json (6-gram token KenLM, alpha,
beta, temperature, beam 64, top tokens 12, threshold 4, blank_collapse .99). Before the search the emissions are calibrated
(temperature, blank bias) and runs of confident blank frames (p(blank) > blank_collapse) are shortened to their last frame; word
timestamps are mapped back to the original frames.

Two searches with the same best path:
  fast        native/ctc_fast_decoder (default): flashlight's LexiconFreeDecoder search, but hypotheses are visited best first and a
              token is skipped once even the largest possible LM score could not keep it in the beam, so fewer LM queries
  flashlight  flashlight-text's LexiconFreeDecoder itself (fast=False)
Optional hotwords (names/words expected in the audio) get a per-character bonus while decoding; see decode_words().
The LM directory holds model.binary (KenLM) and tokenizer.model (SentencePiece; piece i is the LM word chr(i + 100)).
"""
from __future__ import annotations
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from word_timing import token_spans, words_from_spans


def prepare(array, settings):
    """Calibrated emissions with confident-blank runs shortened, and the original frame index of every kept row."""
    temperature, bias = settings.get('temperature', 1.), settings.get('blank_bias', 0.)
    if temperature <= 0: raise ValueError('Positive temperature required')
    a = np.ascontiguousarray(array, dtype=np.float32)
    if temperature != 1 or bias != 0:
        a = a / temperature
        a[:, -1] += bias
        maximum = a.max(axis=1, keepdims=True)
        a = a - (maximum + np.log(np.exp(a - maximum).sum(axis=1, keepdims=True)))
    kept = np.arange(len(a))
    collapse = settings.get('blank_collapse')
    if collapse is not None:
        if not 0 < collapse < 1: raise ValueError('blank_collapse must be a probability')
        confident = a[:, -1] > math.log(collapse)
        # Keep the last blank of every confident run: the blank separating two repeated tokens is never dropped.
        keep = np.ones(len(a), dtype=bool); keep[:-1] = ~(confident[:-1] & confident[1:])
        a, kept = a[keep], kept[keep]
    if not np.isfinite(a).all(): raise FloatingPointError('Nonfinite decoder input')
    return np.ascontiguousarray(a, dtype=np.float32), kept


class KenLMDecoder:
    """CPU decoder; reuse one instance per thread/process."""

    def __init__(self, lm_dir, settings, fast=True):
        import sentencepiece as spm
        from flashlight.lib.text.decoder import KenLM, ZeroLM
        from flashlight.lib.text.dictionary import Dictionary
        sys.path.insert(0, str(HERE / 'native'))
        try:
            from ctc_lm_bonus import HotwordLM, TokenBonusLM
            if fast: from ctc_fast_decoder import FastCTCDecoder
        except ImportError as error:
            raise ImportError('Run python inference/build_native.py in this environment first.') from error
        lm_dir = Path(lm_dir)
        self.settings, self.fast = dict(settings), fast
        s = self.settings
        self.sp = spm.SentencePieceProcessor(model_file=str(lm_dir / 'tokenizer.model'))
        self.vocab_size = self.sp.vocab_size()
        self.blank_id = self.vocab_size
        self.dictionary = Dictionary([chr(i + 100) for i in range(self.vocab_size)] + ['<unk>'])
        self.lm = KenLM(str(lm_dir / 'model.binary'), self.dictionary) if s['alpha'] else ZeroLM()
        self.scale = s['alpha'] if s['alpha'] else 1.0
        self.bonus = s['beta'] / self.scale
        self.wrapped_lm = TokenBonusLM(self.lm, self.bonus)
        self._HotwordLM = HotwordLM
        self._pieces = {self.sp.id_to_piece(i): i for i in range(self.vocab_size)}
        self.decoder = self._make(self.wrapped_lm, self.bonus)
        self._hot = (None, None)   # (hotword list + bonus, its decoder): reused while the list stays the same

    def _make(self, lm, upper):
        """A search over `lm`; upper = the largest score lm can return for a token (KenLM log10 <= 0, plus bonuses)."""
        s = self.settings
        if self.fast:
            from ctc_fast_decoder import FastCTCDecoder
            fd = FastCTCDecoder(int(s['beam']), int(s['top_tokens']), float(s['threshold']), self.scale, lm, self.blank_id, upper)
            return lambda a: list(fd.decode(a.ctypes.data, *a.shape))
        from flashlight.lib.text.decoder import CriterionType, LexiconFreeDecoder, LexiconFreeDecoderOptions
        options = LexiconFreeDecoderOptions(int(s['beam']), int(s['top_tokens']), float(s['threshold']), self.scale, 0.0, True,
                                            CriterionType.CTC)
        decoder = LexiconFreeDecoder(options, lm, -1, self.blank_id, [])
        def run(a):
            hypotheses = decoder.decode(a.ctypes.data, *a.shape)
            if not hypotheses: raise RuntimeError('Beam search returned no hypothesis')
            return list(hypotheses[0].tokens)
        run.keep = (lm, decoder)
        return run

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
                    text, found = '▁' + variant, []
                    # ends[i]: text[i:] can be segmented; without it a character missing from the vocabulary makes the
                    # search below exponential in the word length (every prefix segmentation is tried and fails).
                    ends = [False] * len(text) + [True]
                    for i in range(len(text) - 1, -1, -1):
                        ends[i] = any(ends[j] and text[i:j] in self._pieces for j in range(i + 1, len(text) + 1))
                    def walk(i, path):
                        if len(found) >= 64:
                            return
                        if i == len(text):
                            found.append(tuple(path)); return
                        for j in range(len(text), i, -1):
                            piece = self._pieces.get(text[i:j])
                            if piece is not None and ends[j]:
                                walk(j, path + [piece])
                    walk(0, [])
                    seqs.update(found)
        seqs = sorted(seqs)
        bonuses = [[bonus_per_char * len(self.sp.id_to_piece(t)) / self.scale for t in q] for q in seqs]
        return [list(q) for q in seqs], bonuses

    def decode(self, log_probs, hotwords=None):
        """Best transcript for one utterance's [frames, V + 1] log-probabilities (see decode_words)."""
        return self.decode_words(log_probs, hotwords)['text']

    def decode_words(self, log_probs, hotwords=None, offset_seconds=0.):
        """Best transcript plus per-word timestamps: {'text', 'words': [{'word', 'start', 'end'}]} (seconds, shifted by offset_seconds).

        hotwords: optional list of words/names expected in the audio (e.g. ['Eray', 'Hasan']). Tokens that extend a hotword earn
        `hotword_bonus` (decoder.json) per character; the bonus is taken back if the word is not completed, so only whole hotwords
        are rewarded."""
        a = np.asarray(log_probs)
        if a.ndim != 2 or a.shape[1] != self.vocab_size + 1 or not a.shape[0]:
            raise ValueError(f'Expected nonempty [frames, {self.vocab_size + 1}] log-probabilities, blank last')
        a, kept = prepare(a, self.settings)
        decoder = self.decoder
        if hotwords:
            # A search over the shared KenLM per hotword list, kept for the next call with the same list (chunks of one file,
            # streaming updates); the beam state resets on every decode.
            key = tuple(hotwords), float(self.settings.get('hotword_bonus', 1.0))
            hot = self._hot
            if hot[0] != key:
                seqs, bonuses = self.hotword_sequences(*key)
                lm = self._HotwordLM(self.wrapped_lm, seqs, bonuses)
                hot = self._hot = key, self._make(lm, self.bonus + max([max(b) for b in bonuses if b] + [0.]))
                hot[1].keep_lm = lm
            decoder = hot[1]
        path = decoder(a)
        # Collapse adjacent repeats before removing blanks: a blank separates repeated tokens. Frames back to the original grid.
        spans = [(t, int(kept[f0]), int(kept[f1])) for t, f0, f1 in token_spans(path, self.vocab_size)]
        return dict(text=self.sp.decode([t for t, _, _ in spans]), words=words_from_spans(spans, self.sp, offset_seconds))
