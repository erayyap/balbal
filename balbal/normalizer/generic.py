"""Language-neutral text normalization and corpus WER / CER, for languages without a scorer of their own (balbal.lang).

normalize: NFC, the language's lowercasing, every punctuation (P*) and symbol (S*) character -> space, whitespace collapsed.
Digits stay digits, so a reference with numbers as words scores against a hypothesis with digits as errors (keep both in one form).
"""
from __future__ import annotations

import unicodedata

from balbal.normalizer.turkish import PUNCT_SYMBOL_TO_SPACE, _WHITESPACE_RE, normalized_corpus_metrics


def normalize(text, lower=str.lower):
    value = unicodedata.normalize('NFC', '' if text is None else str(text))
    return _WHITESPACE_RE.sub(' ', lower(value).translate(PUNCT_SYMBOL_TO_SPACE)).strip()


def corpus_metrics(references, hypotheses, lower=str.lower):
    """Corpus WER / CER after normalize (same accounting as normalizer.turkish.normalized_corpus_metrics: empty refs add insertions)."""
    return normalized_corpus_metrics(references, hypotheses, normalize=lambda t: normalize(t, lower))
