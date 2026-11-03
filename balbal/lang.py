"""Per-language text rules: casing (hotword variants), number writing (ITN), the scoring normalizer and the exact-text key that keeps
held-out sentences out of the KenLM. A model folder names its languages in model.json ("languages": ["az"]); the first one is the
primary language and decides these rules.

  tr        Turkish dotted / dotless i casing, Turkish ITN, balbal-normalizer scoring (normalizer.metrics_v2)
  az        Turkish dotted / dotless i casing; no ITN; generic scoring
  others    str.lower casing; no ITN; generic scoring (normalizer.generic: lowercase, punctuation removed)

A new language is one entry in LANGUAGES: give it an ITN (a function returning (itn_text, itn_words), see normalizer/itn.py) or a
scorer of its own when the generic ones are not enough.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


def turkic_lower(text):
    return text.replace('I', 'ı').replace('İ', 'i').lower()


def turkic_capitalize(word):
    return (word[0].replace('i', 'İ').replace('ı', 'I').upper() + word[1:]) if word else word


def capitalize(word):
    return word[:1].upper() + word[1:]


@dataclass(frozen=True)
class Language:
    code: str
    lower: Callable[[str], str] = str.lower
    capitalize: Callable[[str], str] = capitalize
    itn: Optional[Callable[[], tuple]] = None        # () -> (itn_text, itn_words)
    scorer: Optional[Callable] = None                 # (references, hypotheses) -> {'wer', 'cer', ...}
    key: Optional[Callable[[str], str]] = None        # exact-text key of a sentence (held-out matching)

    def score(self, references, hypotheses):
        if self.scorer: return self.scorer(references, hypotheses)
        from balbal.normalizer.generic import corpus_metrics
        return corpus_metrics(references, hypotheses, self.lower)

    def text_key(self, text):
        if self.key: return self.key(text)
        from balbal.normalizer.generic import normalize
        return normalize(text, self.lower)


def _tr_itn():
    from balbal.normalizer.itn import itn_text, itn_words
    return itn_text, itn_words


def _tr_score(references, hypotheses):
    from balbal.normalizer.metrics_v2 import corpus_metrics_v2
    return corpus_metrics_v2(references, hypotheses)


def _tr_key(text):
    from balbal.normalizer.turkish import normalize_turkish
    return normalize_turkish(text)


LANGUAGES = {
    'tr': Language('tr', turkic_lower, turkic_capitalize, itn=_tr_itn, scorer=_tr_score, key=_tr_key),
    'az': Language('az', turkic_lower, turkic_capitalize),
}


def code(name):
    """'tr', 'tr-TR', 'tr_TR' -> 'tr'."""
    return str(name).strip().lower().replace('_', '-').split('-')[0]


def get(name):
    """The rules of a language code (unknown codes get the generic rules)."""
    c = code(name)
    return LANGUAGES.get(c) or Language(c)


def itn_functions(name):
    """(itn_text, itn_words) of a language code, or None when it has no ITN (None / False / '' -> None)."""
    if not name: return None
    language = get(name)
    return language.itn() if language.itn else None
