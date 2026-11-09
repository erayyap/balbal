"""Per-language text rules of a model: casing (hotword variants) and number writing (ITN). A model folder names its languages in
model.json ("languages": ["az"]); the first one is the primary language and picks these rules.

  tr        Turkish dotted / dotless i casing, Turkish ITN (balbal.itn)
  az        Turkish dotted / dotless i casing, no ITN
  others    str.lower casing, no ITN
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
    itn: Optional[Callable[[], tuple]] = None   # () -> (itn_text, itn_words)


def _tr_itn():
    from .itn import itn_text, itn_words
    return itn_text, itn_words


LANGUAGES = {
    'tr': Language('tr', turkic_lower, turkic_capitalize, itn=_tr_itn),
    'az': Language('az', turkic_lower, turkic_capitalize),
}
NAMES = {'tr': 'turkish', 'en': 'english', 'az': 'azerbaijani', 'kk': 'kazakh', 'uz': 'uzbek', 'ky': 'kyrgyz', 'tk': 'turkmen',
         'tt': 'tatar', 'ba': 'bashkir', 'ug': 'uyghur', 'ar': 'arabic', 'ary': 'moroccan arabic', 'grc': 'ancient greek', 'fa': 'persian', 'ru': 'russian',
         'de': 'german', 'fr': 'french', 'es': 'spanish'}


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
