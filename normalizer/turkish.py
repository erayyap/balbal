"""Turkish text normalization (v1 scoring) and corpus WER/CER. normalize_turkish() is also the exact-text key used to keep held-out sentences out of the language model."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Sequence


_WHITESPACE_RE = re.compile(r"\s+")
_THOUSANDS_GROUP_RE = re.compile(r"(?<!\d)\d{1,3}(?:[.,]\d{3})+(?!\d)")
_DIGIT_RUN_RE = re.compile(r"\d+")

_UNITS = ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
_SCALES = [(1_000_000_000, "milyar"), (1_000_000, "milyon"), (1_000, "bin")]


def _under_thousand(number: int) -> list[str]:
    words: list[str] = []
    hundreds, remainder = divmod(number, 100)
    if hundreds:
        if hundreds > 1:
            words.append(_UNITS[hundreds])
        words.append("yüz")
    tens, units = divmod(remainder, 10)
    if tens:
        words.append(_TENS[tens])
    if units:
        words.append(_UNITS[units])
    return words


def turkish_number_words(digits: str) -> str:
    """Expand a digit run for scoring without parsing ambiguous number words."""
    if len(digits) > 1 and digits.startswith("0"):
        return " ".join(_UNITS[int(char)] for char in digits)
    if len(digits) > 9:
        return " ".join(_UNITS[int(char)] for char in digits)
    number = int(digits)
    if number == 0:
        return _UNITS[0]
    words: list[str] = []
    for scale, label in _SCALES:
        count, number = divmod(number, scale)
        if count:
            if not (scale == 1_000 and count == 1):
                words.extend(_under_thousand(count))
            words.append(label)
    words.extend(_under_thousand(number))
    return " ".join(words)


# Spelled codes: references write letters ("C N 1 2 2", "NTV") while speech and some
# models give the spoken names ("ce ne"). Scoring maps both to the Turkish letter name.
_LETTER_NAMES = {"b": "be", "c": "ce", "ç": "çe", "d": "de", "f": "fe", "g": "ge", "ğ": "yumuşak ge",
                 "h": "he", "j": "je", "k": "ke", "l": "le", "m": "me", "n": "ne", "p": "pe", "r": "re",
                 "s": "se", "ş": "şe", "t": "te", "v": "ve", "y": "ye", "z": "ze"}
_CONSONANTS = "BCÇDFGĞHJKLMNPRSŞTVYZ"
# Turkish words always contain a vowel, so a vowel-less run is read letter by letter
# (NTV, PTT, chp'yle, TL'ye) in either case. Y counts as a vowel here so English
# "my"/"rynch" stay words; unit abbreviations are read as words ("70 km").
_RUN_CONSONANTS = _CONSONANTS.replace("Y", "")
_CONSONANT_RUN_RE = re.compile(rf"(?<![\w'’])[{_RUN_CONSONANTS}{_RUN_CONSONANTS.lower()}]{{2,5}}(?![\w])")
_SPOKEN_ABBREVIATIONS = {"km": "kilometre", "cm": "santimetre", "kg": "kilogram", "mg": "miligram", "dr": "doktor"}
# A consonant standing alone (or with a suffix, Z'den) is a spelled letter; the "t" of
# "didn't" follows an apostrophe and is not.
_LONE_LETTER_RE = re.compile(rf"(?<![\w'’])[{_CONSONANTS}{_CONSONANTS.lower()}](?![\w])")


def _letter_name(letter: str) -> str:
    return _LETTER_NAMES[turkish_lower(letter)]


def _spell_run(match: re.Match) -> str:
    run = match.group()
    return _SPOKEN_ABBREVIATIONS.get(turkish_lower(run)) or " ".join(map(_letter_name, run))


def spell_letters(text: str) -> str:
    return _LONE_LETTER_RE.sub(lambda match: _letter_name(match.group()), _CONSONANT_RUN_RE.sub(_spell_run, text))


# Spoken symbols: references write what was said ("ahmet nokta arslan at gmail nokta com",
# "yüzde yirmi", "45 bölü 6", "iki virgül beş"); models often write the symbol. Both
# sides map to the spoken word. Times (12.00, 6.30) and single-letter abbreviations
# (A.B.D.) are left alone.
_SPOKEN_SYMBOLS = [
    (re.compile(r"(?<=[^\W\d_]{2})\.(?=[^\W\d_]{2})"), " nokta "),
    (re.compile(r"@"), " at "),
    (re.compile(r"%\s?(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s?%"), lambda m: f" yüzde {m.group(1) or m.group(2)} "),
    (re.compile(r"\+(?=\s?\d)"), " artı "),
    (re.compile(r"(?<![\d/])(\d+)\s?/\s?(\d+)(?![\d/])"), r"\1 bölü \2"),
]
_DECIMAL_RE = re.compile(r"(?<![\d.,])(\d+)(?:,(\d{1,2})|\.(\d))(?![\d])(?![.,]\d)")
_EMAIL_WORD_RE = re.compile(r"(?<!\w)e[- ]?(mail|posta)")


def spoken_symbols(text: str) -> str:
    for pattern, replacement in _SPOKEN_SYMBOLS:
        text = pattern.sub(replacement, text)
    return text


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def expand_turkish_numbers_preserve_orthography(text: object) -> str:
    """Expand digit runs without changing casing, punctuation, or apostrophes.

    This is a training-target transform, unlike ``normalize_turkish`` which is
    deliberately destructive and intended only for metric scoring.
    """
    value = unicodedata.normalize("NFC", "" if text is None else str(text))
    value = _THOUSANDS_GROUP_RE.sub(
        lambda match: re.sub(r"[.,]", "", match.group()), value
    )
    return _DIGIT_RUN_RE.sub(
        lambda match: turkish_number_words(match.group()), value
    )


def normalize_turkish(text: object) -> str:
    value = unicodedata.normalize("NFC", "" if text is None else str(text))
    value = turkish_lower(spell_letters(spoken_symbols(value)))
    value = _EMAIL_WORD_RE.sub(r"e\1", value)
    # Expand digits to spoken Turkish so `62` and `altmış iki` score equally.
    # Remove only unambiguous thousands separators first; keep time/date separators
    # so 08:15 becomes `sıfır sekiz on beş`, not the unrelated integer 815.
    value = _THOUSANDS_GROUP_RE.sub(lambda match: re.sub(r"[.,]", "", match.group()), value)
    value = _DECIMAL_RE.sub(lambda m: f"{m.group(1)} virgül {m.group(2) or m.group(3)}", value)
    value = _DIGIT_RUN_RE.sub(lambda match: turkish_number_words(match.group()), value)
    normalized: list[str] = []
    for character in value:
        category = unicodedata.category(character)
        if category.startswith("P") or category.startswith("S"):
            normalized.append(" ")
        else:
            normalized.append(character)
    return _WHITESPACE_RE.sub(" ", "".join(normalized)).strip()


def normalized_corpus_metrics(references, hypotheses) -> dict[str, float]:
    """Uncapped corpus metrics; empty refs contribute insertions, not fake words."""
    import jiwer
    if not references or len(references) != len(hypotheses):
        raise ValueError('nonempty aligned references and hypotheses required')
    report = {}
    for prefix, refs, hyps in [
        ('', [normalize_turkish(x) for x in references], [normalize_turkish(x) for x in hypotheses]),
        ('raw_', references, hypotheses),
    ]:
        # jiwer 3 rejects empty references; count their insertions explicitly.
        we = ce = nw = nc = 0
        for ref, hyp in zip(refs, hyps):
            if ref.strip():
                words = jiwer.process_words(ref, hyp)
                chars = jiwer.process_characters(ref, hyp)
                we += words.substitutions + words.deletions + words.insertions
                ce += chars.substitutions + chars.deletions + chars.insertions
                nw += words.hits + words.substitutions + words.deletions
                nc += chars.hits + chars.substitutions + chars.deletions
            else:
                we += len(hyp.split())
                ce += len(hyp.strip())
        report.update({prefix + 'wer': we / max(1, nw), prefix + 'cer': ce / max(1, nc),
                       prefix + 'word_errors': we, prefix + 'reference_words': nw})
    return report
