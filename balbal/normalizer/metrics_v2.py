"""balbal-normalizer: the scoring normalizer (formerly called v2). Report next to v1 (normalizer.turkish); v1 stays the historical metric.

v1 is unchanged. balbal-normalizer = v1 plus these fixes, applied the same way to every model:
  tags        references only: remove annotated non-speech stage directions ("gülme efekti", "müzik", ...) listed in an
              optional normalizer/scoring_v2_reference_tags.json (not shipped; FLEURS and Mu-Bench need none).
              Hypotheses are NOT stripped: a model that writes "gülme" for laughter is charged an insertion.
  fillers     drop nonlexical hesitations on both sides: e+, ı+, hı+/hıhı, hm+, mm+, ım+, aa+ (lexical şey/yani stay;
              a lone capital I stays, it is English).
              A lone e/E is the letter E, not a filler, when its context says letter: previous word
              harf/harfi/harfim/... or değil, next word değil/olacak/ile/diyorsunuz/..., or a neighbouring single
              letter ("Z A Y N E P", "A değil E"). Otherwise it is still dropped as a filler.
  ile         "X ile" and fused "Xyle"/"Xle"/"Xyla"/"Xla" score the same (both sides fused by vowel harmony).
  circumflex  â/î/û = a/i/u (imkân/imkan, rüzgâr/rüzgar).
  ordinals    "15. sırada" = "on beşinci sırada" (digits + dot followed by a lowercase word).
  units       "100m" / "4892 m'lik" = "yüz metre" / "metrelik" (m, km, cm, mm, kg, mg, m², km² after a digit).
  apostrophe  suffix apostrophes join instead of splitting: Otoyolu'nun = otoyolunun.
  compound    space-only word-boundary differences are not errors: kuru yemiş = kuruyemiş, yurt dışı = yurtdışı
              (pairwise: adjacent words are joined only when the joined form is the other side's word).
              Never across a clitic (de/da, ki, mi/mı/mu/mü and its inflections): ben de/bende, kartı mı/kartımı
              are different words, so one side is wrong and it stays an error.
  email_at    in an e-mail address the @ word "at" is
              also said "et": "et" directly before "<domain> nokta" scores as "at" (both sides). Elsewhere "et" stays a word.
  codes       a token mixing letters and digits is a code: CEE2B2X3 / CN122 are split at letter|digit
              boundaries and short letter runs read by letter name (consonants ce/be/ne..., vowels as themselves), so they
              score like the spaced "C E E 2 B 2 X 3". Spelled: single letters, uppercase runs of <= 3, vowel-less runs;
              longer pronounceable runs stay words (FAYTO210 = fayto 2 1 0, ahmet1990 = ahmet 1 9 9 0). Digit + lowercase
              tail (3te, 100m) is a suffix/unit, not a code.
  digits      numbers compare DIGIT BY DIGIT, so
              the reading/grouping of a digit string is not an error: dokuz sıfır = doksan = 9 0, bir iki iki =
              yüz yirmi iki = 1 2 2, bin dokuz yüz doksan beş = bir dokuz dokuz beş = 1 9 9 5. Written digits are
              kept as digits (never verbalized, so the transcriber's grouping can't merge: '90 1' is 9 0 1, not 91);
              spoken number words are parsed to digits (a parse splits where Turkish number grammar can't continue);
              every digit is one token, but a run of zeros is one token (1 000000: a wrong 'milyon' costs 2, not 7).
              A digit may take a scale word (6 bin = 6000). A suffix on the last word of a number splits off (altı
              binine = 6000 ine; beşle = 5 ile). A number written alone with a suffix (3'te, üçte) stays a word as
              before. bir, on, yüz, altı, bin, yedi alone (no number next to them) are words unless the other side
              has that number's digits (pairwise); likewise yüzyıldan = 100 yıldan when the other side has it. A model
              writing "doksan bir" for a spoken "doksan, bir" still reads as 91 (no prosody in text).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from balbal.normalizer.turkish import (_DECIMAL_RE, _DIGIT_RUN_RE, _EMAIL_WORD_RE, _THOUSANDS_GROUP_RE, _WHITESPACE_RE,
                         PUNCT_SYMBOL_TO_SPACE, spell_letters, spoken_symbols, turkish_lower, turkish_number_words)

ALL_FIXES = ('tags', 'fillers', 'ile', 'circumflex', 'ordinals', 'units', 'apostrophe', 'compound', 'email_at', 'codes', 'digits')
_LONE_E_RE = re.compile(r"(?<![\w'’])[Ee](?![\w'’])")
_WORD_RE = re.compile(r"[\w'’]+")
_GENITIVE_RE = re.compile(r"['’]n?[ıiuü]n$")   # "E, Edirne'nin E'si"
_E_PREV = {'harf', 'harfi', 'harfim', 'harfin', 'harfini', 'harfimi', 'harfiniz', 'değil'}
_E_NEXT = {'değil', 'olacak', 'ile', 'diyorsunuz', 'dediniz', 'yazdınız', 'söylediniz', 'yerine', 'harfi',
           'olarak', 'var', 'yok', 'yazın', 'yazınız'}


def _single_letter(word):
    return word is not None and len(word) == 1 and word.isalpha() and word != 'o'


def _lone_e_is_letter(text, match):
    before = _WORD_RE.findall(text[:match.start()])
    after = _WORD_RE.findall(text[match.end():])
    prev = turkish_lower(before[-1]) if before else None
    nxt = turkish_lower(after[0]) if after else None
    return (prev in _E_PREV or nxt in _E_NEXT or (nxt is not None and _GENITIVE_RE.search(nxt) is not None)
            or _single_letter(before[-1] if before else None)
            or _single_letter(after[0] if after else None))


def _drop_filler_e(value):
    return _LONE_E_RE.sub(lambda m: m.group() if _lone_e_is_letter(value, m) else ' ', value)


_EMAIL_AT_RE = re.compile(r'(?<!\S)et(?= \S+ nokta(?!\S))')
_TAGS_PATH = Path(__file__).with_name('scoring_v2_reference_tags.json')
_TAGS = json.loads(_TAGS_PATH.read_text())['annotations'] if _TAGS_PATH.exists() else []
_RAW_FILLER_RE = re.compile(r'(?<![\w\'’])(?:[Ee]{2,}|ı+|[Iı]{2,}|(?:[Hh][Iı])+|[Hh][Iı]+|[Hh][Mm]+|[Mm]{2,}|[Iı][Mm]+|[Aa]{2,})(?![\w\'’])')
_CLITIC_RE = re.compile(r'(?:d[ae]|ki|m[iıuü](?:y?[dmsz]\w*)?)$')
_ORDINAL = {'bir': 'birinci', 'iki': 'ikinci', 'üç': 'üçüncü', 'dört': 'dördüncü', 'beş': 'beşinci',
            'altı': 'altıncı', 'yedi': 'yedinci', 'sekiz': 'sekizinci', 'dokuz': 'dokuzuncu', 'on': 'onuncu',
            'yirmi': 'yirminci', 'otuz': 'otuzuncu', 'kırk': 'kırkıncı', 'elli': 'ellinci', 'altmış': 'altmışıncı',
            'yetmiş': 'yetmişinci', 'seksen': 'sekseninci', 'doksan': 'doksanıncı', 'yüz': 'yüzüncü',
            'bin': 'bininci', 'milyon': 'milyonuncu', 'milyar': 'milyarıncı'}
_ORDINAL_RE = re.compile(r'(?<![\d.,])(\d+)\.\s+(?=[a-zçğıöşü])')
_UNITS = {'km²': 'kilometrekare', 'm²': 'metrekare', 'km': 'kilometre', 'cm': 'santimetre', 'mm': 'milimetre',
          'kg': 'kilogram', 'mg': 'miligram', 'm': 'metre'}
_UNIT_RE = re.compile(r'(?<=\d)\s?(km²|m²|km|cm|mm|kg|mg|m)(?![^\W\d_])')
_APOSTROPHE_RE = re.compile(r"(?<=\w)['’`´ʼ](?=\w)")
_BACK, _VOWELS = set('aıou'), set('aeıioöuü')


_VOWEL_LETTERS = set('aeıioöuü')
_CODE_TOKEN_RE = re.compile(r"(?<![\w'’])[^\W_]+(?![\w])")


def _spell_code(match):
    token = match.group()
    if token.isalpha() or token.isdigit(): return token   # fast path: letters and digits are disjoint
    if not (any(c.isdigit() for c in token) and any(c.isalpha() for c in token)): return token
    if token[0].isdigit() and not all(c.isupper() for c in token if c.isalpha()): return token   # 3te, 100m: suffix/unit
    parts = []
    for run in re.findall(r'\d+|[^\W\d_]+', token):
        if run[0].isdigit(): parts.append(run)
        elif len(run) > 1 and not (len(run) <= 3 and run.isupper()) and set(turkish_lower(run)) & _VOWEL_LETTERS:
            parts.append(turkish_lower(run))   # FAYTO210, ahmet1990: a pronounceable run is read as a word
        else:
            for c in turkish_lower(run):
                parts.append(c if c in _VOWEL_LETTERS else spell_letters(c.upper()).lower() if c.upper() in 'BCÇDFGĞHJKLMNPRSŞTVYZ' else c)
    return ' ' + ' '.join(parts) + ' '


_NUM_UNIT = {'bir': 1, 'iki': 2, 'üç': 3, 'dört': 4, 'beş': 5, 'altı': 6, 'yedi': 7, 'sekiz': 8, 'dokuz': 9}
_NUM_TENS = {'on': 10, 'yirmi': 20, 'otuz': 30, 'kırk': 40, 'elli': 50, 'altmış': 60, 'yetmiş': 70, 'seksen': 80, 'doksan': 90}
_NUM_SCALE = {'bin': 10**3, 'milyon': 10**6, 'milyar': 10**9}
_NUM_WORDS = set(_NUM_UNIT) | set(_NUM_TENS) | set(_NUM_SCALE) | {'sıfır', 'yüz'}
_NUM_AMBIGUOUS = {'bir', 'on', 'yüz', 'altı', 'bin', 'yedi'}   # also: a / ten / face / its bottom / get on / ate
_SUFFIX_AFTER_RE = re.compile(r"[^\W\d_]|['’`´ʼ][^\W\d_]")   # 3te, 3'te, 1990'larda: suffixed number, read as words
# Longest number word that is a proper prefix (alternatives longest first; equal-length prefixes of a word are equal).
_NUM_PREFIX_RE = re.compile('(?:' + '|'.join(sorted(_NUM_WORDS, key=len, reverse=True)) + ')(?=.)', re.S)
_ILE_TAIL_RE = re.compile(r'(.+?)(y?l[ae])')


def _parse_numbers(words):
    """Number words / digit tokens -> digit strings; a new number starts where the grammar can't continue (bir iki =
    1, 2; doksan yüz yirmi üç = 90, 123). A plain digit token may take a scale word (6 bin = 6000)."""
    out, high, low, place = [], 0, 0, None   # place = smallest place filled in the open number
    def close():
        nonlocal high, low, place
        if place is not None: out.append(str(high + low))
        high, low, place = 0, 0, None
    for w in words:
        if w.isdigit():
            close()
            if w.startswith('0') or len(w) > 3 or not w.isdecimal(): out.append(w)   # '²', '①': int() rejects them; passed through and dropped by _digit_split, as before for '0²'
            else: low, place = int(w), 1
        elif w == 'sıfır': close(); out.append('0')
        elif w in _NUM_UNIT:
            if place is not None and place > 1: low += _NUM_UNIT[w]; place = 1
            else: close(); low, place = _NUM_UNIT[w], 1
        elif w in _NUM_TENS:
            if place is not None and place > 10: low += _NUM_TENS[w]; place = 10
            else: close(); low, place = _NUM_TENS[w], 10
        elif w == 'yüz':
            if place == 1 and low < 10: low *= 100; place = 100
            elif place is not None and place > 100: low += 100; place = 100
            else: close(); low, place = 100, 100
        else:
            m = _NUM_SCALE[w]
            if place is not None and place < m: high, low, place = (high + (low or 1)*m) if m == 1000 else (high + (low or 1))*m, 0, m
            else: close(); high, place = m, m
    close(); return out


def _digit_split(digits):
    """One token per digit, a run of zeros is one token (1000000 = '1 000000': round numbers don't weigh by length)."""
    return re.findall(r'0{2,}|\d', digits)


def _number_prefix(word):
    """Longest number word that word starts with, and the rest (binine -> bin, ine)."""
    m = _NUM_PREFIX_RE.match(word)
    return (m.group(), word[m.end():]) if m else None


def _continues(words, word):
    """Does number word `word` extend the number that `words` ends with (altı + bin: yes; yüz + yüz: no)?"""
    k = len(words)
    while k and (words[k-1] in _NUM_WORDS or words[k-1].isdigit()): k -= 1
    run = words[k:]
    return len(_parse_numbers(run + [word])) == len(_parse_numbers(run))


def _digits_tokens(words):
    """Replace runs of number words / digit tokens by digit tokens; a lone ambiguous word stays a word."""
    numeric = lambda w: w in _NUM_WORDS or w.isdigit()
    unpacked = []
    for w in words:
        prev = unpacked[-1] if unpacked else None
        m = _ILE_TAIL_RE.fullmatch(w)
        if (m and m.group(1) in _NUM_WORDS and _fuse_ile(m.group(1)) == w
                and (m.group(1) not in _NUM_AMBIGUOUS or (prev and numeric(prev)))):
            unpacked += [m.group(1), 'ile']   # 'ile' fusion undone after a number (beşle = beş ile; the digit side keeps '5 ile')
        elif prev and numeric(prev) and not numeric(w) and (split := _number_prefix(w)) and split[1].isalpha() and _continues(unpacked, split[0]):
            unpacked += list(split)            # suffix on the last word of a number: altı binine = 6000 + ine
        else: unpacked.append(w)
    out, i = [], 0
    while i < len(unpacked):
        if numeric(unpacked[i]):
            j = i
            while j < len(unpacked) and numeric(unpacked[j]): j += 1
            run = unpacked[i:j]
            out += run if len(run) == 1 and run[0] in _NUM_AMBIGUOUS else _digit_split(''.join(_parse_numbers(run)))
            i = j
        else: out.append(unpacked[i]); i += 1
    return out


def _resolve_numbers_pairwise(words, other):
    """A lone ambiguous number word (yüz, bin, ...) becomes digits when the other side has exactly those digits there;
    a number word glued to a word (yüzyıldan) splits when the other side has the digits + the rest (100 yıldan)."""
    text = ' ' + ' '.join(other) + ' '
    out = []
    for k, w in enumerate(words):
        if w in _NUM_AMBIGUOUS and not (k and words[k-1][0].isdigit()) and not (k + 1 < len(words) and words[k+1][0].isdigit()):
            digits = ' '.join(_digit_split(''.join(_parse_numbers([w]))))
            if f' {digits} ' in text and f' {w} ' not in text: out += digits.split(); continue
        elif not w[0].isdigit() and w not in _NUM_WORDS and (split := _number_prefix(w)) and f' {w} ' not in text:
            glued = ' '.join(_digit_split(''.join(_parse_numbers([split[0]]))) + [split[1]])
            if f' {glued} ' in text: out += glued.split(); continue
        out.append(w)
    return out


def _ordinal(match):
    words = turkish_number_words(match.group(1)).split()
    words[-1] = _ORDINAL.get(words[-1], words[-1])
    return ' '.join(words) + ' '


def _fuse_ile(word):
    last = next((c for c in reversed(word) if c in _VOWELS), 'e')
    return word + ('y' if word[-1] in _VOWELS else '') + ('la' if last in _BACK else 'le')


def strip_reference_tags(text):
    for entry in _TAGS:
        if text.startswith(entry['prefix']):
            for tag in entry['tags']:
                text = re.sub(rf'(?<!\w){re.escape(tag)}(?!\w)', ' ', text, count=1)
    return text


def normalize_v2(text, *, reference, fixes=ALL_FIXES):
    """Single-side balbal-normalizer normalization (everything except the pairwise compound fix)."""
    value = unicodedata.normalize('NFC', '' if text is None else str(text))
    if 'tags' in fixes:
        if reference: value = strip_reference_tags(value)
    if 'circumflex' in fixes: value = value.translate(str.maketrans('âîûÂÎÛ', 'aiuAİU'))
    if 'units' in fixes: value = _UNIT_RE.sub(lambda m: ' ' + _UNITS[m.group(1)], value)
    if 'ordinals' in fixes: value = _ORDINAL_RE.sub(_ordinal, value)
    if 'fillers' in fixes: value = _drop_filler_e(_RAW_FILLER_RE.sub(' ', value))   # before letter spelling turns hmm into he me me
    # normalizer.turkish.normalize_turkish, with suffix apostrophes optionally joined before punctuation becomes space.
    if 'codes' in fixes: value = _CODE_TOKEN_RE.sub(_spell_code, value)
    value = turkish_lower(spell_letters(spoken_symbols(value)))
    value = _EMAIL_WORD_RE.sub(r'e\1', value)
    value = _THOUSANDS_GROUP_RE.sub(lambda m: re.sub(r'[.,]', '', m.group()), value)
    value = _DECIMAL_RE.sub(lambda m: f'{m.group(1)} virgül {m.group(2) or m.group(3)}', value)
    value = _DIGIT_RUN_RE.sub(lambda m: f' {m.group()} ' if 'digits' in fixes and not _SUFFIX_AFTER_RE.match(m.string, m.end())
                              else turkish_number_words(m.group()), value)   # digits fix: unsuffixed numbers stay digits
    if 'apostrophe' in fixes: value = _APOSTROPHE_RE.sub('', value)
    value = value.translate(PUNCT_SYMBOL_TO_SPACE)
    words = _WHITESPACE_RE.sub(' ', value).strip().split()
    if 'ile' in fixes:
        out = []
        for w in words:
            if w == 'ile' and out and not ('digits' in fixes and out[-1].isdigit()): out[-1] = _fuse_ile(out[-1])
            else: out.append(w)
        words = out
    if 'digits' in fixes: words = _digits_tokens(words)
    value = ' '.join(words)
    if 'email_at' in fixes: value = _EMAIL_AT_RE.sub('at', value)
    return value


def _join_pairs(words, targets, own):
    out, i = [], 0
    while i < len(words):
        if (i + 1 < len(words) and words[i] + words[i + 1] in targets and words[i] + words[i + 1] not in own
                and not _CLITIC_RE.match(words[i + 1])):
            out.append(words[i] + words[i + 1]); i += 2
        else:
            out.append(words[i]); i += 1
    return out


def align_compounds(ref, hyp):
    """Join adjacent words on one side when the joined form is a word of the other side (yurt dışı/yurtdışı)."""
    r, h = ref.split(), hyp.split()
    h = _join_pairs(h, set(r), set(h))
    r = _join_pairs(r, set(h), set(r))
    return ' '.join(r), ' '.join(h)


def normalize_pair_v2(reference, hypothesis, fixes=ALL_FIXES):
    ref, hyp = normalize_v2(reference, reference=True, fixes=fixes), normalize_v2(hypothesis, reference=False, fixes=fixes)
    if 'digits' in fixes:
        r, h = ref.split(), hyp.split()
        ref, hyp = ' '.join(_resolve_numbers_pairwise(r, h)), ' '.join(_resolve_numbers_pairwise(h, r))
    return align_compounds(ref, hyp) if 'compound' in fixes else (ref, hyp)


def corpus_metrics_v2(references, hypotheses, fixes=ALL_FIXES):
    """Same accounting as normalizer.turkish.normalized_corpus_metrics (empty refs add insertions), balbal-normalizer normalization."""
    import jiwer
    if not references or len(references) != len(hypotheses):
        raise ValueError('nonempty aligned references and hypotheses required')
    we = ce = nw = nc = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref, hyp = normalize_pair_v2(reference, hypothesis, fixes)
        if ref.strip():
            words, chars = jiwer.process_words(ref, hyp), jiwer.process_characters(ref, hyp)
            we += words.substitutions + words.deletions + words.insertions
            ce += chars.substitutions + chars.deletions + chars.insertions
            nw += words.hits + words.substitutions + words.deletions
            nc += chars.hits + chars.substitutions + chars.deletions
        else:
            we += len(hyp.split()); ce += len(hyp.strip())
    return {'wer': we / max(1, nw), 'cer': ce / max(1, nc), 'word_errors': we, 'reference_words': nw}
