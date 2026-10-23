#!/usr/bin/env python3
"""Deterministic Turkish digit verbalisation for TRAINING transcripts. balbal-v1 is trained on words-only labels: a model
trained on mixed digit/word labels learned to write tens words as single digits ('doksan' -> '9'), so every digit in a
training transcript is spelled out, and rows whose digits cannot be read unambiguously are dropped.
verbalize(text) -> words text, or None when any digit token is not in a safe form (the row is then dropped, never guessed):
  safe: integer without leading zero (or '0') [+ apostrophe suffix: 180'e -> yüz seksene, 4'ü -> dördü], '%N' -> yüzde N,
  'N.' + lowercase word -> ordinal (10. yüzyıl -> onuncu yüzyıl), runs of single spaced digits read one by one (2 0 3 8).
  unsafe -> None: 'N.' before a capitalised word (ordinal or sentence end?), suffix vowel-harmony typos (53'u), leading zeros (0532), decimals / thousands dots (1,2  1.000  26.0), times (14:30), ranges (3-4), glued
  alphanumerics (Paket24, Adsl2), more than 12 digits.
balbal-normalizer scoring equates number words and digits, so a words-only model is not penalised on digit references."""
import re
ONES = ['', 'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz']
TENS = ['', 'on', 'yirmi', 'otuz', 'kırk', 'elli', 'altmış', 'yetmiş', 'seksen', 'doksan']
SCALES = ['', 'bin', 'milyon', 'milyar']
ORD = {'bir': 'birinci', 'iki': 'ikinci', 'üç': 'üçüncü', 'dört': 'dördüncü', 'beş': 'beşinci', 'altı': 'altıncı', 'yedi': 'yedinci',
       'sekiz': 'sekizinci', 'dokuz': 'dokuzuncu', 'on': 'onuncu', 'yirmi': 'yirminci', 'otuz': 'otuzuncu', 'kırk': 'kırkıncı',
       'elli': 'ellinci', 'altmış': 'altmışıncı', 'yetmiş': 'yetmişinci', 'seksen': 'sekseninci', 'doksan': 'doksanıncı',
       'yüz': 'yüzüncü', 'bin': 'bininci', 'milyon': 'milyonuncu', 'milyar': 'milyarıncı', 'sıfır': 'sıfırıncı'}
VOW = set('aeıioöuüAEIİOÖUÜ'); FRONT, BACK = set('eiöüEİÖÜ'), set('aıouAIOU')

def _harmony_ok(words, suf):
    """Suffix front/back vowel harmony vs the number word (source typos like 53'u, 20'un, 2'da -> drop)."""
    last = [c for c in words if c in VOW]; sv = [c for c in suf if c in VOW]
    if not last or not sv: return True
    return (last[-1] in FRONT) == (sv[0] in FRONT)

def _three(n):
    h, t, o = n // 100, n // 10 % 10, n % 10
    return [w for w in ((['yüz'] if h == 1 else [ONES[h], 'yüz']) if h else []) + [TENS[t], ONES[o]] if w]

def cardinal(n):
    if n == 0: return 'sıfır'
    out, k = [], 0
    while n:
        n, g = divmod(n, 1000)
        if g: out = (['bin'] if k == 1 and g == 1 else _three(g) + ([SCALES[k]] if k else [])) + out
        k += 1
    return ' '.join(out)

def _suffix(words, suf):
    last = words.split()[-1]
    if suf and suf[0] in VOW and last == 'dört': words = words[:-4] + 'dörd'   # dört'ü -> dördü
    return words + suf

INT = re.compile(r"^(0|[1-9]\d{0,11})(?:['’]([a-zçğıöşü]+))?$")
def verbalize(text):
    if not re.search(r'\d', text): return text
    toks = text.split(' '); out = []
    for i, tok in enumerate(toks):
        if not re.search(r'\d', tok): out.append(tok); continue
        m0 = re.match(r'^([("“\'‘]*)(.*?)([.,;:!?)"”\'’…]*)$', tok, re.S); pre, core, post = m0.groups()
        nxt = toks[i+1] if i + 1 < len(toks) else ''
        if core.startswith('%'): pct, core = True, core[1:]
        else: pct = False
        m = INT.match(core)
        if not m:
            return None
        words = cardinal(int(m.group(1)))
        if m.group(2):
            if not _harmony_ok(words, m.group(2)): return None
            words = _suffix(words, m.group(2))
        elif post == '.' and nxt[:1].isupper():   # '1. Lig' (ordinal) vs sentence-final '... 1990. Sonra' is ambiguous (audit 2026-10-09)
            return None
        elif post == '.' and nxt[:1].islower() and not pct:   # ordinal: 10. yüzyıl
            ws = words.split(); words = ' '.join(ws[:-1] + [ORD[ws[-1]]]); post = ''
        if pct: words = 'yüzde ' + words
        out.append(pre + words + post)
    return ' '.join(out)

if __name__ == '__main__':
    for t in ["Artı 90 123 0 90 45 67.", "Yani 180 derecenin 90'ı, 180'e eşit.", "MÖ 10. yüzyılda", "4'ü aldım, %50 indirim",
              "1990 yılında 1000 kişi", "kod M N B C 2 8.", "1,2 milyon", "0532 123", "Paket24", "saat 14:30", "2 0 3 8 at hotmail"]:
        print(t, '->', verbalize(t))
