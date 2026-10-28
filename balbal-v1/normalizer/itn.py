"""Inverse text normalization: spoken Turkish number words -> digits.

balbal-v1 is trained on words-only labels, so it writes numbers as words ('bin dokuz yüz doksan beş'). Two views:

- itn_text(text): display form. Each maximal run of number words becomes its spoken groups as numbers
  ('sıfır doksan kırk beş altmış yedi' -> '0 90 45 67', 'dokuz sıfır bir' -> '9 0 1', 'bin dokuz yüz doksan beş' -> '1995').
  A lone ambiguous word (bir, on, yüz, altı, bin) stays a word. Punctuation is kept: a run ends at a token that carries
  punctuation, which is re-attached after the digits; capitalised words match case-insensitively. A suffixed number
  word that continues a run takes the suffix after an apostrophe ('kırk dokuzdu' -> "49'du"); a suffixed word on its
  own is left alone. The rest of the text is unchanged.
- itn_digits(text): scoring form for already-normalized (lowercase, no punctuation) text. Numbers are spelled digit by
  digit ('doksan yüz yirmi üç' == 'dokuz sıfır bir iki üç' == '9 0 1 2 3'), removing the reading/grouping ambiguity of
  phone numbers and codes; digits already present are kept.

Known ambiguity: 'doksan bir' for a spoken 'doksan, bir' reads as 91.
"""
import re

UNIT = dict(bir=1, iki=2, üç=3, dört=4, beş=5, altı=6, yedi=7, sekiz=8, dokuz=9)
TENS = dict(on=10, yirmi=20, otuz=30, kırk=40, elli=50, altmış=60, yetmiş=70, seksen=80, doksan=90)
NUM = set(UNIT) | set(TENS) | {'sıfır', 'yüz', 'bin', 'milyon'}
AMBIG = {'bir', 'on', 'yüz', 'altı', 'bin'}


def parse(words):
    """Number words -> list of numbers (as digit strings), splitting where the grammar cannot continue."""
    out, hi, lo, place = [], 0, 0, None   # place = smallest place filled in the open number (None = no open number)

    def close():
        nonlocal hi, lo, place
        if place is not None: out.append(str(hi + lo))
        hi, lo, place = 0, 0, None
    for w in words:
        if w == 'sıfır': close(); out.append('0'); continue
        if w in UNIT:
            if place is not None and place > 1: lo += UNIT[w]; place = 1
            else: close(); lo, place = UNIT[w], 1
        elif w in TENS:
            if place is not None and place > 10: lo += TENS[w]; place = 10
            else: close(); lo, place = TENS[w], 10
        elif w == 'yüz':
            if place == 1 and lo < 10: lo *= 100; place = 100
            elif place is not None and place > 100: lo += 100; place = 100
            else: close(); lo, place = 100, 100
        elif w == 'bin':
            if place is not None and place < 1000: hi, lo, place = hi + (lo or 1)*1000, 0, 1000
            else: close(); hi, place = 1000, 1000
        elif w == 'milyon':
            if place is not None and place < 10**6: hi, lo, place = (hi + lo or 1)*10**6, 0, 10**6
            else: close(); hi, place = 10**6, 10**6
    close(); return out


def itn_digits(text):
    """Normalized text -> number runs spelled digit by digit (for WER on numbers)."""
    toks = re.sub(r'(?<=\d)(?=\D)|(?<=\D)(?=\d)', ' ', text).split()
    res, i = [], 0
    while i < len(toks):
        if toks[i] in NUM or toks[i].isdigit():
            j = i
            while j < len(toks) and (toks[j] in NUM or toks[j].isdigit()): j += 1
            run = toks[i:j]
            if len(run) == 1 and run[0] in AMBIG: res.append(run[0])
            else:
                buf, words = [], []
                for t in run:
                    if t.isdigit(): buf += parse(words) + [t]; words = []
                    else: words.append(t)
                buf += parse(words); res += list(''.join(buf))
            i = j
        else: res.append(toks[i]); i += 1
    return ' '.join(res)


low = lambda w: w.replace('İ', 'i').replace('I', 'ı').lower()


def _split(tok):
    m = re.match(r'^(\W*)(.*?)(\W*)$', tok, re.S); return m.group(1), m.group(2), m.group(3)


def itn_text(text):
    """Model output -> the same text with spoken number runs written as numbers."""
    toks = text.split(); out, i = [], 0
    while i < len(toks):
        pre, core, post = _split(toks[i])
        if low(core) not in NUM: out.append(toks[i]); i += 1; continue
        j, words, last_post = i, [], ''
        while j < len(toks):
            p0, c, p1 = _split(toks[j])
            if low(c) not in NUM and j > i and not p0:   # suffixed last number word continuing a run: 'kırk dokuzdu' -> 49'du
                w = max((n for n in NUM if low(c).startswith(n) and re.fullmatch(r'[a-zçğıöşü]{1,4}', low(c)[len(n):])), key=len, default=None)
                if w: words.append(w); j += 1; last_post = "'" + c[len(w):] + p1
                break
            if low(c) not in NUM or (p0 and j > i): break
            words.append(low(c)); j += 1; last_post = p1
            if p1: break
        if len(words) == 1 and words[0] in AMBIG: out.append(toks[i]); i += 1; continue
        out.append(pre + ' '.join(parse(words)) + last_post); i = j
    return ' '.join(out)


if __name__ == '__main__':
    import sys
    for line in sys.stdin:
        print(itn_text(line.rstrip('\n')))
