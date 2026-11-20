"""Inverse text normalization: spoken Turkish number words -> digits.

The balbal models are trained on words-only labels, so it writes numbers as words ('bin dokuz yüz doksan beş'). Two views:

- itn_text(text): display form (itn_words: the same over timed words). Each maximal run of number words becomes its spoken groups as numbers
  ('sıfır doksan kırk beş altmış yedi' -> '0 90 45 67', 'dokuz sıfır bir' -> '9 0 1', 'bin dokuz yüz doksan beş' -> '1995').
  A lone ambiguous word (bir, on, yüz, altı, bin) stays a word. Punctuation is kept: a run ends at a token that carries
  punctuation, which is re-attached after the digits; capitalised words match case-insensitively. A suffixed number
  word that continues a run takes the suffix after an apostrophe ('kırk dokuzdu' -> "49'du"); a suffixed word on its
  own is left alone. The rest of the text is unchanged.
  Then the number-writing conventions of Turkish call transcripts (0.6.4): a single suffixed number word is written with digits and an
  apostrophe ('saat ikide' -> "saat 2'de", 'beşten' -> "5'ten", 'ayın onunda' -> "ayın 10'unda", 'yirmi dördüne' -> "24'üne");
  possessive forms (dokuzu, yirmisinde) and the ambiguous stems bir / on / yüz / bin (biri, onu, ona) only after a time or date word, ordinals
  (ikinci), distributives (beşer) and the idiom 'ikide bir' stay words; a lone on / yüz / bin before a unit ('on gün', 'bin lira') or as the
  whole text becomes digits; 'yüzde 20' -> '%20'; a half hour after a time word -> 'saat 8.30', "sabah 8.30'da".
- itn_digits(text): scoring form for already-normalized (lowercase, no punctuation) text. Numbers are spelled digit by
  digit ('doksan yüz yirmi üç' == 'dokuz sıfır bir iki üç' == '9 0 1 2 3'), removing the reading/grouping ambiguity of
  phone numbers and codes; digits already present are kept.

  0.6.5: a date read with digits splits from the year ('sıfır dokuz bin dokuz yüz altmış üç' -> '0 9 1963', 'yirmi üç dokuz bin dokuz yüz
  seksen' -> '23 9 1980'), and acronyms are joined: 2-3 spaced capitals ('T L' -> 'TL', 'E F T' -> 'EFT'; not next to a digit, not in a
  letter-by-letter spelling, never 'A Ş') and runs of letter names ('es em es' -> 'SMS', 'de ne es' -> 'DNS', 'be de de ka' -> 'BDDK') when
  they hold at least two unambiguous letter names (or three names with one).

Known ambiguity: 'doksan bir' for a spoken 'doksan, bir' reads as 91.
"""
import re

UNIT = dict(bir=1, iki=2, üç=3, dört=4, beş=5, altı=6, yedi=7, sekiz=8, dokuz=9)
TENS = dict(on=10, yirmi=20, otuz=30, kırk=40, elli=50, altmış=60, yetmiş=70, seksen=80, doksan=90)
NUM = set(UNIT) | set(TENS) | {'sıfır', 'yüz', 'bin', 'milyon'}
AMBIG = {'bir', 'on', 'yüz', 'altı', 'bin'}


def parse(words):
    """Number words -> list of numbers (as digit strings), splitting where the grammar cannot continue.
    Date splits: 'sıfır dokuz bin dokuz yüz ...' -> 0 9 19.. (a digit read after sıfır does not multiply a following bin), and in a run
    that already holds a number, '<n> bin dokuz yüz ...' -> n 19.. (day / month before a 19xx year: 'yirmi üç dokuz bin dokuz yüz seksen')."""
    out, hi, lo, place = [], 0, 0, None   # place = smallest place filled in the open number (None = no open number)

    def close():
        nonlocal hi, lo, place
        if place is not None: out.append(str(hi + lo))
        hi, lo, place = 0, 0, None
    for k, w in enumerate(words):
        nxt = words[k + 1] if k + 1 < len(words) else ''
        if w == 'sıfır': close(); out.append('0'); continue
        if w in UNIT:
            if k > 0 and words[k - 1] == 'sıfır' and nxt == 'bin': close(); out.append(str(UNIT[w])); continue
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
            year = out and hi == 0 and place is not None and lo < 100 and words[k + 1:k + 3] == ['dokuz', 'yüz']
            if place is not None and place < 1000 and not year: hi, lo, place = hi + (lo or 1)*1000, 0, 1000
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
    return ' '.join(tok for tok, _, _ in _itn_runs(text.split()))


def itn_words(words):
    """itn_text over timed words [{'word', 'start', 'end'(, 'confidence')}]: a number run becomes one entry spanning its words (with
    the lowest confidence of them)."""
    out = []
    for tok, i, j in _itn_runs([w['word'] for w in words]):
        w = dict(words[i], word=tok, end=words[j - 1]['end'])
        if j - i > 1 and 'confidence' in w: w['confidence'] = min(x['confidence'] for x in words[i:j])
        out.append(w)
    return out


def _itn_runs(toks):
    """Tokens -> [(output token, first input index, end input index)]: number runs, the writing conventions (_conventions), letter runs (_letters)."""
    return _letters(_conventions(_number_runs(toks)))


VAL = {**UNIT, **TENS, 'yüz': 100, 'bin': 1000}
STEMS = sorted({**{w: w for w in VAL}, 'dörd': 'dört'}.items(), key=lambda kv: -len(kv[0]))   # longest first; dörd = softened dört
CASE = {'de', 'da', 'te', 'ta', 'e', 'a', 'ye', 'ya', 'den', 'dan', 'ten', 'tan'}
POSS = {'i', 'ı', 'u', 'ü', 'si', 'sı', 'su', 'sü', 'yi', 'yı', 'yu', 'yü'} | {
    p + c for p in ('i', 'ı', 'u', 'ü', 'si', 'sı', 'su', 'sü') for c in ('nde', 'nda', 'ne', 'na', 'ni', 'nı', 'nu', 'nü', 'nden', 'ndan')}
STEM_AMBIG = {'bir', 'on', 'yüz', 'bin'}                       # biri, onu, ona, onda, yüzü, bini: pronoun / noun readings
CTX_STRONG = {'saat', 'saati', 'ayın'}                          # context that makes those stems numbers
CTX = CTX_STRONG | {'sabah', 'akşam', 'öğlen', 'öğle', 'öğleden', 'gece', 'tarih', 'tarihi', 'tarihinde', 'günü'}
TIME = {'saat', 'sabah', 'akşam', 'öğlen', 'öğle', 'öğleden', 'sonra', 'gece', 'yarın', 'bugün'}
UNITW = {'gün', 'günde', 'günü', 'gündür', 'saat', 'saattir', 'dakika', 'dakikada', 'dakikadır', 'saniye', 'yaşında', 'yaş', 'yaşındayım', 'lira',
         'tl', 'liralık', 'lirası', 'lirasını', 'tane', 'kişi', 'kişiyiz', 'ay', 'aydır', 'yıl', 'yıldır', 'sene', 'senedir', 'hafta', 'haftadır',
         'kilo', 'metre', 'adet', 'kere', 'defa', 'sefer', 'kat', 'numara', 'puan', 'dolar', 'euro', 'avro', 'gb', 'mb', 'taksit', 'ile', 'bin', 'milyon'}
LONE = {'on': 10, 'yüz': 100, 'bin': 1000}
BUCUK = {'buçuk': '', 'buçukta': "'da", 'buçuğa': "'a", 'buçuktan': "'dan"}


def _suffixed(core):
    """'dördüne' -> (4, 'üne', 'poss', 'dört'); None unless a number stem plus a case / possessive suffix."""
    lc = low(core)
    for st, w in STEMS:
        if lc.startswith(st) and len(lc) > len(st):
            suf = lc[len(st):]
            if suf in CASE or suf in POSS: return VAL[w], core[len(st):], ('case' if suf in CASE else 'poss'), w
    return None


def _conventions(runs):
    """Number-writing conventions over the (token, i, j) runs; merged entries span both inputs (itn_words keeps the timing)."""
    out = []
    for k, (tok, i, j) in enumerate(runs):
        pre, core, post = _split(tok); lc = low(core)
        pp, pc, ppost = _split(out[-1][0]) if out else ('', '', '')
        prev = low(pc); attached = bool(out) and not ppost and not pre   # previous token ends without punctuation
        nxt = low(_split(runs[k + 1][0])[1]) if k + 1 < len(runs) else ''
        if core.isdigit() and prev == 'yüzde' and attached:                                       # yüzde 20 -> %20
            out[-1] = (f'{pp}%{core}{post}', out[-1][1], j); continue
        if lc in BUCUK and prev.isdigit() and attached and int(prev) <= 24 and len(out) >= 2 and low(_split(out[-2][0])[1]) in TIME:
            out[-1] = (f'{pp}{prev}.30{BUCUK[lc]}{post}', out[-1][1], j); continue                # saat 8 buçukta -> saat 8.30'da
        s = None if (pre or j - i > 1 or (lc == 'ikide' and nxt == 'bir')) else _suffixed(core)
        if s:
            v, suf, kind, w = s
            if prev.isdigit() and attached and int(prev) >= 10 and int(prev) % 10 == 0 and v < 10:   # '20 dördüne' -> 24'üne
                out[-1] = (f"{pp}{int(prev) + v}'{suf}{post}", out[-1][1], j); continue
            ctx = prev in (CTX_STRONG if w in STEM_AMBIG else CTX) or (prev.isdigit() and attached)
            if (kind == 'case' and w not in STEM_AMBIG) or ctx:
                out.append((f"{v}'{suf}{post}", i, j)); continue
        if lc in LONE and not pre and not prev.isdigit() and not (len(prev) == 1 and prev.isalpha()) and (nxt in UNITW or len(runs) == 1):   # on gün -> 10 gün; 'x bin lira' stays
            out.append((f'{LONE[lc]}{post}', i, j)); continue
        out.append((tok, i, j))
    return out


# Letter runs (0.6.5). The models write an acronym as spaced capitals ('T L', 'E F T') or as letter names ('es em es', 'be de de ka').
TR = {'be': 'B', 'ce': 'C', 'çe': 'Ç', 'de': 'D', 'fe': 'F', 'ge': 'G', 'ğe': 'Ğ', 'he': 'H', 'je': 'J', 'ka': 'K', 'ke': 'K', 'le': 'L', 'me': 'M',
      'ne': 'N', 'pe': 'P', 're': 'R', 'se': 'S', 'şe': 'Ş', 'te': 'T', 've': 'V', 'ye': 'Y', 'ze': 'Z', 'a': 'A', 'e': 'E', 'ı': 'I', 'i': 'İ',
      'o': 'O', 'ö': 'Ö', 'u': 'U', 'ü': 'Ü'}
EN = {'es': 'S', 'em': 'M', 'en': 'N', 'el': 'L', 'ef': 'F', 'pi': 'P', 'ti': 'T', 'di': 'D', 'ey': 'A', 'ay': 'I', 'bi': 'B', 'kyu': 'Q', 'vi': 'V'}
NAME = {**TR, **EN}
STRONG = {'be', 'ce', 'çe', 'ge', 'je', 'ka', 'ke', 'le', 'me', 'pe', 're', 'se', 'şe', 'te', 'fe', 'ze', 'es', 'em', 'ef', 'pi', 'ti', 'di', 'ey', 'kyu'}
EDGE = {'ve', 'ki', 'o', 'a', 'e', 'en', 'el', 'ay', 'bi', 'he', 'ye', 'ı', 'i', 'u', 'ü', 'ö'}
CAP = re.compile(r'^[A-ZÇĞİÖŞÜ]$')
_isnum = lambda c: bool(c) and c.replace(' ', '').isdigit()   # a number run token can hold spaces ('0 532')


def _spelling(runs):
    """True when the text spells something letter by letter: a spaced-capital run of >= 4, or a single capital next to a digit token."""
    cores = [_split(t)[1] for t, _, _ in runs]; run = 0
    for x, c in enumerate(cores):
        if CAP.match(c):
            run += 1
            if run >= 4: return True
            if x + 1 < len(cores) and _isnum(cores[x + 1]): return True
        else: run = 0
    return False


def _letters(runs):
    """[(token, i, j)] -> same with letter runs joined (spans merged)."""
    out, k, n = [], 0, len(runs); spell = _spelling(runs)
    while k < n:
        pre, core, post = _split(runs[k][0])
        # L1 spaced capitals
        if CAP.match(core) and not post and k + 1 < n:
            m = k
            while m + 1 < n:
                p0, c1, p1 = _split(runs[m + 1][0])
                if p0 or not CAP.match(c1): break
                m += 1
                if p1: break
            joined = ''.join(_split(runs[x][0])[1] for x in range(k, m + 1))
            if 1 <= m - k <= 2 and not spell and joined != 'AŞ' and not (m + 1 < n and _isnum(_split(runs[m + 1][0])[1])):
                cs = [_split(runs[x][0]) for x in range(k, m + 1)]
                out.append((pre + ''.join(c for _, c, _ in cs) + cs[-1][2], runs[k][1], runs[m][2])); k = m + 1; continue
        # L2 letter names
        if low(core) in NAME and not pre:
            m = k
            while m + 1 < n and not _split(runs[m][0])[2]:
                p0, c1, p1 = _split(runs[m + 1][0])
                if p0 or low(c1) not in NAME: break
                m += 1
            a, b = k, m
            while a < b and low(_split(runs[a][0])[1]) in EDGE: a += 1
            while b > a and low(_split(runs[b][0])[1]) in EDGE: b -= 1
            ok = lambda a, b: (lambda nm: len(nm) >= 2 and (sum(x in STRONG for x in nm) >= 2 or (len(nm) >= 3 and any(x in STRONG for x in nm))))(
                [low(_split(runs[x][0])[1]) for x in range(a, b + 1)])
            if ok(a, b):   # v2: a de / ne at either edge is dropped when the rest still qualifies ('ne es em es' -> ne SMS; 'de ne es' -> DNS)
                while low(_split(runs[a][0])[1]) in ('de', 'ne') and ok(a + 1, b): a += 1
                while low(_split(runs[b][0])[1]) in ('de', 'ne') and ok(a, b - 1): b -= 1
            names = [low(_split(runs[x][0])[1]) for x in range(a, b + 1)]
            if ok(a, b):
                out.extend(runs[k:a])
                out.append((''.join(NAME[x] for x in names) + _split(runs[b][0])[2], runs[a][1], runs[b][2]))
                out.extend(runs[b + 1:m + 1]); k = m + 1; continue
        out.append(runs[k]); k += 1
    return out


def _number_runs(toks):
    """Tokens -> [(output token, first input index, end input index)] for maximal number-word runs (the 0.6.3 ITN)."""
    out, i = [], 0
    while i < len(toks):
        pre, core, post = _split(toks[i])
        if low(core) not in NUM: out.append((toks[i], i, i + 1)); i += 1; continue
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
        if len(words) == 1 and words[0] in AMBIG: out.append((toks[i], i, i + 1)); i += 1; continue
        out.append((pre + ' '.join(parse(words)) + last_post, i, j)); i = j
    return out


if __name__ == '__main__':
    import sys
    for line in sys.stdin:
        print(itn_text(line.rstrip('\n')))
