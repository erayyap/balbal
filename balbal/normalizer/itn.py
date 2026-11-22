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
  whole text became digits until 0.6.10; 'yüzde 20' -> '%20'; a half hour after a time word -> 'saat 8.30', "sabah 8.30'da".
- itn_digits(text): scoring form for already-normalized (lowercase, no punctuation) text. Numbers are spelled digit by
  digit ('doksan yüz yirmi üç' == 'dokuz sıfır bir iki üç' == '9 0 1 2 3'), removing the reading/grouping ambiguity of
  phone numbers and codes; digits already present are kept.

  0.6.5: a date read with digits splits from the year ('sıfır dokuz bin dokuz yüz altmış üç' -> '0 9 1963', 'yirmi üç dokuz bin dokuz yüz
  seksen' -> '23 9 1980'), and acronyms are joined: 2-3 spaced capitals ('T L' -> 'TL', 'E F T' -> 'EFT'; not next to a digit, not in a
  letter-by-letter spelling, never 'A Ş') and runs of letter names ('es em es' -> 'SMS', 'de ne es' -> 'DNS', 'be de de ka' -> 'BDDK') when
  they hold at least two unambiguous letter names (or three names with one).

  0.6.6: decades and centuries with a plural suffix keep the whole number ('bin sekiz yüzlerden' -> "1800'lerden", was '1008 yüzlerden';
  'bin dokuz yüz seksenlerde' -> "1980'lerde", 'iki binlerin' -> "2000'lerin").

  0.6.7: spoken e-mail addresses become written addresses ('burak nokta çelik at gmail nokta com' -> 'burak.celik@gmail.com'; see the
  e-mail section at the end).

  0.6.8: a proper noun with a case suffix gets its apostrophe (TDK): 'ahmetle' -> "Ahmet'le", 'nihatın' -> "Nihat'ın", 'ankarada' ->
  "Ankara'da". Names come from itn_names (Turkish given names and surnames, places, car models, companies) and are kept only where neither the
  name nor its inflected forms are ordinary words (no 'Pek'i', 'Gün'ler'); no dative after a person name (Sabriye, Habibe are names), no
  plural, a name that is itself listed is never split. In cased text the name takes its written form; in all-lowercase text it stays lowercase
  ("nihat'la").

  0.6.9: Volkswagen model and engine names as the maker writes them: 'ay di 7' -> 'ID.7', 'TSİ' -> 'TSI', 'E T S İ' -> 'eTSI',
  'bir buçuk TSI' -> '1.5 TSI'.

  0.6.10: a lone on / yüz / bin that is the whole text stays a word again ('On.' -> 'On.', as in 0.6.3; 0.6.4 wrote '10.').
  Before a unit it is still digits.

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
    """Tokens -> [(output token, first input index, end input index)]: spoken e-mail addresses (email_spans, 0.6.7) first, then on the text
    between them number runs, the writing conventions (_conventions) and letter runs (_letters)."""
    out, a = [], 0
    for s, e, tok in email_spans(toks) + [(len(toks), len(toks), None)]:
        out += [(t, i + a, j + a) for t, i, j in _letters(_conventions(_number_runs(toks[a:s])))]
        if tok is not None: out.append((tok, s, e))
        a = e
    cased = any(c.isupper() for t in toks for c in t)
    out = [(_proper(t, cased) if j - i == 1 and t == toks[i] else t, i, j) for t, i, j in out]   # 0.6.8: untouched tokens only
    return _car_models(out)                                                                         # 0.6.9


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
        if lc in LONE and not pre and not prev.isdigit() and not (len(prev) == 1 and prev.isalpha()) and nxt in UNITW:   # on gün -> 10 gün; 'x bin lira' stays; 0.6.10: a lone 'On.' stays a word
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


# Suffix on the last word of a number run: up to 4 letters ('kırk dokuzdu' -> 49'du), or a plural -ler/-lar with up to 5 more
# (0.6.6: 'bin sekiz yüzlerden' -> 1800'lerden, 'bin dokuz yüz seksenlerde' -> 1980'lerde); -lerce/-larca stays a word ('on binlerce').
SUFFIX = re.compile(r'[a-zçğıöşü]{1,4}|l[ae]r(?!c[ae])[a-zçğıöşü]{0,5}')


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
                w = max((n for n in NUM if low(c).startswith(n) and SUFFIX.fullmatch(low(c)[len(n):])), key=len, default=None)
                if w: words.append(w); j += 1; last_post = "'" + c[len(w):] + p1
                break
            if low(c) not in NUM or (p0 and j > i): break
            words.append(low(c)); j += 1; last_post = p1
            if p1: break
        if len(words) == 1 and words[0] in AMBIG: out.append((toks[i], i, i + 1)); i += 1; continue
        out.append((pre + ' '.join(parse(words)) + last_post, i, j)); i = j
    return out


# ---- e-mail addresses (0.6.7) ----
# Spoken e-mail addresses -> written addresses (2026-11-21, balbal 0.6.7).
#
# Model output spells addresses the way people say them: 'burak nokta çelik at live nokta com', 'mega alt tire ufuk bin dokuz yüz doksan beş
# et yandex nokta com', city-name spelling 'Edirne Rize Denizli ... at superonline nokta com', letter runs 'T R'. This pass rewrites such a span
# as one token 'burak.celik@live.com'. It fires only on 'at' / 'et' / '@' followed by a domain of >= 2 parts joined by 'nokta' whose last part is a
# known top-level domain, so ordinary sentences ('at', 'et' as Turkish words) are left alone.
#
# Local part (before at): tokens back from 'at' while they are connectors (nokta, tire, alt tire, alt çizgi), letters / letter names, city-name
# spelling, number words, or plain words (at most 3, stopping at punctuation or a stop word such as 'adresim'). Pieces: connectors -> . - _,
# a run of >= 2 city names -> their initials (Turkish spelling alphabet), number-word runs -> digits, single letters as is; everything lower-cased
# and folded to ASCII (ç->c, ğ->g, ı->i, ö->o, ş->s, ü->u), as addresses are written.
CITY = {'adana': 'a', 'bursa': 'b', 'ceyhan': 'c', 'çanakkale': 'c', 'denizli': 'd', 'edirne': 'e', 'fatsa': 'f', 'giresun': 'g', 'hatay': 'h',
        'ısparta': 'i', 'isparta': 'i', 'izmir': 'i', 'jandarma': 'j', 'kars': 'k', 'lüleburgaz': 'l', 'muş': 'm', 'niğde': 'n', 'ordu': 'o',
        'ödemiş': 'o', 'polatlı': 'p', 'rize': 'r', 'sinop': 's', 'şırnak': 's', 'tokat': 't', 'uşak': 'u', 'ünye': 'u', 'van': 'v',
        'yozgat': 'y', 'zonguldak': 'z', 'ankara': 'a', 'antalya': 'a', 'istanbul': 'i', 'muğla': 'm', 'malatya': 'm', 'samsun': 's', 'trabzon': 't',
        'kayseri': 'k', 'konya': 'k'}
_FOLD = str.maketrans({'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u', 'â': 'a', 'î': 'i', 'û': 'u'})
CITY.update({k.translate(_FOLD): v for k, v in list(CITY.items())})
LETTER = {'be': 'b', 'ce': 'c', 'de': 'd', 'fe': 'f', 'ge': 'g', 'he': 'h', 'je': 'j', 'ke': 'k', 'ka': 'k', 'le': 'l', 'me': 'm', 'ne': 'n', 'pe': 'p',
          're': 'r', 'se': 's', 'te': 't', 've': 'v', 'ye': 'y', 'ze': 'z', 'ku': 'q', 'kü': 'q', 'iks': 'x', 'dabılyu': 'w'}
TLD = {'com', 'net', 'org', 'edu', 'gov', 'tr', 'co', 'me', 'io', 'info', 'biz', 'av', 'bel', 'k12', 'mil', 'pol', 'gen', 'web', 'tv', 'app',
       'dev', 'ai', 'uk', 'de', 'eu', 'us'}
AT = {'at', 'et', '@'}
EMAIL_GLUE = 2          # plain words glued into a local part without connectors ('Sevda Ekiz at' -> sevdaekiz)
STOP = {'adresim', 'adres', 'adresi', 'adresimiz', 'mail', 'mailim', 'maili', 'mailimiz', 'eposta', 'e-posta', 'epostam', 'postam', 'benim',
        'şu', 'su', 'evet', 'tamam', 'olarak', 'şöyle', 'soyle', 'yazıyorum', 'yazın', 'yazar', 'misiniz', 'mısınız', 'hanım', 'bey', 've',
        'ile', 'de', 'da', 'bir', 've', 'hocam', 'efendim', 'iste', 'işte', 'yani', 'aynen', 'mailine', 'mailinize', 'adresine', 'adresinize',
        'gönderin', 'gönderir', 'atın', 'atar', 'şeyi', 'şey', 'size', 'bana', 'onu', 'bu', 'o', 'var', 'yok', 'olan', 'için', 'gibi', 'ama',
        'fakat', 'ben', 'sen', 'biz', 'siz', 'adım', 'ismim', 'adı', 'ismi', 'ki', 'mi', 'mı', 'mu', 'mü', 'ya', 'hayır', 'peki', 'yazayım',
        'söyleyeyim', 'kaydedin', 'not', 'alın', 'yazdınız', 'gönderdim', 'gönderiyorum', 'attım', 'şimdi', 'tekrar', 'yeni', 'eski', 'hani', 'değil', 'degil', 'evet'}
NUMW = {'sıfır', 'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz', 'on', 'yirmi', 'otuz', 'kırk', 'elli', 'altmış', 'yetmiş',
        'seksen', 'doksan', 'yüz', 'bin'}
_EP = re.compile(r'^(.*?)([.,!?;:]*)$')


def _bare(tok):
    return _EP.match(tok).group(1)


def _num(words):
    if len(words) == 1 and low(words[0]) in LONE: return str(LONE[low(words[0])])   # 0.6.10: a lone on / yüz / bin is a number inside an address
    out = itn_text(' '.join(words))
    return out.replace(' ', '') if re.fullmatch(r'[\d ]+', out) else None


def _piece(words):
    """Tokens of one segment (no connectors) -> written text, or None when they do not form an address piece."""
    lw = [low(_bare(w)) for w in words]
    out, i = [], 0
    while i < len(lw):
        w = lw[i]
        if w in NUMW:
            j = i
            while j < len(lw) and lw[j] in NUMW: j += 1
            n = _num(lw[i:j])
            if n is None: return None
            out.append(n); i = j; continue
        if w in CITY and ((i + 1 < len(lw) and lw[i + 1] in CITY) or (i > 0 and lw[i - 1] in CITY)):
            out.append(CITY[w]); i += 1; continue
        if w in ('çift', 'cift') and i + 1 < len(lw) and lw[i + 1] == 've':
            out.append('w'); i += 2; continue
        if len(w) == 1 or w.isdigit(): out.append(w); i += 1; continue
        if w in LETTER and len(lw) > 1: out.append(LETTER[w]); i += 1; continue
        j = i + 1
        while j < len(lw) and len(lw[j]) == 1: j += 1
        if j - i > 2 and ''.join(lw[i + 1:j]).translate(_FOLD) == w.translate(_FOLD): out.append(w); i = j; continue   # spelled confirmation
        out.append(w); i += 1
    return ''.join(out).translate(_FOLD)


def _domain(toks, k):
    """toks[k:] after the at sign -> (written domain, end index, trailing punctuation) or None."""
    segs, cur, j = [], [], k
    while j < len(toks):
        b = low(_bare(toks[j]))
        if b == 'nokta':
            if not cur: return None
            segs.append(cur); cur = []; j += 1; continue
        if cur and len(cur) >= 4: break
        cur.append(toks[j]); j += 1
        if _bare(toks[j - 1]) != toks[j - 1]: break           # punctuation ends the address
        nxt = low(_bare(toks[j])) if j < len(toks) else None
        if segs and nxt != 'nokta':
            ahead = [low(_bare(x)) for x in toks[j:j + 4 - len(cur) + 1]]
            if 'nokta' in ahead and all(_bare(x) == x for x in toks[j - 1:j + ahead.index('nokta')]): continue   # multi-token middle part
            # last part: one TLD token, or two single letters ('T R')
            if len(cur) == 1 and len(_bare(cur[0])) == 1 and nxt and len(nxt) == 1: cur.append(toks[j]); j += 1
            elif len(cur) == 1 and low(_bare(cur[0])) in LETTER and nxt in LETTER: cur.append(toks[j]); j += 1   # 'te re' = tr
            break
    if cur: segs.append(cur)
    if len(segs) < 2: return None
    FIX = {'ko': 'co', 'kom': 'com'}
    last = ''.join(LETTER.get(low(_bare(x)), low(_bare(x))) if len(segs[-1]) > 1 else low(_bare(x)) for x in segs[-1]).translate(_FOLD)
    if FIX.get(last, last) not in TLD: return None
    parts = [_piece(s) for s in segs]
    if any(p is None or p == '' for p in parts): return None
    parts = [FIX.get(x, x) for x in parts]
    return '.'.join(parts), j, _EP.match(toks[j - 1]).group(2)


def _local(toks, a):
    """Local part ending before index a -> (written local part, start index) or None."""
    i, plain, pieces = a, 0, []
    while i > 0:
        t = toks[i - 1]; b = low(_bare(t))
        if _bare(t) != t and not (_EP.match(t).group(2) == ',' and pieces and (b in CITY or b == 'nokta')
                                  and (pieces[-1][0] == 'sep' or low(_bare(pieces[-1][1])) in CITY)): break   # punctuation before the address
        if b in ('nokta', 'tire', 'çizgi', 'artı', 'arti'):
            sep = {'nokta': '.', 'tire': '-', 'çizgi': '-', 'artı': '+', 'arti': '+'}[b]
            if i >= 2 and low(toks[i - 2]) == 'alt': sep = '_'; i -= 1
            pieces.append(('sep', sep)); i -= 1; continue
        if b == 've' and i >= 2 and low(toks[i - 2]) in ('çift', 'cift'):          # 'çift ve' = w
            pieces.append(('w', t)); pieces.append(('w', toks[i - 2])); i -= 2; continue
        if (b in STOP and not (b in NUMW or len(b) == 1)) or re.fullmatch(r'[eaıi]*[eaı]{2,}', b): break    # stop words, fillers (eee, ııı)
        is_plain = not (b in NUMW or b in CITY or len(b) == 1 or b.isdigit() or b in LETTER)
        if is_plain:
            # with a connector in the address, a plain word must sit right before one ('Belli kamile nokta akkuş' -> kamile.akkus);
            # without connectors up to 3 glued words ('Ahmet Erdem P M K at' -> ahmeterdempmk)
            if any(k == 'sep' for k, _ in pieces):
                if pieces[-1][0] != 'sep': break
            elif plain >= EMAIL_GLUE: break
            else: plain += 1
        pieces.append(('w', t)); i -= 1
    if not pieces or pieces[-1][0] == 'sep': return None
    pieces.reverse()
    out, run = [], []
    for kind, v in pieces + [('sep', '')]:
        if kind == 'w': run.append(v); continue
        if run:
            p = _piece(run)
            if p is None: return None
            out.append(p); run = []
        out.append(v)
    return ''.join(out), i


def email_spans(toks):
    """Tokens -> [(start, end, written address)] for spoken e-mail addresses, left to right, non-overlapping."""
    out, k, floor = [], 0, 0
    while k < len(toks):
        if low(_bare(toks[k])) in AT and k > floor and _bare(toks[k]) == toks[k]:
            d = _domain(toks, k + 1)
            l = _local(toks[floor:], k - floor) if d else None
            if d and l:
                (dom, end, punct), (loc, start) = d, l
                out.append((start + floor, end, f'{loc}@{dom}{punct}'))
                k = floor = end
                continue
        k += 1
    return out


def email_itn(text):
    """Only the e-mail rewrite: spoken addresses in a model output -> written addresses; other text unchanged."""
    toks = text.split()
    for s, e, tok in reversed(email_spans(toks)): toks[s:e] = [tok]
    return ' '.join(toks)


# ---- car model and engine names (0.6.9) ---------------------------------------------------------------------------------------------------
_ENGINE = re.compile(r"^(e?TS|GT|TD|eTD)[İI]((?:['’][a-zçğıöşü]+)?[.,!?;:]*)$")
_ID_NUM = re.compile(r"^(\d)((?:['’][a-zçğıöşü]+)?[.,!?;:]*)$")
_ENGINE_W = re.compile(r"^(e?TS|GT|e?TD)I(['’]|[.,!?;:]|$)")
_HALF = {'bir': '1.5', 'iki': '2.5', 'Bir': '1.5', 'İki': '2.5', '1': '1.5', '2': '2.5'}


def _car_models(out):
    """Volkswagen model and engine names as the maker writes them: 'ay di 4' -> 'ID.4' (and "ay di üçe" -> "ID.3'e"), 'TSİ' / 'GTİ' / 'TDİ'
    -> 'TSI' / 'GTI' / 'TDI', spelled 'E T S İ' -> 'eTSI', 'bir buçuk' / 'iki buçuk' before an engine name -> '1.5' / '2.5'."""
    res, k = [], 0
    while k < len(out):
        t, i, j = out[k]; nxt = [x[0] for x in out[k + 1:k + 4]]
        if t.lower() == 'ay' and len(nxt) >= 2 and nxt[0].lower() == 'di' and _ID_NUM.match(nxt[1]):
            res.append(('ID.' + nxt[1], i, out[k + 2][2])); k += 3; continue
        if t == 'E' and len(nxt) == 3 and nxt[:2] == ['T', 'S'] and re.match(r"^[İI][.,!?;:]*$", nxt[2]):
            res.append(('eTSI' + nxt[2][1:], i, out[k + 3][2])); k += 4; continue
        m = _ENGINE.match(t)
        if m: t = m.group(1).replace('ETD', 'eTD') + 'I' + m.group(2)
        res.append((t, i, j)); k += 1
    for k in range(len(res) - 2):                                     # 'bir buçuk' + engine -> '1.5'
        (a, i, _), (b, _, j), (c, _, _) = res[k:k + 3]
        if a in _HALF and b == 'buçuk' and _ENGINE_W.match(c):
            res[k:k + 2] = [(_HALF[a], i, j), ('', j, j)]
    return [x for x in res if x[0] != '']


# ---- proper noun + suffix -> apostrophe (0.6.8) ---------------------------------------------------------------------------------------------
from .itn_names import PERSON as _PERSON_NAMES, OTHER as _OTHER_NAMES, NEVER as _NEVER_SPLIT


def _tr_lower(s):
    return s.replace('İ', 'i').replace('I', 'ı').lower()


_PROPER = {_tr_lower(n): n for n in _PERSON_NAMES}
_PROPER.update({_tr_lower(n): n for n in _OTHER_NAMES})                    # a place / brand spelling wins over a same-spelled person name
_PERSON = frozenset(_tr_lower(n) for n in _PERSON_NAMES) - {_tr_lower(n) for n in _OTHER_NAMES}
_NEVER = frozenset(_NEVER_SPLIT)
_PV = set('aeıioöuü')
_PP = re.compile(r"^(.*?)([.,!?;:]*)$")


def _proper_suffixes(stem):
    """Case suffixes a name takes, by vowel harmony and buffer letters: a/e/ya/ye, ı/i/u/ü/yı.., da/ta.., dan/tan.., ın/nın.., -daki, la/yla."""
    v = next((c for c in reversed(stem) if c in _PV), 'e'); back = v in 'aıou'; rnd = v in 'ouöü'
    A = 'a' if back else 'e'; I = ('u' if rnd else 'ı') if back else ('ü' if rnd else 'i'); D = 't' if stem[-1] in 'çfhkpsşt' else 'd'
    y, n = ('y', 'n') if stem[-1] in _PV else ('', '')
    return {y + A, y + I, n + I + 'n', D + A, D + A + 'n', D + A + 'ki', y + 'l' + A}


def _proper(tok, cased):
    """'nihatla' -> "Nihat'la" (cased text) / "nihat'la" (lowercase text); any other token unchanged."""
    core, punct = _PP.match(tok).groups(); low = _tr_lower(core)
    if not core or not core[0].isalpha() or "'" in core or '’' in core or low in _PROPER or low in _NEVER: return tok
    for k in range(len(low) - 1, 2, -1):
        stem, suf = low[:k], low[k:]
        if stem in _PROPER and suf in _proper_suffixes(stem) and not (stem in _PERSON and suf in ('a', 'e', 'ya', 'ye')):
            return (_PROPER[stem] if cased else core[:k]) + "'" + core[k:] + punct
    return tok


if __name__ == '__main__':
    import sys
    for line in sys.stdin:
        print(itn_text(line.rstrip('\n')))
