"""Digit spelling for steering targets (balbal-v1 outputs numbers as words). spell2 keeps case/punctuation and picks, per
token with a digit, the first spelling under which the balbal-normalizer-normalised text equals the balbal-normalizer normalisation of the original; falls
back to spell()."""
import re
from normalizer.metrics_v2 import normalize_pair_v2

ONES = ['', 'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz']
TENS = ['', 'on', 'yirmi', 'otuz', 'kırk', 'elli', 'altmış', 'yetmiş', 'seksen', 'doksan']
def below_thousand(n):
    h, r = divmod(n, 100); t, o = divmod(r, 10)
    return ([] if h == 0 else (['yüz'] if h == 1 else [ONES[h], 'yüz'])) + ([TENS[t]] if t else []) + ([ONES[o]] if o else [])
def number(n):
    if n == 0: return ['sıfır']
    out = []
    for value, name in ((10**9, 'milyar'), (10**6, 'milyon'), (1000, 'bin')):
        q, n = divmod(n, value)
        if q: out += ([name] if (q == 1 and name == 'bin') else below_thousand(q) + [name])
    return out + below_thousand(n)
def spell(text):
    text = re.sub(r'%\s*(\d)', r'yüzde \1', text)
    text = re.sub(r'(\d)[.](\d{3})\b', r'\1\2', text)   # 1.500 -> 1500
    text = re.sub(r'(\d+),(\d+)', lambda m: ' '.join(number(int(m[1])) + ['virgül'] + number(int(m[2]))), text)
    return re.sub(r'\d+', lambda m: ' '.join(number(int(m[0]))), text)


ORD = {'bir': 'birinci', 'iki': 'ikinci', 'üç': 'üçüncü', 'dört': 'dördüncü', 'beş': 'beşinci', 'altı': 'altıncı', 'yedi': 'yedinci',
       'sekiz': 'sekizinci', 'dokuz': 'dokuzuncu', 'on': 'onuncu', 'yirmi': 'yirminci', 'otuz': 'otuzuncu', 'kırk': 'kırkıncı',
       'elli': 'ellinci', 'altmış': 'altmışıncı', 'yetmiş': 'yetmişinci', 'seksen': 'sekseninci', 'doksan': 'doksanıncı',
       'yüz': 'yüzüncü', 'bin': 'bininci', 'milyon': 'milyonuncu', 'milyar': 'milyarıncı', 'sıfır': 'sıfırıncı'}
LETTER = dict(zip('ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWX', 'a be ce çe de e fe ge yumuşak_ge he ı i je ke le me ne o ö pe re se şe te u ü ve ye ze kü dabılyu iks'.split()))
UNIT = {'mm': 'milimetre', 'cm': 'santimetre', 'km': 'kilometre', 'm': 'metre', 'kg': 'kilogram', 'g': 'gram', 'mg': 'miligram',
        'ml': 'mililitre', 'lt': 'litre', 'km/s': 'kilometre bölü saat', 'mph': 'mil bölü saat', 'TL': 'TL', 'dk': 'dakika', 'sn': 'saniye'}
W = lambda n: ' '.join(number(n))
def ordinal(n): w = number(n); return ' '.join(w[:-1] + [ORD[w[-1]]])
def norm(t): return normalize_pair_v2(t, t)[0]

def cands(tok, nxt):
    m = re.fullmatch(r"([^\w%]*)([A-Za-zÇĞİÖŞÜçğıöşü]*)(%?)([\d.,:]*\d)([.,:]?)((?:['’][\wçğıöşü]+)?)([^\w']*)", tok)
    if not m: return []
    pre, let, pct, num, dot, suf, post = m.groups(); out = []
    lead = ' '.join(LETTER.get(c.upper(), c) for c in let) + ' ' if let else ''
    if pct: lead += 'yüzde '
    if re.fullmatch(r'\d+', num):
        n = int(num); base = [W(n)] + ([ordinal(n)] if dot == '.' else []) + [' '.join(W(int(d)) for d in num)]
        for b in base:
            d = '' if (dot == '.' and b == ordinal(n)) else dot
            out += [pre + lead + b + d + suf + post, pre + lead + b + d + suf.lstrip("'’") + post]
            if let: out.append(pre + let + ' ' + b + d + suf + post)
    elif re.fullmatch(r'\d{1,3}([.]\d{3})+', num): out.append(pre + lead + W(int(num.replace('.', ''))) + dot + suf + post)
    a = re.fullmatch(r'(\d+)[.,:](\d+)', num)
    if a:
        x, y = a.groups()
        for sep in (' virgül ', ' nokta ', ' '):
            yy = W(int(y)) if not (y.startswith('0') and sep != ' ') else ' '.join(W(int(d)) for d in y)
            out.append(pre + lead + W(int(x)) + sep + yy + dot + suf + post)
            if y == '00' or y.startswith('0'): out.append(pre + lead + W(int(x)) + ' ' + ' '.join(W(int(d)) for d in y) + dot + suf + post)
    return out

def spell2(text):
    toks = text.split(); goal = norm(text)
    for i, t in enumerate(toks):
        if not re.search(r'\d', t): continue
        nxt = toks[i + 1] if i + 1 < len(toks) else ''
        best = None
        for c in cands(t, nxt):
            trial = toks[:i] + [c] + toks[i + 1:]
            if norm(' '.join(trial)) == goal: best = c; break
            um = re.fullmatch(r"([A-Za-z/]+)((?:['’][\wçğıöşü]+)?)([^\w']*)", nxt)
            if um and um[1] in UNIT:   # number + unit: expand the unit too ("35 mm'dir" -> "otuz beş milimetredir")
                for u in (UNIT[um[1]] + um[2] + um[3], UNIT[um[1]] + um[2].lstrip("'’") + um[3]):
                    trial2 = toks[:i] + [c, u] + toks[i + 2:]
                    if norm(' '.join(trial2)) == goal: best = (c, u); break
                if best: break
        if best is None: toks[i] = spell(t)
        elif isinstance(best, tuple): toks[i], toks[i + 1] = best
        else: toks[i] = best
    return ' '.join(toks)
