"""Training-label cleaning for real (human-transcribed) speech rows, applied before digit verbalisation (numbers.py).

clean(rows) drops rows whose transcript has
  tag      a non-speech stage direction ('gülme Bir ...', 'alkış', 'müzik' before a capitalised word, a dash or the end),
  stutter  a stutter or truncation ('ya-yapıştı', 'payla-'),
  symbol   symbols, URLs or parentheses,
  rate     more than 25 characters per second of audio (label does not match the audio),
  repeat   an exact text (case/edge punctuation folded) already kept 5 times,
and turns curly quotes into straight ones. Rows: dicts with 'text', 'duration_seconds' and 'source'.
Returns (kept_rows, Counter of '<source>:<reason>' drops).
"""
import collections
import re

TAG = re.compile(r'(^|\s)(gülme|güler|gülüyor|gülüş|gülüşmeler|kahkaha|alkış|alkışlar|müzik|şarkı|sessizlik)( atar)?(?=\s+[A-ZÇĞİÖŞÜ]|\s*[-–]|\s*$)')
STUTTER = re.compile(r'\w-(?=\s|$|[.,;!?])|\b(\w{1,4})-\1')
SYMBOL = re.compile(r'[²³♪=%@#&*/\\<>\[\]{}()]|\.com|www|http|:\)')


def clean(rows):
    out, seen, dropped = [], collections.Counter(), collections.Counter()
    for r in rows:
        t = r['text'].replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
        why = ('tag' if TAG.search(t) else 'stutter' if STUTTER.search(t) else 'symbol' if SYMBOL.search(t)
               else 'rate' if len(t)/max(float(r['duration_seconds']), .1) > 25 else 'repeat' if seen[t.lower().strip(' .!?')] >= 5 else '')
        if why: dropped[r.get('source', '?') + ':' + why] += 1; continue
        seen[t.lower().strip(' .!?')] += 1; out.append(dict(r, text=t) if t != r['text'] else r)
    return out, dropped


def words_only(rows):
    """Digits -> words with numbers.verbalize; rows with an unsafe digit form are dropped (never guessed)."""
    from .numbers import verbalize
    out, dropped = [], collections.Counter()
    for r in rows:
        t = verbalize(r['text'])
        if t is None: dropped[r.get('source', '?') + ':digits'] += 1
        else: out.append(dict(r, text=t) if t != r['text'] else r)
    return out, dropped
