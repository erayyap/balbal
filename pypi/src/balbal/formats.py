"""Output formats for a transcription result: plain text, JSON, SRT and WebVTT subtitles (used by `balbal run --format` and the
server's response_format).

Subtitle segments come from the timed words (see segments_from_words): at most SEGMENT_SECONDS and SEGMENT_CHARS, shown on up to two
lines of about LINE_CHARS.
"""
import json
import math

FORMATS = ('txt', 'json', 'srt', 'vtt')
SEGMENT_PAUSE, SEGMENT_SECONDS, SEGMENT_CHARS, LINE_CHARS = 0.6, 7., 84, 42
SENTENCE_END = ('.', '?', '!', '…')


def group_words(words):
    """Words ({'word', 'start', 'end'}) -> lists of words, one per subtitle-sized segment: a new segment after a pause of SEGMENT_PAUSE
    seconds, after a sentence end (. ? !), or before the segment would pass SEGMENT_SECONDS or SEGMENT_CHARS."""
    groups, current = [], []
    for w in words:
        if current:
            chars = sum(len(x['word']) + 1 for x in current) + len(w['word'])
            if (w['start'] - current[-1]['end'] >= SEGMENT_PAUSE or w['end'] - current[0]['start'] > SEGMENT_SECONDS
                    or chars > SEGMENT_CHARS or current[-1]['word'].endswith(SENTENCE_END)):
                groups.append(current); current = []
        current.append(w)
    if current: groups.append(current)
    return groups


def segments_from_words(words):
    """Subtitle segments {'id', 'start', 'end', 'text'(, 'avg_logprob')} of group_words(words). avg_logprob = mean log word
    confidence, like Whisper's (when the words carry confidences)."""
    out = []
    for i, g in enumerate(group_words(words)):
        seg = dict(id=i, start=round(g[0]['start'], 3), end=round(g[-1]['end'], 3), text=' '.join(w['word'] for w in g))
        if all('confidence' in w for w in g): seg['avg_logprob'] = round(sum(math.log(max(w['confidence'], 1e-6)) for w in g) / len(g), 4)
        out.append(seg)
    return out


def two_lines(text, width=LINE_CHARS):
    """A subtitle text over at most two lines: split at the space nearest the middle once it is longer than `width`."""
    if len(text) <= width or ' ' not in text: return text
    middle = len(text) / 2
    cut = min((i for i, c in enumerate(text) if c == ' '), key=lambda i: abs(i - middle))
    return text[:cut] + '\n' + text[cut + 1:]


def _clock(seconds, separator):
    ms = int(round(max(0., seconds) * 1000))
    return f'{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{separator}{ms % 1000:03d}'


def to_srt(segments):
    return ''.join(f'{i + 1}\n{_clock(s["start"], ",")} --> {_clock(s["end"], ",")}\n{two_lines(s["text"])}\n\n' for i, s in enumerate(segments))


def to_vtt(segments):
    return 'WEBVTT\n\n' + ''.join(f'{_clock(s["start"], ".")} --> {_clock(s["end"], ".")}\n{two_lines(s["text"])}\n\n' for s in segments)


def render(result, fmt):
    """A transcribe() result as `fmt` ('txt', 'json', 'srt' or 'vtt') text."""
    if fmt == 'txt': return result['text'] + '\n'
    if fmt == 'json': return json.dumps(result, ensure_ascii=False) + '\n'
    if fmt == 'srt': return to_srt(segments_from_words(result['words']))
    if fmt == 'vtt': return to_vtt(segments_from_words(result['words']))
    raise ValueError(f'format must be one of {FORMATS}')
