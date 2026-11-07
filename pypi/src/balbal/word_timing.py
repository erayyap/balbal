"""Per-word timestamps from the CTC beam-search path (same method as the balbal Android app's captions).

The decoder's best path has one entry per acoustic frame (40 ms: 10 ms hop x 4 subsampling), framed by start/end nodes,
so entry i is frame i - 1. Each emitted token gets its first and last frame; a piece starting with the word marker opens
a new word. CTC spikes trail the spoken word by ~40-80 ms and stop before it ends, so starts move LEAD frames earlier
and ends reach the next word (gap under JOIN frames) or extend by TAIL frames. Checked against MMS forced alignment on
100 DEV clips: 94.5% of word starts within 100 ms.
Confidence (when the decoder passes token confidences): the lowest of the word's content-token confidences, where a token's confidence is
its highest calibrated CTC posterior over its frames (punctuation pieces do not count). On 38 hand-checked phone recordings it ranks correct above
wrong words with AUC 0.88 (balbal-v2) / 0.87 (manas); wrong words average ~0.4, correct ones ~0.85.
"""
FRAME = 0.04
_LEAD, _TAIL, _JOIN, _MAX_GAP = LEAD, TAIL, JOIN, MAX_GAP = 1, 5, 12, 8
PUNCT = set('.,?!:;"\'“”‘’()-…')
# Other scripts' punctuation (none of these is a piece of the balbal-v2 / manas vocabularies, whose output is unchanged; '%', '/',
# '&', '#', '_' and dashes other than '-' stay word content).
PUNCT |= set('«»„‚‹›¿¡،؟؛。、！？：；')


def token_spans(path, vocab_size):
    """Frame-level path (blanks/repeats included, start/end nodes -1) -> [(token_id, first_frame, last_frame)]."""
    spans, prev = [], None
    for i, t in enumerate(path):
        t = int(t)
        if 0 <= t < vocab_size:
            if t != prev:
                spans.append([t, i - 1, i - 1])
            else:
                spans[-1][2] = i - 1
        prev = t
    return [tuple(s) for s in spans]


def _content(piece):
    p = piece.replace('▁', '')
    return bool(p) and not set(p) <= PUNCT


def words_from_spans(spans, sp, offset_seconds=0., confidences=None, frame_seconds=FRAME):
    """[(id, f0, f1)] -> [{'word', 'start', 'end'}] in seconds; `sp` is the SentencePiece tokenizer. With `confidences` (one per
    span) every word also gets 'confidence'. frame_seconds: length of an acoustic frame (0.04 s for balbal-v2 / manas; the frame-count
    constants above are tuned for that and are scaled to the same durations for slower frame rates, e.g. 0.08 s)."""
    scale = FRAME / frame_seconds
    LEAD, TAIL, JOIN, MAX_GAP = (max(1, round(x * scale)) for x in (_LEAD, _TAIL, _JOIN, _MAX_GAP))
    groups = []
    for k, (t, f0, f1) in enumerate(spans):
        if sp.id_to_piece(t).startswith('▁') or not groups:
            groups.append([])
        groups[-1].append((t, f0, f1, k))
    raw = []
    for g in groups:
        text = sp.decode([t for t, _, _, _ in g]).strip()
        content = [(f0, f1, k) for t, f0, f1, k in g if _content(sp.id_to_piece(t))]
        if not content:   # a punctuation-only piece joins the previous word
            if raw:
                raw[-1][0] += text
            continue
        # The model sometimes emits a word-initial piece far ahead of the rest of the word (often at frame 0).
        i = 0
        while i + 1 < len(content) and content[i + 1][0] - content[i][1] > MAX_GAP:
            i += 1
        confidence = None if confidences is None else min(float(confidences[k]) for _, _, k in content)
        raw.append([text, content[i][0], content[-1][1] + 1, confidence])
    if not raw:
        return []
    if len(raw) > 1 and raw[0][1] <= 1 and raw[1][1] - raw[0][2] > JOIN:
        raw[0][1] = max(0, raw[1][1] - max(6, round(len(raw[0][0]) * 1.75)))
        raw[0][2] = raw[1][1]   # the moved word runs up to the next one
    for w in raw:
        w[1] = max(0, w[1] - LEAD)
    for k, w in enumerate(raw):
        if k + 1 < len(raw):
            nxt = raw[k + 1][1]
            w[2] = min(nxt if nxt - w[2] < JOIN else w[2] + TAIL, nxt)
        else:
            w[2] += TAIL
    words = [dict(word=text, start=round(offset_seconds + f0 * frame_seconds, 2), end=round(offset_seconds + f1 * frame_seconds, 2))
             for text, f0, f1, _ in raw]
    if confidences is not None:
        for w, r in zip(words, raw): w['confidence'] = round(r[3], 3)
    return words
