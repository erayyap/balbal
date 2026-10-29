"""Per-word timestamps from the CTC beam-search path (same method as the balbal Android app's captions).

The decoder's best path has one entry per acoustic frame (40 ms: 10 ms hop x 4 subsampling), framed by start/end nodes,
so entry i is frame i - 1. Each emitted token gets its first and last frame; a piece starting with the word marker opens
a new word. CTC spikes trail the spoken word by ~40-80 ms and stop before it ends, so starts move LEAD frames earlier
and ends reach the next word (gap under JOIN frames) or extend by TAIL frames. Checked against MMS forced alignment on
100 DEV clips: 94.5% of word starts within 100 ms.
"""
FRAME = 0.04
LEAD, TAIL, JOIN, MAX_GAP = 1, 5, 12, 8
PUNCT = set('.,?!:;"\'“”‘’()-…')


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


def words_from_spans(spans, sp, offset_seconds=0.):
    """[(id, f0, f1)] -> [{'word', 'start', 'end'}] in seconds; `sp` is the SentencePiece tokenizer."""
    groups = []
    for t, f0, f1 in spans:
        if sp.id_to_piece(t).startswith('▁') or not groups:
            groups.append([])
        groups[-1].append((t, f0, f1))
    raw = []
    for g in groups:
        text = sp.decode([t for t, _, _ in g]).strip()
        content = [(f0, f1) for t, f0, f1 in g if _content(sp.id_to_piece(t))]
        if not content:   # a punctuation-only piece joins the previous word
            if raw:
                raw[-1][0] += text
            continue
        # The model sometimes emits a word-initial piece far ahead of the rest of the word (often at frame 0).
        i = 0
        while i + 1 < len(content) and content[i + 1][0] - content[i][1] > MAX_GAP:
            i += 1
        raw.append([text, content[i][0], content[-1][1] + 1])
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
    return [dict(word=text, start=round(offset_seconds + f0 * FRAME, 2), end=round(offset_seconds + f1 * FRAME, 2))
            for text, f0, f1 in raw]
