"""Training data: manifests, frame-level language labels, batching.

Manifest rows (JSON lines; your own data):
  {"id", "audio_path", "duration_seconds", "text", ...}                      one audio file (any sample rate; mono-mixed, resampled)
  {"id", "parts": [{"audio_path", "start", "end", "gap_before", "lang"}, ...], "duration_seconds", "text", ...}
      a virtual row: segments joined on the fly (start / end in 16 kHz samples, gap_before = seconds of zeros before the part)
optional fields:
  "pcm_sha256"   sha256 of the canonical PCM16 samples (balbal.audio.pcm_hash); checked when present (row or part)
  "lang"         "tr" | "en" | "ns" (non-speech); default: "ns" for an empty text, else --default-lang
  "lang_spans"   [[start_s, end_s, "en"], ...]: stretches in another language inside the row (code-switching)
  "source"       a group name for the per-source dev scores
An empty "text" is a non-speech clip (CTC target = no tokens).

Language labels (language head): every frame gets tr 0 / en 1 / non-speech 2 from the row / part / span language at the frame centre;
`band` frames on both sides of a language change and the gap_before zeros are not scored (-100).
Condition (models with lang_cond): per clip tr / en if only that language is labelled (with probability cond_p, else auto), auto for
code-switched or non-speech clips.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from balbal.audio import pcm_hash, read_audio

TR, EN, NS, IGNORE = 0, 1, 2, -1
CODE = {'tr': TR, 'en': EN, 'ns': NS, 'non-speech': NS, 'nonspeech': NS}
COND_TR, COND_EN, COND_AUTO = 0, 1, 2


def ctc_frames_needed(label):
    """CTC frames a label needs: one per token plus a blank between repeated tokens."""
    return len(label) + sum(a == b for a, b in zip(label, label[1:]))


def load_manifests(paths, drop=(), relabel=()):
    """Rows of comma-separated manifests. drop: files of ids to leave out (one id per line or JSON lines with "id"); relabel: JSON
    lines {"id", "text"} replacing texts. A row repeated on consecutive lines is kept that many times (oversampling); other repeated
    ids are skipped. Returns (rows, counts)."""
    dropped, new_text, counts = set(), {}, dict(rows=0, repeat=0, duplicate=0, dropped=0, relabelled=0)
    for f in drop:
        for line in open(f):
            line = line.strip()
            if line: dropped.add(json.loads(line)['id'] if line.startswith('{') else line)
    for f in relabel:
        for r in map(json.loads, open(f)): new_text[r['id']] = r['text']
    rows, seen = [], set()
    for f in paths:
        prev = None
        for r in map(json.loads, open(f)):
            if r['id'] == prev and rows and rows[-1]['id'] == r['id']:
                rows.append(rows[-1]); counts['repeat'] += 1; continue
            prev = r['id']
            if r['id'] in seen: counts['duplicate'] += 1; continue
            seen.add(r['id'])
            if r['id'] in dropped: counts['dropped'] += 1; continue
            if r['id'] in new_text: r = dict(r, text=new_text[r['id']]); counts['relabelled'] += 1
            rows.append(r)
    counts['rows'] = len(rows)
    return rows, counts


def _checked(path, digest, row_id):
    wave = read_audio(path)
    if digest and pcm_hash(wave) != digest: raise ValueError('Audio identity changed: ' + row_id)
    return np.asarray(wave, np.float32)


def materialise(row, default_lang='tr'):
    """(wave float32 16 kHz, per-sample language track int8) of a manifest row."""
    base = CODE[row['lang']] if row.get('lang') else (NS if not row.get('text', '').strip() else CODE[default_lang])
    if 'parts' in row:
        waves, tracks = [], []
        for p in row['parts']:
            w = _checked(p['audio_path'], p.get('pcm_sha256'), row['id'])
            if p.get('gap_before'):
                g = int(p['gap_before'] * 16000); waves.append(np.zeros(g, np.float32)); tracks.append(np.full(g, IGNORE, np.int8))
            w = w[int(p.get('start') or 0):p.get('end')]
            waves.append(w); tracks.append(np.full(len(w), CODE[p['lang']] if p.get('lang') else base, np.int8))
        return np.concatenate(waves).astype(np.float32), np.concatenate(tracks)
    wave = _checked(row['audio_path'], row.get('pcm_sha256'), row['id']); track = np.full(len(wave), base, np.int8)
    for s, e, lang in row.get('lang_spans') or ():
        track[max(0, int(s * 16000)):min(len(wave), int(e * 16000))] = CODE[lang]
    return wave, track


def band(track, width):
    """Mark `width` samples on both sides of every change between two languages as IGNORE."""
    if width <= 0 or not len(track): return track
    cuts = np.concatenate([[0], np.flatnonzero(track[1:] != track[:-1]) + 1, [len(track)]])
    runs = [(int(a), int(b), int(track[a])) for a, b in zip(cuts[:-1], cuts[1:]) if track[a] != IGNORE]
    for (a0, a1, la), (b0, b1, lb) in zip(runs, runs[1:]):
        if la != lb: track[max(a0, a1 - width):a1] = IGNORE; track[b0:min(b1, b0 + width)] = IGNORE
    return track


def frames(track, hop):
    """Frame labels: the track at each frame centre (frame k = samples k x hop ...); IGNORE -> -100."""
    k = np.arange(len(track) // hop + 1)
    lab = track[np.minimum(k * hop + hop // 2, len(track) - 1)].astype(np.int64)
    lab[lab == IGNORE] = -100
    return torch.from_numpy(lab.astype(np.int8))


class Rows(Dataset):
    """Items (index, wave, label ids, frame language labels, condition id). seed: (run seed, epoch) for the condition draws."""
    def __init__(self, rows, labels, hop, band_frames=5, cond_p=.7, seed=(0, 0), default_lang='tr'):
        self.rows, self.labels, self.hop, self.band, self.cond_p = rows, labels, hop, band_frames, cond_p
        self.seed, self.default_lang = seed, default_lang

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        wave, track = materialise(self.rows[i], self.default_lang)
        track = band(track, self.band * self.hop)
        has_tr, has_en = bool((track == TR).any()), bool((track == EN).any())
        draw = np.random.default_rng([*self.seed, i]).random()
        cond = COND_AUTO if has_tr == has_en else (COND_TR if has_tr else COND_EN) if draw < self.cond_p else COND_AUTO
        return i, torch.from_numpy(wave), torch.tensor(self.labels[i], dtype=torch.long), frames(track, self.hop), cond


def collate(items):
    ids, waves, labels, langs, conds = zip(*items)
    pad = torch.nn.utils.rnn.pad_sequence
    return (list(ids), pad(waves, batch_first=True), torch.tensor([len(w) for w in waves]), pad(labels, batch_first=True),
            torch.tensor([len(t) for t in labels]), pad([x.long() for x in langs], batch_first=True, padding_value=-100),
            torch.tensor(conds, dtype=torch.long))


def split_long(batch, padded_seconds):
    """Run a micro-batch above `padded_seconds` of padded audio as halves (recursively), each trimmed to its own padding: same clips,
    same CTC sum (each part is weighted by its clip count); only BatchNorm statistics see the smaller part. Bounds peak memory."""
    ids, wave, lens, lab, tl, lang, cond = batch
    if wave.shape[0] < 2 or wave.shape[0] * wave.shape[1] / 16000 <= padded_seconds: return [batch]
    h = wave.shape[0] // 2; out = []
    for sl in (slice(0, h), slice(h, None)):
        out += split_long((ids[sl], wave[sl, :int(lens[sl].max())], lens[sl], lab[sl, :max(1, int(tl[sl].max()))], tl[sl],
                           lang[sl], cond[sl]), padded_seconds)
    return out


class RankSlice(Sampler):
    """This rank's micro-batches of the global update order: chunks of `micro` indices, every world-th starting at rank."""
    def __init__(self, order, micro, rank, world):
        self.order, self.micro, self.rank, self.world = order, micro, rank, world

    def __iter__(self):
        for c in range(self.rank * self.micro, len(self.order), self.micro * self.world): yield from self.order[c:c + self.micro]

    def __len__(self):
        return len(self.order) // self.world


def loader_workers():
    return int(os.getenv('LOADER_WORKERS', 4))
