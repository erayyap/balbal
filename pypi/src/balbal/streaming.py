"""Live (streaming) recognition: audio arrives piece by piece, text comes back as it is heard.

    import balbal
    model = balbal.load('balbal-v2')
    stream = model.stream()                         # one per live audio stream
    for piece in pieces:                            # float32 in [-1, 1] or int16, mono, 16 kHz, any piece size
        for event in stream.accept(piece): print(event)
    for event in stream.finish(): print(event)
    print(stream.transcript())

Events:
  {'type': 'partial', 'text', 'stable', 'start', 'joins', 'audio_end'}   the open segment, including a provisional tail over the
                                                                        lookahead; 'stable' = its leading words that no longer change
  {'type': 'final',   'text', 'start', 'end', 'joins'}                   a closed segment (pause, length cap or end of stream)
joins: the segment's first piece continues the previous segment's last word (no space between them). Times in seconds.

The models are full-context (bidirectional attention), so streaming re-runs the acoustic graph on recent audio instead of caching
states. Two modes (same code as inference/streaming of github.com/erayyap/balbal):
  segment (default)  every 0.32 s of new audio the graph runs once over the open segment (+ 2 s of left context) and the whole segment
          is re-decoded; text older than ~0.64 s has full context on both sides, so partials equal the final except for the last
          ~0.64 s. A segment closes after 0.5 s of quiet, blank audio, or at 5 s (cut at a short pause first).
  window  a sliding window (8 s left context, 0.32 s chunk, 0.64 s lookahead) commits every chunk once to an incremental beam search;
          a closed segment is re-decoded with full context for its final text.
Decoder: the native incremental search (_native.ctc_stream_decoder), the same best path as the offline decoder on the same frames.
After a segment the KenLM context carries over, as in one long utterance. A stream uses its model's ONNX session: like the model,
one stream must not be fed from two threads at once, and streams of one model must be fed from one thread (or use one model per thread).
"""
import math
import time

import numpy as np

from .engine import MIN_SAMPLES

FRAME, FPS = 640, 25   # samples per output frame (16 kHz, 10 ms hop, 4x subsampling), frames per second
MODES = ('segment', 'window')


class _Engine:
    """What the stream classes need from a loaded balbal.Model: its ONNX session, decoder settings, tokenizer and LM."""

    def __init__(self, model, hotwords=None):
        d = model.decoder
        self.model, self.session, self.cfg = model.model, model.session, model.settings
        self.sp, self.v = d.sp, d.vocab_size
        self.lm, self.upper = d.wrapped_lm, d.bonus
        if hotwords:   # the hotword search of decoder.decode_words, over the streaming decoder
            seqs, bonuses = d.hotword_sequences(tuple(hotwords), float(self.cfg.get('hotword_bonus', 1.0)))
            self.lm = d._HotwordLM(d.wrapped_lm, seqs, bonuses)
            self.upper = d.bonus + max([max(b) for b in bonuses if b] + [0.])
        self.scale = d.scale

    def forward(self, wave):
        """float32 [n] -> log-probs [n // 640, V + 1] (blank last); n is a multiple of 640 except for the real end of a stream."""
        wave = np.ascontiguousarray(wave, dtype=np.float32)
        if len(wave) >= MIN_SAMPLES: return self.session.run(None, {'audio': wave[None]})[0]
        lp = self.session.run(None, {'audio': np.pad(wave, (0, MIN_SAMPLES - len(wave)))[None]})[0]   # very short stream
        return lp[:max(1, -(-len(wave) // FRAME))]

    def decoder(self):
        from ._native.ctc_stream_decoder import StreamCTCDecoder
        c = self.cfg
        return StreamCTCDecoder(int(c['beam']), int(c['top_tokens']), float(c['threshold']), self.scale, self.lm, self.v, self.upper)

    def text(self, ids):
        return self.sp.decode([int(t) for t in ids if 0 <= t < self.v])


class Emissions:
    """Streamed decoder.prepare: temperature (per frame) + blank collapse (keep the last blank of every confident-blank run; needs the
    next frame, so one frame is held back until more frames or the end arrive)."""
    def __init__(self, cfg):
        self.t = cfg.get('temperature', 1.); self.bias = cfg.get('blank_bias', 0.); c = cfg.get('blank_collapse')
        self.thr = None if c is None else math.log(c); self.pending = None

    def scale(self, lp):
        a = np.ascontiguousarray(lp, dtype=np.float32)
        if self.t != 1 or self.bias != 0:
            a = a / self.t; a[:, -1] += self.bias
            m = a.max(1, keepdims=True); a = a - (m + np.log(np.exp(a - m).sum(1, keepdims=True)))
        return a

    def collapse(self, a, pending, final):
        """Frames ready for the decoder from pending + a; returns (ready, new pending)."""
        x = a if pending is None else np.concatenate([pending, a])
        if self.thr is None or not len(x): return (x, None) if final or not len(x) else (x[:-1], x[-1:])
        conf = x[:, -1] > self.thr; keep = np.ones(len(x), bool); keep[:-1] = ~(conf[:-1] & conf[1:])
        if final: return np.ascontiguousarray(x[keep]), None
        return np.ascontiguousarray(x[:-1][keep[:-1]]), x[-1:]

    def push(self, lp, final=False):
        ready, self.pending = self.collapse(self.scale(lp), self.pending, final); return ready

    def preview(self, lp):
        """Frames the decoder would see if lp were appended and the stream ended (provisional text; state unchanged)."""
        return self.collapse(self.scale(lp), self.pending, True)[0]


class Stream:
    """One live stream, window mode. Times in seconds; frames are 40 ms. first_right: lookahead of the first commit (the model puts the
    first token of an utterance on frame 0 of its input, so the start needs more right context). endpoint_mode: 'rebase' (default: the
    segment is final, the LM context and the CTC state carry over) or 'finish' (KenLM end of sentence + reset)."""
    def __init__(self, engine, left=8., chunk=.32, right=.64, first_right=1.6, endpoint=.5, endpoint_mode='rebase', max_segment=20.,
                 vad_db=30., provisional=True, second_pass=True, second_left=2., max_chunk=3., peak_decay=.02, trace=False):
        self.e = engine; self.dec = engine.decoder(); self.em = Emissions(engine.cfg)
        self.peak_decay = peak_decay   # dB per frame of the VAD peak
        self.second_pass = second_pass   # finals from one full-context forward over the closed segment + a persistent second decoder
        if second_pass: self.dec2, self.em2 = engine.decoder(), Emissions(engine.cfg)
        self.P = round(second_left * FPS)   # left context of the second pass (a cold start re-emits a word start at frame 0)
        self.L, self.C, self.R, self.R0 = (round(x * FPS) for x in (left, chunk, right, max(right, first_right)))
        self.endpoint, self.max_seg, self.vad_db, self.provisional = round(endpoint * FPS), round(max_segment * FPS), vad_db, provisional
        self.K = max(1, round(max_chunk / chunk))   # adaptive chunk: a window may commit up to K chunks when compute fell behind
        assert endpoint_mode in ('rebase', 'finish'); self.mode = endpoint_mode
        self.buf = np.zeros(0, np.float32); self.buf0 = 0   # buf[0] is global sample buf0
        self.done = 0            # committed frames
        self.seg_start = 0       # first frame of the open segment
        self.quiet = 0           # trailing committed frames that are blank (p > .95) AND quiet (energy < peak - vad_db)
        self.peak = -100.        # slowly decaying peak frame energy (dBFS)
        self.has_text = False
        self.voiced = 0          # frames of the open segment that are non-blank AND loud: speech, not the frame-0 token over silence
        self.final_ids = []      # token ids of all closed segments (the transcript is the decode of these: words split by a cut re-join)
        self.stats = dict(windows=0, model_s=0., decode_s=0., window_audio_s=0., second_s=0.)
        self.last_partial = None; self.trace = [] if trace else None   # trace: decoded log-probs (exactness tests)
        self.previewed = 0       # frames covered by the last preview-only window (before the first commit)
        self.real_end = None     # samples of real audio once finish() zero-pads the last frame (the second pass uses the real audio)

    # ------------------------------------------------------------------ input
    def accept(self, audio):
        """Feed audio (float32, 16 kHz, any length). Returns the events produced (zero or more)."""
        self.buf = np.concatenate([self.buf, np.asarray(audio, np.float32)]); ev = []
        while True:
            have, r = (self.buf0 + len(self.buf)) // FRAME, self.R if self.done else self.R0
            k = min(self.K, (have - self.done - r) // self.C)   # > 1 only when audio piled up (slow device, busy server)
            if k < 1: break
            ev += self._window(self.done + k * self.C + r, final=False, commit_to=self.done + k * self.C)
        if self.provisional and not self.done and have >= self.previewed + self.C:   # first commit still waits for R0: show text early
            ev += self._preview(have)
        return ev

    def finish(self):
        """End of stream: commit everything left, close the segment (KenLM end of sentence). Returns the remaining events."""
        ev = []; self.real_end = self.buf0 + len(self.buf); total = -(-self.real_end // FRAME)
        if total > self.done:
            pad = total * FRAME - (self.buf0 + len(self.buf))
            if pad: self.buf = np.concatenate([self.buf, np.zeros(pad, np.float32)])
            ev += self._window(total, final=True)
        return ev + self._close(end_of_sentence=True)

    def transcript(self):
        return self.e.text(self.final_ids)

    # ------------------------------------------------------------------ internals
    def _window(self, end, final, commit_to=None):
        start = max(0, self.done - self.L); s0 = start * FRAME - self.buf0
        t0 = time.perf_counter(); lp = self._relocate(self.e.forward(self.buf[s0:end * FRAME - self.buf0]), start); t1 = time.perf_counter()
        to = end if final else commit_to
        commit, look = lp[self.done - start:to - start], lp[to - start:]
        a = self.buf[self.done * FRAME - self.buf0:to * FRAME - self.buf0].reshape(-1, FRAME)
        db = 10 * np.log10(np.square(a, dtype=np.float64).mean(1) + 1e-10); pb = np.exp(commit[:, -1])
        ev = []; k = 0
        for j in range(len(commit)):   # endpoint / length cap, frame by frame
            self.peak = max(self.peak - self.peak_decay, db[j])
            loud = db[j] >= max(-60., self.peak - self.vad_db)
            self.quiet = self.quiet + 1 if pb[j] > .95 and not loud else 0
            self.voiced += pb[j] <= .95 and loud
            seg = self.done + j + 1 - self.seg_start
            ep = self.has_text and self.voiced >= 3 and self.quiet >= self.endpoint
            # length cap: cut in a short pause if possible, else at a blank frame, else anywhere
            cap = (seg >= self.max_seg and self.quiet >= 3) or (seg >= 1.5 * self.max_seg and pb[j] > .95) or seg >= 2 * self.max_seg
            if ep or cap:
                self._feed(commit[k:j + 1]); k = j + 1
                ev += self._close(end_of_sentence=ep and self.mode == 'finish', upto=self.done + k)
        self._feed(commit[k:]); self.done = to
        t2 = time.perf_counter()
        if not final:
            ids = self._call(self.dec.peek, self.em.preview(look)) if self.provisional and len(look) else self.dec.best()
            stable = self.e.text(self.dec.best()); text = self.e.text(ids)
            if text and text != self.last_partial:
                ev.append(dict(type='partial', text=text, stable=stable, start=self.seg_start / FPS, joins=self._joins(ids), audio_end=end / FPS))
            self.last_partial = text
        t3 = time.perf_counter()
        st = self.stats; st['windows'] += 1; st['model_s'] += t1 - t0; st['decode_s'] += t3 - t1; st['window_audio_s'] += (end - start) * FRAME / 16000
        keep = min(self.done - self.L, self.seg_start - self.P) if self.second_pass else self.done - self.L
        drop = max(0, keep * FRAME - self.buf0)   # keep only what the next window (and the second pass) can need
        if drop: self.buf = self.buf[drop:]; self.buf0 += drop
        return ev

    def _preview(self, end):
        """Provisional-only window on frames [0, end) before the first commit (decoder state unchanged)."""
        t0 = time.perf_counter(); lp = self._relocate(self.e.forward(self.buf[:end * FRAME - self.buf0]), 0); t1 = time.perf_counter()
        ids = self._call(self.dec.peek, self.em.preview(lp)); text = self.e.text(ids); ev = []
        if text and text != self.last_partial:
            ev.append(dict(type='partial', text=text, stable='', start=0., joins=False, audio_end=end / FPS))
        self.last_partial, self.previewed = text, end
        st = self.stats; st['windows'] += 1; st['model_s'] += t1 - t0; st['decode_s'] += time.perf_counter() - t1; st['window_audio_s'] += end * FRAME / 16000
        return ev

    def _relocate(self, lp, s0):
        """The model puts the first token of speech that follows silence on frame 0 of its input, whatever the input start (zero padding
        or a later start moves it along). Move that emission to just before the window's next emission, so window starts and cuts
        cannot drop or split it: only when frame 0 (global frame s0) is quiet and the next emission is > 8 frames away."""
        a = lp.argmax(1); blank = lp.shape[1] - 1
        if a[0] == blank: return lp
        x = self.buf[s0 * FRAME - self.buf0:(s0 + 1) * FRAME - self.buf0]
        if 10 * np.log10(np.square(x, dtype=np.float64).mean() + 1e-10) >= max(-60., self.peak - 20.): return lp
        nz = np.flatnonzero(a[1:] != blank); k = nz[0] + 1 if len(nz) else len(lp)
        if k <= 8: return lp
        out = lp.copy(); out[:k - 1] = lp[1:k]; out[k - 1] = lp[0]; return out

    @staticmethod
    def _call(fn, x):
        """Call a native decoder method on frames x; x stays referenced for the whole call (a bare .ctypes.data of a temporary dangles)."""
        x = np.ascontiguousarray(x, dtype=np.float32); return fn(x.ctypes.data, x.shape[0], x.shape[1])

    def _joins(self, ids):
        """True if the segment's first token continues the previous segment's last word (no word-start marker)."""
        return bool(ids) and bool(self.final_ids) and not self.e.sp.id_to_piece(int(ids[0])).startswith('▁')

    def _feed(self, lp):
        if not len(lp): return
        if self.trace is not None: self.trace.append(np.array(lp))
        ready = self.em.push(lp)
        if len(ready): self._call(self.dec.step, ready)
        if not self.has_text and len(self.dec.best()): self.has_text = True

    def _close(self, end_of_sentence, upto=None):
        upto = self.done if upto is None else upto
        if end_of_sentence:
            rest = self.em.push(np.zeros((0, self.e.v + 1), np.float32), final=True)
            if len(rest): self._call(self.dec.step, rest)
        ids = list(self.dec.finish() if end_of_sentence else self.dec.rebase())
        if self.second_pass and upto > self.seg_start:
            t0 = time.perf_counter(); have = (self.buf0 + len(self.buf)) // FRAME; end = min(have, upto + self.R)
            s0 = max(self.seg_start - self.P, -(-self.buf0 // FRAME))
            hi = end * FRAME if self.real_end is None else min(end * FRAME, self.real_end)
            lp = self._relocate(self.e.forward(self.buf[s0 * FRAME - self.buf0:hi - self.buf0]), s0)[self.seg_start - s0:upto - s0]
            t1 = time.perf_counter()
            ready = self.em2.push(lp, final=True)
            if len(ready): self._call(self.dec2.step, ready)
            ids = list(self.dec2.finish() if end_of_sentence else self.dec2.rebase())
            st = self.stats; st['model_s'] += t1 - t0; st['decode_s'] += time.perf_counter() - t1; st['second_s'] += time.perf_counter() - t0
            st['second_audio_s'] = st.get('second_audio_s', 0.) + (hi - s0 * FRAME) / 16000
        ev = []
        if ids:
            ev.append(dict(type='final', text=self.e.text(ids), start=self.seg_start / FPS, end=upto / FPS, joins=self._joins(ids)))
            self.final_ids += ids
        self.seg_start, self.has_text, self.last_partial, self.voiced = upto, False, None, 0
        return ev


class SegmentStream(Stream):
    """Segment mode. Every CHUNK of new audio the graph runs once over the open segment (+ SECOND_LEFT of left context) and the
    second decoder re-decodes the whole segment with peek() (state at the segment start, unchanged). stable = the word prefix two
    consecutive re-decodes agree on. Endpoint / length cap run over frames with >= RIGHT lookahead (same rules as Stream); a closed
    segment's final = step() of its frames from the same window + rebase (no extra forward). Leading silence of a segment without
    text is dropped from the window, so silence costs ~one short window per chunk. Window cost per chunk ~ the open segment's
    length: MAX_SEGMENT bounds it (cuts at short pauses first)."""
    def __init__(self, engine, chunk=.32, right=.64, endpoint=.5, endpoint_mode='rebase', max_segment=5., vad_db=30., second_left=2.,
                 max_chunk=3., peak_decay=.02, trace=False, **_):
        super().__init__(engine, left=0., chunk=chunk, right=right, first_right=right, endpoint=endpoint, endpoint_mode=endpoint_mode,
                         max_segment=max_segment, vad_db=vad_db, provisional=True, second_pass=True, second_left=second_left,
                         max_chunk=max_chunk, peak_decay=peak_decay, trace=trace)
        self.nonblank = 0        # non-blank frames of the open segment (any level): leading silence is dropped only while 0
        self.ran = 0             # frames covered by the last window
        self.prev_words = []     # words of the previous re-decode of the open segment (local agreement)

    def accept(self, audio):
        self.buf = np.concatenate([self.buf, np.asarray(audio, np.float32)])
        have = (self.buf0 + len(self.buf)) // FRAME
        return self._run(have, final=False) if have >= self.ran + self.C else []

    def finish(self):
        self.real_end = self.buf0 + len(self.buf); total = -(-self.real_end // FRAME)
        pad = total * FRAME - self.real_end
        if pad: self.buf = np.concatenate([self.buf, np.zeros(pad, np.float32)])
        return self._run(total, final=True)

    def _run(self, have, final):
        s0 = max(self.seg_start - self.P, -(-self.buf0 // FRAME))
        hi = have * FRAME if self.real_end is None else min(have * FRAME, self.real_end)
        t0 = time.perf_counter(); lp = self._relocate(self.e.forward(self.buf[s0 * FRAME - self.buf0:hi - self.buf0]), s0); t1 = time.perf_counter()
        have = min(have, s0 + len(lp)); lp = lp[:have - s0]   # the real (unpadded) last piece can give one frame less
        to = have if final else max(self.done, have - self.R); ev = []
        a = self.buf[self.done * FRAME - self.buf0:to * FRAME - self.buf0].reshape(-1, FRAME)
        db = 10 * np.log10(np.square(a, dtype=np.float64).mean(1) + 1e-10); pb = np.exp(lp[self.done - s0:to - s0, -1])
        for j in range(to - self.done):   # endpoint / length cap over frames with >= RIGHT lookahead
            f = self.done + j; self.peak = max(self.peak - self.peak_decay, db[j])
            loud = db[j] >= max(-60., self.peak - self.vad_db)
            self.quiet = self.quiet + 1 if pb[j] > .95 and not loud else 0
            self.voiced += pb[j] <= .95 and loud; self.nonblank += pb[j] <= .95
            seg = f + 1 - self.seg_start
            if not self.nonblank and self.quiet >= self.endpoint and f + 1 - self.endpoint > self.seg_start:
                self.seg_start = f + 1 - self.endpoint   # no speech yet: drop leading silence (keeps ENDPOINT of it)
                continue
            ep = self.voiced >= 3 and self.quiet >= self.endpoint
            cap = (seg >= self.max_seg and self.quiet >= 3) or (seg >= 1.5 * self.max_seg and pb[j] > .95) or seg >= 2 * self.max_seg
            if ep or cap: ev += self._final(lp, s0, f + 1, end_of_sentence=ep and self.mode == 'finish')
        self.done = to
        if final:
            ev += self._final(lp, s0, have, end_of_sentence=True)
        else:
            rows = lp[self.seg_start - s0:]
            ids = self._call(self.dec2.peek, self.em2.preview(rows)) if len(rows) else []
            text = self.e.text(ids); words = text.split()
            k = 0
            while k < min(len(words), len(self.prev_words)) and words[k] == self.prev_words[k]: k += 1
            self.prev_words = words
            if text and text != self.last_partial:
                ev.append(dict(type='partial', text=text, stable=' '.join(words[:k]), start=self.seg_start / FPS, joins=self._joins(ids),
                               audio_end=have / FPS))
            self.last_partial = text
        t2 = time.perf_counter(); self.ran = have
        st = self.stats; st['windows'] += 1; st['model_s'] += t1 - t0; st['decode_s'] += t2 - t1; st['window_audio_s'] += (have - s0) * FRAME / 16000
        drop = max(0, (self.seg_start - self.P) * FRAME - self.buf0)
        if drop: self.buf = self.buf[drop:]; self.buf0 += drop
        return ev

    def _final(self, lp, s0, upto, end_of_sentence):
        ev = []
        if upto > self.seg_start:
            if self.trace is not None: self.trace.append(np.array(lp[self.seg_start - s0:upto - s0]))
            ready = self.em2.push(lp[self.seg_start - s0:upto - s0], final=True)
            if len(ready): self._call(self.dec2.step, ready)
            ids = list(self.dec2.finish() if end_of_sentence else self.dec2.rebase())
            if ids:
                ev.append(dict(type='final', text=self.e.text(ids), start=self.seg_start / FPS, end=upto / FPS, joins=self._joins(ids)))
                self.final_ids += ids
        self.seg_start, self.last_partial, self.voiced, self.nonblank, self.prev_words = upto, None, 0, 0, []
        return ev


def pcm(audio):
    """A piece of live audio -> float32 in [-1, 1]: float arrays as they are, int16 arrays and raw bytes (little-endian int16 PCM)
    scaled by 1 / 32768."""
    if isinstance(audio, (bytes, bytearray, memoryview)): audio = np.frombuffer(audio, '<i2')
    a = np.asarray(audio)
    if a.ndim != 1: raise ValueError('mono audio expected (a 1-D array of samples)')
    if a.dtype.kind in 'iu': return a.astype(np.float32) / 32768.
    return a.astype(np.float32, copy=False)


class LiveStream:
    """One live stream over a loaded balbal.Model (see the module docstring); made by Model.stream().

    mode      'segment' (default) or 'window'
    hotwords  names / words expected in the audio, favoured while decoding
    itn       spoken numbers as digits in the event texts and the transcript (default True; the model's ITN, Turkish only)
    options   stream settings in seconds: chunk (.32), right (.64 lookahead), endpoint (.5 pause that closes a segment),
              max_segment (5 segment / 20 window), and for window mode left (8) and first_right (1.6)"""

    def __init__(self, model, mode='segment', hotwords=None, itn=True, **options):
        if mode not in MODES: raise ValueError(f'mode must be one of {MODES}')
        if isinstance(hotwords, str): hotwords = [w.strip() for w in hotwords.split(',') if w.strip()]
        self.mode = mode
        itn_text = model._itn[0] if itn and model._itn else None
        self.itn = itn_text
        self.engine = _Engine(model, hotwords or None)
        if options.get('max_segment') is None: options['max_segment'] = 5. if mode == 'segment' else 20.
        self.options = options
        self._stream = (SegmentStream if mode == 'segment' else Stream)(self.engine, **options)
        self.audio_seconds, self.finished = 0., False

    def _events(self, events):
        if not self.itn: return events
        for e in events:
            e['text'] = self.itn(e['text'])
            if e.get('stable'): e['stable'] = self.itn(e['stable'])
        return events

    def accept(self, audio):
        """Feed the next piece of audio (float32 / int16 / int16 bytes, mono, 16 kHz). Returns the events it produced (often none)."""
        if self.finished: raise RuntimeError('this stream has finished; make a new one')
        a = pcm(audio); self.audio_seconds += len(a) / 16000
        return self._events(self._stream.accept(a)) if len(a) else []

    def finish(self):
        """End of the audio: decodes what is left and closes the last segment. Returns the remaining events."""
        if self.finished: return []
        self.finished = True
        return self._events(self._stream.finish())

    def transcript(self):
        """Text of all closed segments (all of the audio after finish())."""
        text = self._stream.transcript()
        return self.itn(text) if self.itn else text

    @property
    def stats(self):
        """Compute spent so far: windows run, model / decoder seconds, audio seconds and compute_rtf (compute / audio)."""
        st = dict(self._stream.stats, audio_seconds=round(self.audio_seconds, 3))
        st['compute_rtf'] = round((st['model_s'] + st['decode_s']) / max(self.audio_seconds, 1e-6), 4)
        return st


def transcribe_stream(model, pieces, **kwargs):
    """Generator: the events of a LiveStream fed with `pieces` (an iterable of audio pieces), the finish() events last."""
    stream = LiveStream(model, **kwargs)
    for piece in pieces:
        yield from stream.accept(piece)
    yield from stream.finish()


def simulate(model, wave, piece=.16, **kwargs):
    """Stream a whole recording as if live, in pieces of `piece` seconds, as fast as the CPU allows. Returns (transcript, events,
    stream); every event gets 't': the time it would reach a live client = audio fed so far + compute of that call."""
    stream = LiveStream(model, **kwargs); events = []; n = max(1, int(piece * 16000)); wave = pcm(wave)
    for i in range(0, len(wave), n):
        t0 = time.perf_counter(); e = stream.accept(wave[i:i + n]); t = min(i + n, len(wave)) / 16000 + time.perf_counter() - t0
        events += [dict(x, t=t) for x in e]
    t0 = time.perf_counter(); e = stream.finish(); t = len(wave) / 16000 + time.perf_counter() - t0
    events += [dict(x, t=t) for x in e]
    return stream.transcript(), events, stream
