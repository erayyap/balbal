"""`balbal wyoming`: speech-to-text for Home Assistant's voice assistant over the Wyoming protocol (pip install 'balbal[wyoming]').

  balbal wyoming [--model manas] [--uri tcp://0.0.0.0:10300]
then in Home Assistant: Settings -> Devices & services -> Add integration -> Wyoming Protocol -> host of this machine, port 10300, and
pick balbal as the speech-to-text engine of a voice assistant.

Every voice command is transcribed whole at AudioStop (full-context decoding; a few tens of milliseconds for a command). Audio of any
rate / width / channels is converted to 16 kHz mono. The Transcribe event's model name picks a served model; otherwise its language
goes to the first served model with that language (e.g. 'en' -> balbal-v2-manas), else to the first served model. --use-ha-names turns the names / terms Home Assistant sends with a request (transcript_names,
transcript_terms) into hotwords, next to the fixed --hotwords. A command where no word reaches the no-speech confidence (silence, noise;
result['no_speech']) gets an empty transcript, so it cannot trigger an action; --keep-no-speech returns its text anyway.
"""
import asyncio
from functools import partial
import sys

import numpy as np

from . import __version__
from .engine import default_threads
from . import lang
from .server import BusyError, Pool

MAX_HOTWORDS = 200


def info_event(languages):
    """languages: {served model name: its language codes}."""
    from wyoming.info import AsrModel, AsrProgram, Attribution, Info
    who = Attribution(name='Eray Yapağcı', url='https://erayy.com/balbal')
    return Info(asr=[AsrProgram(
        name='balbal', description='balbal: open speech recognition on CPU', attribution=who, installed=True, version=__version__,
        models=[AsrModel(name=m, description=f'{m} ({", ".join(ls)})', attribution=who, installed=True, version=__version__,
                         languages=list(ls)) for m, ls in languages.items()])]).event()


def handler_class():
    from wyoming.asr import Transcribe, Transcript
    from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
    from wyoming.info import Describe
    from wyoming.server import AsyncEventHandler

    class Handler(AsyncEventHandler):
        def __init__(self, pool, info, hotwords, use_ha_names, itn, keep_no_speech, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pool, self.info, self.hotwords, self.use_ha_names, self.itn = pool, info, hotwords, use_ha_names, itn
            self.keep_no_speech = keep_no_speech
            self.converter = AudioChunkConverter(rate=16000, width=2, channels=1)
            self.reset()

        def reset(self):
            self.audio, self.model, self.language, self.request_words = [], self.pool.models[0], None, []

        async def handle_event(self, event):
            if Describe.is_type(event.type):
                await self.write_event(self.info); return True
            if Transcribe.is_type(event.type):
                t = Transcribe.from_event(event)
                self.language = t.language
                self.model = choose(self.pool, t.name, t.language)
                if self.use_ha_names:
                    self.request_words = [w for w in (getattr(t, 'transcript_names', None) or []) + (getattr(t, 'transcript_terms', None) or [])
                                          if isinstance(w, str) and w.strip()]
                return True
            if AudioStart.is_type(event.type):
                self.audio = []; return True
            if AudioChunk.is_type(event.type):
                self.audio.append(self.converter.convert(AudioChunk.from_event(event)).audio); return True
            if AudioStop.is_type(event.type):
                pcm = b''.join(self.audio)
                wave = np.frombuffer(pcm[:len(pcm) // 2 * 2], '<i2').astype(np.float32) / 32768.
                words = (self.hotwords + self.request_words)[:MAX_HOTWORDS] or None
                text = ''
                if len(wave):
                    model, itn = self.model, self.itn
                    try:
                        result = await asyncio.get_running_loop().run_in_executor(
                            None, self.pool.run, lambda w: w[model].transcribe(wave, hotwords=words, itn=itn))
                        text = '' if result['no_speech'] and not self.keep_no_speech else result['text']
                    except BusyError:
                        text = ''
                await self.write_event(Transcript(text=text, language=self.language).event())
                self.reset()
                return False
            return True

    return Handler


def choose(pool, name=None, language=None):
    """Served model for a request: the named one if it is served, else the first served model with the request's language, else the
    first served model."""
    if name:
        try: return pool.lookup(name)
        except ValueError: pass
    if language:
        want = lang.code(language)
        for m in pool.models:
            if want in (lang.code(x) for x in pool.loaded[m].languages): return m
    return pool.models[0]


def serve(models, uri='tcp://0.0.0.0:10300', workers=1, threads=None, runtime='fast', hotwords=None, use_ha_names=False, itn=True,
          keep_no_speech=False):
    try:
        from wyoming.server import AsyncServer
    except ImportError:
        raise SystemExit("balbal wyoming needs the wyoming extra: pip install 'balbal[wyoming]'")
    sources = [str(m) for m in models] or ['balbal-v2']
    threads = threads or max(1, default_threads() // workers)
    print(f'balbal: loading {", ".join(sources)} ({workers} worker(s) x {threads} thread(s), {runtime} runtime)', file=sys.stderr, flush=True)
    pool = Pool(sources, workers, threads, runtime, backlog=64)
    hotwords = [w.strip() for w in (hotwords or '').split(',') if w.strip()]
    info = info_event({m: pool.loaded[m].languages for m in pool.models})
    handler = partial(handler_class(), pool, info, hotwords, use_ha_names, itn, keep_no_speech)
    print(f'balbal: Wyoming speech-to-text on {uri}', file=sys.stderr, flush=True)
    try:
        asyncio.run(AsyncServer.from_uri(uri).run(handler))
    except KeyboardInterrupt:
        pass
