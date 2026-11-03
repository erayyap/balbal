"""`balbal wyoming`: speech-to-text for Home Assistant's voice assistant over the Wyoming protocol (pip install 'balbal[wyoming]').

  balbal wyoming [--model manas] [--uri tcp://0.0.0.0:10300]
then in Home Assistant: Settings -> Devices & services -> Add integration -> Wyoming Protocol -> host of this machine, port 10300, and
pick balbal as the speech-to-text engine of a voice assistant.

Every voice command is transcribed whole at AudioStop (full-context decoding; a few tens of milliseconds for a command). Audio of any
rate / width / channels is converted to 16 kHz mono. The Transcribe event's model name picks a served model; language 'en' goes to
balbal-v2-manas when it is served. --use-ha-names turns the names / terms Home Assistant sends with a request (transcript_names,
transcript_terms) into hotwords, next to the fixed --hotwords. A command where no word reaches the no-speech confidence (silence, noise;
result['no_speech']) gets an empty transcript, so it cannot trigger an action; --keep-no-speech returns its text anyway.
"""
import asyncio
from functools import partial
import sys

import numpy as np

from . import __version__
from .engine import default_threads
from .hub import model_name
from .server import BusyError, Pool

LANGUAGES = {'balbal-v2': ['tr'], 'balbal-v2-manas': ['tr', 'en']}
MAX_HOTWORDS = 200


def info_event(models):
    from wyoming.info import AsrModel, AsrProgram, Attribution, Info
    who = Attribution(name='Eray Yapağcı', url='https://erayy.com/balbal')
    return Info(asr=[AsrProgram(
        name='balbal', description='balbal: open Turkish speech recognition on CPU', attribution=who, installed=True, version=__version__,
        models=[AsrModel(name=m, description=f'{m} ({", ".join(LANGUAGES[m])})', attribution=who, installed=True, version=__version__,
                         languages=LANGUAGES[m]) for m in models])]).event()


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
                served = self.pool.models
                name = None
                if t.name:
                    try: name = model_name(t.name)
                    except ValueError: name = None
                if name not in served:
                    name = 'balbal-v2-manas' if (t.language or '').startswith('en') and 'balbal-v2-manas' in served else served[0]
                self.model = name
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


def serve(models, uri='tcp://0.0.0.0:10300', workers=1, threads=None, runtime='fast', hotwords=None, use_ha_names=False, itn=True,
          keep_no_speech=False):
    try:
        from wyoming.server import AsyncServer
    except ImportError:
        raise SystemExit("balbal wyoming needs the wyoming extra: pip install 'balbal[wyoming]'")
    models = [model_name(m) for m in models] or ['balbal-v2']
    threads = threads or max(1, default_threads() // workers)
    print(f'balbal: loading {", ".join(models)} ({workers} worker(s) x {threads} thread(s), {runtime} runtime)', file=sys.stderr, flush=True)
    pool = Pool(models, workers, threads, runtime, backlog=64)
    hotwords = [w.strip() for w in (hotwords or '').split(',') if w.strip()]
    handler = partial(handler_class(), pool, info_event(models), hotwords, use_ha_names, itn, keep_no_speech)
    print(f'balbal: Wyoming speech-to-text on {uri}', file=sys.stderr, flush=True)
    try:
        asyncio.run(AsyncServer.from_uri(uri).run(handler))
    except KeyboardInterrupt:
        pass
