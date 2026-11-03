"""OpenAI-compatible transcription server: `pip install 'balbal[server]'`, then `balbal serve`.

  POST /v1/audio/transcriptions   multipart/form-data like OpenAI's endpoint: file, model, response_format (json, text,
                                  verbose_json, srt, vtt), timestamp_granularities[] (word, segment), language (echoed
                                  only), prompt / temperature (accepted, ignored); balbal extras: hotwords ("Eray,Hasan"),
                                  itn (true / false)
  WS   /v1/stream                 live transcription (balbal.streaming): send binary little-endian int16 PCM, mono, 16 kHz, in
                                  pieces of any size, then the text {"type": "end"}; receive JSON {"type": "ready", ...}, then
                                  "partial" / "final" events, then {"type": "done", "transcript", "stats"}. Query parameters:
                                  model, mode (segment / window), hotwords, itn; api_key (or the Authorization header; a wrong
                                  key refuses the handshake with 403). A refused stream gets {"type": "error", "message"} and is
                                  closed (1008 bad request, 1013 too many streams)
  GET  /v1/models                 the models this server serves (--model: registry names, model folders or Hugging Face repo ids)
  GET  /health
  GET  /                          browser microphone demo of /v1/stream (http://HOST:PORT/#key=KEY when an API key is set)

Any OpenAI client works by pointing its base URL at http://HOST:PORT/v1. OpenAI model names (whisper-1, gpt-4o-transcribe,
gpt-4o-mini-transcribe) map to the server's default model. A balbal.Model must not be used by two threads at once, so every
worker owns one Model per served model; requests wait for a free worker, and once --queue requests are waiting the next
one gets 503 (OpenAI clients retry it). A live stream is pinned to one worker (round robin) and takes it only while it decodes a
piece; at most --max-streams streams are open at once (the next one is closed with code 1013, try again later).
"""
import asyncio
import io
import itertools
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading

from .audio import canonical_audio, read_audio
from .engine import default_threads, Model
from .formats import group_words, segments_from_words, to_srt, to_vtt
from . import hub, lang

OPENAI_NAMES = ('whisper-1', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe', 'gpt-4o-transcribe-diarize')
FORMATS = ('json', 'text', 'verbose_json', 'srt', 'vtt')
LOCAL_HOSTS = ('127.0.0.1', 'localhost', '::1')


def decode_upload(data, filename=''):
    """Uploaded bytes -> 16 kHz mono float32. soundfile reads WAV / FLAC / OGG from memory; anything else (MP3, M4A,
    WebM, video) goes through a temporary file to read_audio's FFmpeg path."""
    if not data: raise ValueError('empty audio file')
    try:
        import soundfile as sf
        audio, rate = sf.read(io.BytesIO(data), dtype='float32')
        return canonical_audio(audio, rate)
    except ValueError:
        raise
    except Exception:
        pass
    suffix = os.path.splitext(filename or '')[1][:10]
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(data); f.flush()
        return read_audio(f.name)


class Pool:
    """Workers x served models; each worker owns its own Model instances and a lock, so one Model never runs in two threads.
    run() takes any free worker and fails fast once `backlog` requests are already waiting; run_on() uses one given worker
    (live streams stay on the worker whose Model they were made from). models: what to load (registry names, folders, repo ids);
    self.models: their display names (the keys of a worker), first = default."""

    def __init__(self, models, workers, threads, runtime, backlog):
        self.backlog, first = backlog, {}
        for source in models:
            m = Model(source, runtime, threads)
            if m.model in first: raise ValueError(f'two served models are both named {m.model!r}')
            first[m.model] = m
        self.models = list(first)
        self.loaded = first
        self.workers = [first] + [{name: Model(m.source, m.runtime, threads) for name, m in first.items()} for _ in range(workers - 1)]
        self.locks = [threading.Lock() for _ in self.workers]
        self.free = queue.Queue()
        for i in range(len(self.workers)): self.free.put(i)
        self.waiting, self.lock = 0, threading.Lock()
        self.turn = itertools.count()

    def run(self, fn):
        with self.lock:
            if self.waiting >= self.backlog: raise BusyError()
            self.waiting += 1
        try:
            i = self.free.get()
        finally:
            with self.lock: self.waiting -= 1
        try:
            with self.locks[i]: return fn(self.workers[i])
        finally:
            self.free.put(i)

    def run_on(self, i, fn):
        with self.locks[i]: return fn(self.workers[i])

    def next_worker(self):
        return next(self.turn) % len(self.workers)

    def lookup(self, model):
        """The served model a request names: '' / an OpenAI name -> the default; a display name, registry name or alias, folder or
        repo id of a served model -> its name. ValueError otherwise."""
        if not model or model in OPENAI_NAMES: return self.models[0]
        if model in self.loaded: return model
        try:
            where = str(hub.resolve(model)[1])
        except ValueError:
            where = None
        for name, m in self.loaded.items():
            if where is not None and where == m.source: return name
        raise ValueError(f'model {model} is not served here (served: {", ".join(self.models)})')

    def language_name(self, name):
        """OpenAI-style language of a served model: its language's name, 'auto' for a multilingual model."""
        languages = self.loaded[name].languages
        return 'auto' if len(languages) > 1 else lang.NAMES.get(lang.code(languages[0]), languages[0])


class BusyError(Exception):
    pass


def create_app(models=('balbal-v2',), workers=1, threads=None, runtime='fast', api_key=None, max_bytes=25 << 20, backlog=16,
               max_streams=8):
    try:
        from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
        from fastapi.concurrency import run_in_threadpool
        from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
    except ImportError:
        raise SystemExit("balbal serve needs the server extra: pip install 'balbal[server]'")
    from . import __version__

    sources = [str(m) for m in models] or ['balbal-v2']
    threads = threads or max(1, default_threads() // workers)
    print(f'balbal: loading {", ".join(sources)} ({workers} worker(s) x {threads} thread(s), {runtime} runtime)', file=sys.stderr, flush=True)
    pool = Pool(sources, workers, threads, runtime, backlog)
    models, default = pool.models, pool.models[0]
    app = FastAPI(title='balbal', version=__version__)

    def error(status, message, kind='invalid_request_error', code=None, headers=None):
        return JSONResponse({'error': {'message': message, 'type': kind, 'param': None, 'code': code}}, status_code=status, headers=headers)

    @app.middleware('http')
    async def check_key(request: Request, call_next):
        if api_key and request.url.path.startswith('/v1/'):
            if request.headers.get('authorization', '') != f'Bearer {api_key}':
                return error(401, 'Incorrect API key provided.', 'invalid_request_error', 'invalid_api_key')
        return await call_next(request)

    @app.get('/health')
    def health():
        return {'status': 'ok', 'models': models, 'version': __version__}

    @app.get('/v1/models')
    def list_models():
        return {'object': 'list', 'data': [{'id': m, 'object': 'model', 'created': 0, 'owned_by': 'balbal'} for m in models]}

    @app.post('/v1/audio/transcriptions')
    async def transcriptions(request: Request, file: UploadFile = File(...), model: str = Form(default),
                             response_format: str = Form('json'), language: str | None = Form(None),
                             prompt: str | None = Form(None), temperature: float | None = Form(None),
                             hotwords: str | None = Form(None), itn: bool = Form(True)):
        form = await request.form()
        granularities = form.getlist('timestamp_granularities[]') or form.getlist('timestamp_granularities')
        if response_format not in FORMATS:
            return error(400, f'response_format must be one of {", ".join(FORMATS)}', code='invalid_value')
        if any(g not in ('word', 'segment') for g in granularities):
            return error(400, 'timestamp_granularities must be word and/or segment', code='invalid_value')
        try:
            name = pool.lookup(model)
        except ValueError as e:
            return error(400, str(e), code='model_not_found')
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            return error(413, f'file larger than {max_bytes >> 20} MB', code='file_too_large')
        try:
            wave = await run_in_threadpool(decode_upload, data, file.filename)
        except (OSError, ValueError) as e:
            return error(400, f'could not read the audio file: {e}', code='invalid_audio')
        words_list = [w.strip() for w in (hotwords or '').split(',') if w.strip()] or None
        try:
            result = await run_in_threadpool(pool.run, lambda w: w[name].transcribe(wave, hotwords=words_list, itn=itn))
        except BusyError:
            return error(503, 'server busy, retry later', 'server_error', 'busy', headers={'Retry-After': '1'})
        text = result['text']
        if response_format == 'text':
            return PlainTextResponse(text + '\n')
        segments = segments_from_words(result['words'])
        if response_format == 'srt':
            return PlainTextResponse(to_srt(segments))
        if response_format == 'vtt':
            return PlainTextResponse(to_vtt(segments), media_type='text/vtt')
        if response_format == 'json':
            return {'text': text}
        body = {'task': 'transcribe', 'language': language or pool.language_name(name),
                'duration': result['duration_seconds'], 'text': text}
        if 'word' in granularities:   # OpenAI's word fields + balbal's confidence (clients ignore unknown fields)
            body['words'] = [dict(word=w['word'], start=w['start'], end=w['end'], confidence=w.get('confidence')) for w in result['words']]
        if 'segment' in granularities or not granularities:
            body['segments'] = [dict(dict(seek=0, tokens=[], temperature=0.0, avg_logprob=0.0, compression_ratio=0.0, no_speech_prob=0.0), **s)
                                for s in segments]
            # no_speech_prob: 1 - the segment's highest word confidence (a score, not a calibrated probability)
            for seg, seg_words in zip(body['segments'], group_words(result['words'])):
                confidences = [w['confidence'] for w in seg_words if 'confidence' in w]
                if confidences: seg['no_speech_prob'] = round(1. - max(confidences), 3)
        return body

    streams = dict(open=0)

    @app.websocket('/v1/stream')
    async def stream_socket(ws: WebSocket):
        q = ws.query_params
        if api_key and ws.headers.get('authorization', '') != f'Bearer {api_key}' and q.get('api_key') != api_key:
            await ws.close(code=1008, reason='invalid api key'); return
        async def refuse(code, message):   # accepted first, so the client gets the reason (a refused handshake is a bare 403)
            await ws.accept()
            await ws.send_json(dict(type='error', message=message))
            await ws.close(code=code, reason=message[:120])
        try:
            name = pool.lookup(q.get('model'))
            mode = q.get('mode', 'segment')
            if mode not in ('segment', 'window'): raise ValueError('mode must be segment or window')
        except ValueError as e:
            await refuse(1008, str(e)); return
        if streams['open'] >= max_streams:
            await refuse(1013, 'too many live streams, retry later'); return
        streams['open'] += 1
        try:
            await ws.accept()
            itn_on = q.get('itn', 'true').strip().lower() not in ('0', 'false', 'no', 'off')
            hot = [w.strip() for w in q.get('hotwords', '').split(',') if w.strip()] or None
            k = pool.next_worker()
            stream = await run_in_threadpool(pool.run_on, k, lambda w: w[name].stream(mode, hotwords=hot, itn=itn_on))
            await ws.send_json(dict(type='ready', model=name, config=dict(stream.options, mode=mode), worker=k))
            await _pump(ws, stream, lambda fn: run_in_threadpool(pool.run_on, k, lambda w: fn()), WebSocketDisconnect)
        finally:
            streams['open'] -= 1

    @app.get('/', response_class=HTMLResponse)
    def demo():
        return (Path(__file__).resolve().parent / 'web' / 'demo.html').read_text()

    return app


async def _pump(ws, stream, run, disconnect):
    """Reader + processor of one live stream: the reader queues audio, the processor feeds everything pending in one accept(), so
    a busy worker catches up with a bigger window instead of letting audio pile up in the socket."""
    pending, wake, state = [], asyncio.Event(), dict(ended=False)

    async def send(events):
        for e in events: await ws.send_json(e)

    async def processor():
        while True:
            await wake.wait(); wake.clear()
            if pending:
                piece = b''.join(pending); pending.clear()
                await send(await run(lambda: stream.accept(piece)))
            if state['ended'] and not pending:
                await send(await run(stream.finish))
                await ws.send_json(dict(type='done', transcript=stream.transcript(), stats=stream.stats))
                return

    task = asyncio.create_task(processor())
    try:
        while not task.done():
            receive = asyncio.create_task(ws.receive())
            await asyncio.wait({receive, task}, return_when=asyncio.FIRST_COMPLETED)
            if not receive.done():
                receive.cancel(); break
            msg = receive.result()
            if msg['type'] == 'websocket.disconnect': break
            if msg.get('bytes'):
                data = msg['bytes']
                if len(data) % 2: data = data[:-1]   # int16 samples
                pending.append(data); wake.set()
            elif msg.get('text'):
                try:
                    end = json.loads(msg['text']).get('type') == 'end'
                except (ValueError, AttributeError):
                    end = False
                if end:
                    state['ended'] = True; wake.set()
                    await task; break
    except disconnect:
        pass
    finally:
        if not task.done(): task.cancel()
        try:
            await ws.close()
        except Exception:
            pass


def serve(models, host='127.0.0.1', port=8000, workers=1, threads=None, runtime='fast', api_key=None, max_mb=25, backlog=16,
          max_streams=8):
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("balbal serve needs the server extra: pip install 'balbal[server]'")
    api_key = api_key or os.getenv('BALBAL_API_KEY') or None
    if not api_key and host not in LOCAL_HOSTS:
        print(f'balbal: WARNING: serving on {host} without an API key (set --api-key or BALBAL_API_KEY)', file=sys.stderr, flush=True)
    app = create_app(models, workers, threads, runtime, api_key, max_mb << 20, backlog, max_streams)
    uvicorn.run(app, host=host, port=port, log_level='info')
