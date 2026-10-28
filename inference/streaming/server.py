"""WebSocket streaming ASR server (aiohttp) for balbal-v2 / balbal-v2-manas on CPU.

  python inference/streaming/server.py --model-dir MODEL_DIR [--mode segment|window] [--port 8791] [--host 127.0.0.1]
                                       [--workers 1 --threads 8] [--chunk .32 --right .64 --endpoint .5 --max-segment S]
                                       [--left 8 --first-right 1.6 (window mode)]
Browser microphone demo at http://HOST:PORT/, GET /health.

Protocol (ws://HOST:PORT/ws):
  client -> server  binary messages: little-endian int16 PCM, mono, 16 kHz (any size; ~100-200 ms pieces are typical)
                    text {"type": "end"}: no more audio -> the server flushes, sends the last finals, then {"type": "done", ...}
  server -> client  {"type": "ready", "model", "config", "worker"}
                    {"type": "partial", "text": open segment incl. the provisional tail, "stable", "start", "joins", "audio_end"}
                    {"type": "final", "text", "start", "end", "joins"}      (joins: continues the previous segment's last word)
                    {"type": "done", "transcript", "stats"}
Streams are pinned round-robin to --workers compute threads (one ONNX session of --threads cores each, KenLM shared); asyncio handles
the sockets. Partials carry audio_end (seconds of audio the text covers), so a client can see how far behind real time the server is.
"""
import argparse
import asyncio
import itertools
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from aiohttp import WSMsgType, web

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import streamer as T


async def ws_handler(request):
    app = request.app; ws = web.WebSocketResponse(max_msg_size=8 << 20); await ws.prepare(request)
    loop = asyncio.get_running_loop(); k = next(app['next']) % len(app['engines'])   # pin the stream to one worker
    pool, engine = app['pools'][k], app['engines'][k]
    cfg = dict(app['stream_cfg']); stream = T.make_stream(engine, **cfg); t_open = time.time()
    await ws.send_json(dict(type='ready', model=engine.model, config=app['stream_cfg'], worker=k))

    async def send(events):
        for e in events: await ws.send_json(e)

    # the reader (below) queues audio; the processor feeds everything pending in one accept(), so a backlog (busy worker) is caught up
    # with bigger windows (adaptive chunk) instead of piling up in the socket
    pending, wake, state = [], asyncio.Event(), dict(ended=False, audio_s=0.)

    async def processor():
        while True:
            await wake.wait(); wake.clear()
            if pending:
                pcm = np.concatenate(pending); pending.clear()
                await send(await loop.run_in_executor(pool, stream.accept, pcm))
            if state['ended'] and not pending:
                await send(await loop.run_in_executor(pool, stream.finish))
                st = dict(stream.stats, audio_s=round(state['audio_s'], 2), wall_s=round(time.time() - t_open, 2))
                st['compute_rtf'] = round((st['model_s'] + st['decode_s']) / max(state['audio_s'], 1e-6), 4)
                await ws.send_json(dict(type='done', transcript=stream.transcript(), stats=st)); return

    task = asyncio.create_task(processor())
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                pcm = np.frombuffer(msg.data, '<i2').astype(np.float32) / 32768.; state['audio_s'] += len(pcm) / 16000
                pending.append(pcm); wake.set()
            elif msg.type == WSMsgType.TEXT:
                if json.loads(msg.data).get('type') == 'end':
                    state['ended'] = True; wake.set(); await task; break
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        if not task.done(): task.cancel()
        await ws.close()
    return ws


async def index(request):
    return web.FileResponse(HERE / 'demo.html')


async def health(request):
    app = request.app
    return web.json_response(dict(ok=True, model=app['engines'][0].model, config=app['stream_cfg'], workers=len(app['engines'])))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-dir', type=Path, default=os.getenv('BALBAL_MODEL_DIR'), required=not os.getenv('BALBAL_MODEL_DIR'))
    p.add_argument('--model', choices=T.M.MODELS, default=None, help='default: MODEL_DIR/model.json')
    p.add_argument('--runtime', choices=('fast', 'onnx'), default='fast')
    p.add_argument('--lm-dir', type=Path, default=None, help='default: MODEL_DIR/lm')
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8791)
    p.add_argument('--threads', type=int, default=8, help='threads per worker')
    p.add_argument('--workers', type=int, default=1, help='ONNX sessions (one compute thread each); KenLM is shared')
    p.add_argument('--mode', default='segment', choices=('segment', 'window'))
    p.add_argument('--chunk', type=float, default=.32); p.add_argument('--right', type=float, default=.64)
    p.add_argument('--endpoint', type=float, default=.5)
    p.add_argument('--max-segment', type=float, default=None, help='default: 5 s (segment), 20 s (window)')
    p.add_argument('--left', type=float, default=8., help='window mode'); p.add_argument('--first-right', type=float, default=1.6, help='window mode')
    a = p.parse_args()
    app = web.Application()
    first = T.Engine(a.model_dir, a.model, a.runtime, a.threads, lm_dir=a.lm_dir)
    app['engines'] = [first] + [T.Engine(a.model_dir, a.model, a.runtime, a.threads, share=first) for _ in range(a.workers - 1)]
    app['pools'] = [ThreadPoolExecutor(1) for _ in app['engines']]; app['next'] = itertools.count()
    cfg = dict(mode=a.mode, chunk=a.chunk, right=a.right, endpoint=a.endpoint, max_segment=a.max_segment or (5. if a.mode == 'segment' else 20.))
    if a.mode == 'window': cfg.update(left=a.left, first_right=a.first_right)
    app['stream_cfg'] = cfg
    app.add_routes([web.get('/ws', ws_handler), web.get('/', index), web.get('/health', health)])
    print(f'serving {first.model} on http://{a.host}:{a.port} (ws /ws), {a.workers} x {a.threads} threads, mode {a.mode}', flush=True)
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == '__main__':
    main()
