"""Stream an audio file to server.py like a live microphone and print the live transcript.

  python inference/streaming/client.py AUDIO [--url ws://127.0.0.1:8791/ws] [--piece .1] [--speed 1 (0 = as fast as possible)] [--quiet]
Prints partials (~) and finals (=); at the end: the transcript, the time from the last audio to the final transcript, server stats.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from balbal.audio import read_audio


async def run(a):
    wave = np.asarray(read_audio(a.audio), np.float32); n = int(a.piece * 16000)
    pcm = (np.clip(wave, -1, 1) * 32767).astype('<i2')
    async with aiohttp.ClientSession() as s, s.ws_connect(a.url, max_msg_size=8 << 20) as ws:
        ready = json.loads((await ws.receive()).data); print('#', ready['model'], ready['config'], file=sys.stderr)
        t_start = time.perf_counter(); first = None

        async def sender():
            for i in range(0, len(pcm), n):
                if a.speed > 0:   # a piece exists once it ends
                    await asyncio.sleep(max(0., t_start + min(i + n, len(pcm)) / 16000 / a.speed - time.perf_counter()))
                await ws.send_bytes(pcm[i:i + n].tobytes())
            await ws.send_str(json.dumps(dict(type='end'))); return time.perf_counter()

        task = asyncio.create_task(sender())
        async for msg in ws:
            e = json.loads(msg.data); now = time.perf_counter() - t_start
            if e['type'] in ('partial', 'final') and first is None: first = now
            if not a.quiet and e['type'] == 'partial': print(f'{now:7.2f}s  ~ {e["text"]}')
            if e['type'] == 'final': print(f'{now:7.2f}s  = {e["text"]}')
            if e['type'] == 'done':
                t_end = await task
                print('\nTRANSCRIPT:', e['transcript'])
                print(f'# audio {len(wave) / 16000:.2f} s, first text at {first if first is not None else float("nan"):.2f} s, final transcript '
                      f'{1000 * (time.perf_counter() - t_end):.0f} ms after the last audio; server compute '
                      f'{100 * e["stats"]["compute_rtf"]:.1f}% of real time ({e["stats"]["windows"]} windows)')
                break


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('audio'); p.add_argument('--url', default='ws://127.0.0.1:8791/ws')
    p.add_argument('--piece', type=float, default=.1); p.add_argument('--speed', type=float, default=1.); p.add_argument('--quiet', action='store_true')
    asyncio.run(run(p.parse_args()))


if __name__ == '__main__':
    main()
