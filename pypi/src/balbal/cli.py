"""balbal command line.

  balbal run speech.wav                      text of each file (balbal-v2; downloaded from Hugging Face on first use)
  balbal run --model manas a.wav b.mp3       balbal-v2-manas (Turkish + English)
  balbal run --model zellige a.wav           balbal-v2.1-zellige (Moroccan Darija; CC BY-NC-SA 4.0, non-commercial; no live streaming)
  balbal run --json speech.wav               one JSON line per file: text, per-word timestamps, timings
  balbal run --format srt -o subs/ videos/   one subtitle file per audio / video file of a folder (txt, json, srt, vtt)
  balbal run --jobs 4 -o out/ calls/         four processes at once (CPU threads are split between them)
  balbal run --hotwords "Eray,Hasan" call.wav
  balbal run --domain-lm my-lm/ notes.wav    a domain KenLM (training/build_lm.py on your own text) mixed into the general one
  balbal run --model owner/name speech.wav   a model trained on a balbal architecture (Hugging Face repo id or local folder)
  balbal download [--model manas]            fetch the model files now (e.g. while building an image)
  balbal live [--model manas]                live microphone transcription (pip install 'balbal[live]')
  balbal live --file speech.wav              a file played as if live (partial + final text as it is heard)
  balbal serve [--model manas] [--port 8000]  OpenAI-compatible transcription server + live WebSocket (pip install 'balbal[server]')
  balbal wyoming [--model manas]             speech-to-text for Home Assistant, Wyoming protocol on port 10300 (pip install 'balbal[wyoming]')
"""
import argparse
import os
from pathlib import Path
import sys

from . import __version__
from .engine import RUNTIMES
from .hub import LANGUAGE_NAMES, LICENSES, MODELS, download

COMMANDS = ('run', 'download', 'live', 'serve', 'wyoming')


def hub_kind(model):
    from .hub import resolve
    try: return resolve(model)[0]
    except ValueError: return None


def model_list():
    """'balbal-v2 (Turkish; balbal Community License 1.1), ...': the registry models with language and licence."""
    return '; '.join(f'{m} ({LANGUAGE_NAMES[m]}; {LICENSES[m]})' for m in MODELS)


def model_options(p):
    p.add_argument('--model', '-m', default='balbal-v2',
                   help=f'{model_list()} - short: v2, manas, zellige - a model folder or a Hugging Face repo id owner/name; default balbal-v2')
    p.add_argument('--runtime', choices=RUNTIMES, default='fast', help='fast (custom kernels, default) or onnx (standard operators)')
    p.add_argument('--revision', default=None, help='Hugging Face revision (default: the tested one, main for other repos)')


def lm_options(p):
    p.add_argument('--domain-lm', type=Path, default=None,
                   help='a domain KenLM directory (model.binary + the model\'s tokenizer.model, e.g. from training/build_lm.py) mixed '
                        'into the general LM')
    p.add_argument('--domain-weight', type=float, default=None, help='its weight in the mix, 0-1 (default 0.3)')


def lm_settings(args):
    """Decoder settings of --domain-lm / --domain-weight, or None."""
    if args.domain_lm is None and args.domain_weight is None: return None
    settings = {}
    if args.domain_lm is not None: settings['domain_lm'] = str(args.domain_lm.resolve())
    if args.domain_weight is not None: settings['domain_weight'] = args.domain_weight
    return settings


def utf8_output():
    """Windows: output piped to a file or another program uses the code page (cp1252, cp1254...), which lacks letters such as ş
    or ğ. Write it as UTF-8 (the console itself already takes any text)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure') and (stream.encoding or '').lower().replace('-', '') != 'utf8':
            stream.reconfigure(encoding='utf-8')


def main(argv=None):
    if sys.platform == 'win32': utf8_output()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith('-'): argv.insert(0, 'run')   # balbal speech.wav
    p = argparse.ArgumentParser(prog='balbal', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', action='version', version=f'balbal {__version__}')
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='transcribe audio files')
    r.add_argument('audio', type=Path, nargs='+', help='files or folders: WAV / FLAC / OGG, other formats (MP3, M4A, video) through FFmpeg')
    model_options(r)
    r.add_argument('--model-dir', type=Path, default=None, help='a local model directory instead of the download')
    r.add_argument('--threads', '-t', type=int, default=None, help='CPU threads (default: BALBAL_THREADS, else min(8, CPU count))')
    r.add_argument('--hotwords', default='', help='comma-separated names / words to favour, e.g. "Eray,Hasan"')
    r.add_argument('--no-itn', action='store_true', help='keep numbers as spoken words (default: digits, for Turkish models)')
    r.add_argument('--json', action='store_true', help='one JSON line per file (text, words with start/end seconds, timings)')
    r.add_argument('--format', '-f', choices=('txt', 'json', 'srt', 'vtt'), default='txt', help='output format (default txt)')
    r.add_argument('--output-dir', '-o', type=Path, default=None, help='write one file per input here (folder structure kept)')
    r.add_argument('--skip-existing', action='store_true', help='with --output-dir: skip inputs whose output file exists')
    r.add_argument('--jobs', '-j', type=int, default=1, help='processes transcribing at once (default 1)')
    lm_options(r)
    d = sub.add_parser('download', help='download the model files without transcribing')
    model_options(d)
    lv = sub.add_parser('live', help="live transcription from the microphone (pip install 'balbal[live]') or a file played as if live")
    model_options(lv)
    lv.add_argument('--model-dir', type=Path, default=None, help='a local model directory instead of the download')
    lv.add_argument('--threads', '-t', type=int, default=None, help='CPU threads (default: BALBAL_THREADS, else min(8, CPU count))')
    lm_options(lv)
    lv.add_argument('--file', type=Path, default=None, help='play this audio file as if live instead of the microphone')
    lv.add_argument('--speed', type=float, default=1., help='--file playback speed (1 = real time, 0 = as fast as possible)')
    lv.add_argument('--device', default=None, help='input device (number or name part; python -m sounddevice lists them)')
    lv.add_argument('--mode', choices=('segment', 'window'), default='segment', help='streaming mode (default segment)')
    lv.add_argument('--hotwords', default='', help='comma-separated names / words to favour, e.g. "Eray,Hasan"')
    lv.add_argument('--no-itn', action='store_true', help='keep numbers as spoken words (default: digits, for Turkish models)')
    lv.add_argument('--json', action='store_true', help='every event (partial / final) as a JSON line')
    lv.add_argument('--timestamps', action='store_true', help='start and end seconds before every final segment')
    lv.add_argument('--transcript', action='store_true', help='print the whole transcript at the end')
    for name, text in (('chunk', 'seconds of new audio per update (default .32)'), ('right', 'lookahead seconds (default .64)'),
                       ('endpoint', 'pause in seconds that closes a segment (default .5)'),
                       ('max-segment', 'longest segment in seconds (default 5 segment mode, 20 window mode)')):
        lv.add_argument('--' + name, type=float, default=None, help=text)
    s = sub.add_parser('serve', help="OpenAI-compatible server: POST /v1/audio/transcriptions (pip install 'balbal[server]')")
    s.add_argument('--model', '-m', action='append', default=None,
                   help='model(s) to serve (name, folder or repo id), repeatable; the first is the default (OpenAI names like whisper-1 '
                        'map to it)')
    s.add_argument('--runtime', choices=RUNTIMES, default='fast', help='fast (custom kernels, default) or onnx (standard operators)')
    s.add_argument('--host', default=os.getenv('BALBAL_HOST', '127.0.0.1'),
                   help='bind address (default: BALBAL_HOST, else 127.0.0.1; 0.0.0.0 for all interfaces)')
    s.add_argument('--port', type=int, default=8000)
    s.add_argument('--workers', type=int, default=1, help='concurrent transcriptions; each worker loads its own model copy')
    s.add_argument('--threads', '-t', type=int, default=None, help='CPU threads per worker (default: CPU threads / workers, up to 8)')
    s.add_argument('--api-key', default=None, help='require "Authorization: Bearer KEY" (default: BALBAL_API_KEY, else none)')
    s.add_argument('--max-mb', type=int, default=25, help='largest accepted upload in MB (default 25, as OpenAI)')
    s.add_argument('--queue', type=int, default=16, help='requests allowed to wait for a worker before 503 (default 16)')
    s.add_argument('--max-streams', type=int, default=8, help='live WebSocket streams open at once (default 8)')
    lm_options(s)
    y = sub.add_parser('wyoming', help="speech-to-text for Home Assistant over the Wyoming protocol (pip install 'balbal[wyoming]')")
    y.add_argument('--model', '-m', action='append', default=None,
                   help='model(s) to serve (name, folder or repo id), repeatable; the first is the default')
    y.add_argument('--runtime', choices=RUNTIMES, default='fast', help='fast (custom kernels, default) or onnx (standard operators)')
    y.add_argument('--uri', default='tcp://0.0.0.0:10300', help='listen address (default tcp://0.0.0.0:10300)')
    y.add_argument('--workers', type=int, default=1, help='concurrent transcriptions; each worker loads its own model copy')
    y.add_argument('--threads', '-t', type=int, default=None, help='CPU threads per worker (default: CPU threads / workers, up to 8)')
    y.add_argument('--hotwords', default='', help='comma-separated names / words to favour in every command, e.g. "salon,mutfak"')
    y.add_argument('--use-ha-names', action='store_true', help='also favour the names / terms Home Assistant sends with a request')
    y.add_argument('--no-itn', action='store_true', help='keep numbers as spoken words (default: digits, for Turkish models)')
    y.add_argument('--keep-no-speech', action='store_true', help='return the text of commands without confident speech (default: empty)')
    args = p.parse_args(argv)
    if args.command in ('run', 'live', 'serve'):
        if args.domain_lm is not None and not args.domain_lm.is_dir(): p.error(f'no such directory: {args.domain_lm}')
        args.settings = lm_settings(args)

    if args.command == 'wyoming':
        from .wyoming_server import serve as wyoming
        wyoming(args.model or ['balbal-v2'], args.uri, max(1, args.workers), args.threads, args.runtime, args.hotwords, args.use_ha_names,
                not args.no_itn, args.keep_no_speech)
        return

    if args.command == 'serve':
        from .server import serve
        serve(args.model or ['balbal-v2'], args.host, args.port, max(1, args.workers), args.threads, args.runtime, args.api_key, args.max_mb,
              max(0, args.queue), max(0, args.max_streams), settings=args.settings)
        return

    if args.command == 'live':
        if args.file is not None and not args.file.is_file(): p.error(f'no such file: {args.file}')
        if args.device is not None and args.device.isdigit(): args.device = int(args.device)
        from .live import main as live
        live(args)
        return

    if args.command == 'download':
        path = download(args.model, args.runtime, args.revision)
        if Path(path).is_dir() and hub_kind(args.model) == 'local':
            from .hub import license_notice
            license_notice(path)
        print(path)
        return
    missing = [str(f) for f in args.audio if not f.exists()]
    if missing: p.error('no such file: ' + ', '.join(missing))
    from .batch import run
    sys.exit(run(args))

if __name__ == '__main__':
    main()
