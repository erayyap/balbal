"""Transcripts of a folder of clips as JSON ({clip: {text, words}}) for the cross-platform comparison in CI; with two JSON files,
compares them instead (exit status 1 if any text differs).
  python .github/scripts/transcripts.py MODEL CLIPS_DIR OUT.json
  python .github/scripts/transcripts.py --compare A.json B.json [...]"""
import json
from pathlib import Path
import sys


def transcribe(model_name, clips, out):
    import balbal
    model = balbal.load(model_name)
    rows = {}
    for f in sorted(Path(clips).glob('*.wav')):
        r = model.transcribe(f)
        rows[f.name] = dict(text=r['text'], words=[[w['word'], round(w['start'], 2), round(w['end'], 2)] for w in r['words']])
        print(f.name, r['text'], flush=True)
    Path(out).write_text(json.dumps(dict(model=model_name, cpu=cpu(), clips=rows), ensure_ascii=False, indent=1), encoding='utf-8')


def cpu():
    import platform
    from balbal import engine
    return dict(platform=sys.platform, machine=platform.machine(), ops=engine.ops_library().name)


def compare(files):
    runs = [(f, json.loads(Path(f).read_text(encoding='utf-8'))) for f in files]
    (base_name, base), status = runs[0], 0
    for name, run in runs[1:]:
        same_words = 0
        for clip, ref in base['clips'].items():
            got = run['clips'].get(clip)
            if got is None or got['text'] != ref['text']:
                status = 1; print(f'DIFF {clip}\n  {base_name}: {ref["text"]}\n  {name}: {got and got["text"]}')
            elif got['words'] == ref['words']:
                same_words += 1
        print(f'{name} ({run["cpu"]}) vs {base_name} ({base["cpu"]}): {len(base["clips"])} clips, texts '
              f'{"identical" if not status else "DIFFER"}, word timings identical on {same_words}')
    return status


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')   # Windows runners: cp1252 has no ı / ş
    if sys.argv[1] == '--compare': sys.exit(compare(sys.argv[2:]))
    transcribe(*sys.argv[1:4])
