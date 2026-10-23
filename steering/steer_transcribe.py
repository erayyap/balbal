"""Transcribe with balbal-v1, optionally steering its acoustic-memory reads toward a sentence.

  python steering/steer_transcribe.py --model-dir MODEL_DIR audio.wav                          # plain
  python steering/steer_transcribe.py --model-dir MODEL_DIR --mode auto audio.wav              # toward Whisper's text
  python steering/steer_transcribe.py --model-dir MODEL_DIR --mode weak --text "..." audio.wav
  python steering/steer_transcribe.py --model-dir MODEL_DIR --mode strong --text "..." audio.wav

Run in the inference environment (inference/setup_env.py). MODEL_DIR = the weights (huggingface.co/erayyapagci/balbal-v1).
auto mode runs Whisper-large-v3-turbo (pip install transformers; --turbo = Hugging Face id or local directory) and steers toward
its text with the weak config. Digits in the sentence are spelled as words before steering; keep case and punctuation.
Clips longer than 30 s are cut into fixed 30 s chunks (no VAD); each chunk is steered and decoded separately.
Prints one JSON line per file: text, plain_text, target, guarded (true = sentence refused), steered_frames, timings."""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIGS = json.loads((HERE / 'steer.json').read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('audio', nargs='+')
    p.add_argument('--model-dir', type=Path, default=Path(os.getenv('BALBAL_MODEL_DIR', HERE.parent / 'models')))
    p.add_argument('--mode', choices=['plain', 'auto', 'weak', 'strong'], default='plain')
    p.add_argument('--text', help='target sentence (weak / strong); one sentence for all files')
    p.add_argument('--turbo', default=CONFIGS['auto']['teacher'])
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.mode in ('weak', 'strong') and not a.text:
        p.error('--text is required for weak / strong')
    sys.path[:0] = [str(HERE.parent / 'inference'), str(HERE.parent), str(HERE)]
    from transcribe import Transcriber, read_audio
    from steer import Steerer
    from spelling import spell2
    tr = Transcriber(a.model_dir, a.device, a.threads)
    for q in tr.model.parameters(): q.requires_grad_(False)
    steerer = Steerer(tr.model)
    teacher = None
    if a.mode == 'auto':
        from teacher import TurboTeacher
        teacher = TurboTeacher(a.turbo, tr.device, int(CONFIGS['auto']['beam']))
    cfg = CONFIGS[CONFIGS['auto']['config'] if a.mode == 'auto' else a.mode] if a.mode != 'plain' else None
    for path in a.audio:
        wave = read_audio(path); step = 30 * 16000; parts = []
        for start in range(0, len(wave), step):
            chunk = wave[start:start + step]
            if len(chunk) < 4000: chunk = np.pad(chunk, (0, 4000 - len(chunk)))   # the subsampler needs at least 0.25 s
            t = {}; tick = time.perf_counter()
            text = teacher.transcribe(chunk) if teacher else a.text
            if teacher: t['teacher'] = time.perf_counter() - tick
            target = spell2(text.replace('²', '2').replace('³', '3')) if cfg and text else ''
            tick = time.perf_counter()
            r = steerer.run(chunk, target, steps=cfg['steps'], lr=cfg['lr'], gate=cfg['gate'], guard=cfg['guard']) if cfg else \
                steerer.run(chunk, '', guard=1)
            t['acoustic_and_steer'] = time.perf_counter() - tick; tick = time.perf_counter()
            out = dict(text=tr.decoder.decode(r['log_probs']))
            out['plain_text'] = tr.decoder.decode(r['plain_log_probs']) if cfg else out['text']
            t['decode'] = time.perf_counter() - tick
            if cfg: out.update(teacher_text=text if teacher else None, target=target, guarded=r['guarded'],
                               steered_frames=round(r['steered_frames'], 3))
            out['timings_ms'] = {k: round(1000 * v, 1) for k, v in t.items()}; parts.append(out)
        res = dict(file=path, mode=a.mode, text=' '.join(x['text'] for x in parts if x['text']),
                   plain_text=' '.join(x['plain_text'] for x in parts if x['plain_text']), chunks=parts)
        print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
