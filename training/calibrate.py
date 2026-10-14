"""DEV-only decoder calibration check for a new checkpoint + its own LM.

A 9-point grid around the deployed decoder (configs/decoder.json: alpha 0.75, beta 1.5, temperature 1.2, beam 64,
top-12 tokens, threshold 8): alpha {0.6, 0.75, 0.9} x temperature {1.0, 1.2, 1.4}, v2 WER/CER on DEV.
Switch rule: the best grid point replaces the deployed settings only if its WER is lower with a paired clip-level
bootstrap 95% CI (2000 resamples, seed 20261009) entirely below 0 and its CER is no worse. For balbal-v1 the
deployed settings were kept. Never calibrate on TEST.
  python training/calibrate.py --model-dir MODEL_DIR --dev dev.jsonl --out OUT_DIR [--device cuda] [--workers 8]
DEV rows: {"audio_path", "text", "source"} (clips <= 30 s). Writes OUT_DIR/check.json and OUT_DIR/dev-predictions.json.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'inference'), str(ROOT)]
_DECODER = {}


def _decode(task):
    lm_dir, settings, paths = task
    os.environ['OMP_NUM_THREADS'] = '1'
    import numpy as np
    from decoder import KenLMDecoder
    key = json.dumps(settings, sort_keys=True)
    if key not in _DECODER: _DECODER.clear(); _DECODER[key] = KenLMDecoder(lm_dir, settings)
    return [_DECODER[key].decode(np.load(p)) for p in paths]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-dir', type=Path, required=True); p.add_argument('--dev', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--device', default='cpu')
    p.add_argument('--workers', type=int, default=min(8, os.cpu_count()))
    args = p.parse_args()
    import numpy as np
    import torch
    from transcribe import build_model, read_audio
    from normalizer.metrics_v2 import corpus_metrics_v2
    out = args.out; out.mkdir(parents=True, exist_ok=True); cache = out / 'logprobs'; cache.mkdir(exist_ok=True)
    dev = [json.loads(l) for l in open(args.dev)]
    dev = [r for r in dev if r['text'].strip()]
    model = build_model(args.model_dir).to(args.device).float().eval()
    paths = []
    with torch.inference_mode():
        for i, r in enumerate(dev):
            path = cache / f'{i:06d}.npy'; paths.append(str(path))
            if path.exists(): continue
            w = read_audio(r['audio_path'])
            lp, n, _ = model(input_signal=torch.from_numpy(w)[None].to(args.device),
                             input_signal_length=torch.tensor([len(w)], device=args.device))
            np.save(path, lp[0, :int(n[0])].float().cpu().numpy()); model.ctc_extra.clear()
    del model
    deployed = json.loads((ROOT / 'configs/decoder.json').read_text())
    configs = [dict(deployed, alpha=a, temperature=t) for a, t in itertools.product([.6, .75, .9], [1., 1.2, 1.4])]
    refs = [r['text'] for r in dev]; lm_dir = str(args.model_dir / 'lm'); shards = 4 * args.workers
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        for c in configs:
            hyps = [None] * len(paths)
            for j, part in enumerate(pool.map(_decode, [(lm_dir, c, paths[j::shards]) for j in range(shards)])):
                for k, h in enumerate(part): hyps[j + shards*k] = h
            m = corpus_metrics_v2(refs, hyps)
            results.append(dict(config=c, wer=100*m['wer'], cer=100*m['cer'], hypotheses=hyps))
            print(json.dumps(dict(alpha=c['alpha'], temperature=c['temperature'], wer=round(100*m['wer'], 3),
                                  cer=round(100*m['cer'], 3))), flush=True)
    per_clip = lambda r: np.array([corpus_metrics_v2([x], [h])['word_errors'] for x, h in zip(refs, r['hypotheses'])], float)
    nw = np.array([corpus_metrics_v2([x], [x])['reference_words'] for x in refs], float)
    ref = next(r for r in results if r['config'] == deployed); best = min(results, key=lambda r: (r['wer'], r['cer']))
    eb, er = per_clip(best), per_clip(ref); rng = np.random.default_rng(20261009)
    idx = rng.integers(0, len(dev), (2000, len(dev)))
    diff = 100*(eb[idx].sum(1) - er[idx].sum(1))/nw[idx].sum(1); lo, hi = np.percentile(diff, [2.5, 97.5])
    switch = best is not ref and hi < 0 and best['cer'] <= ref['cer']
    chosen = best if switch else ref
    grid = [dict(alpha=r['config']['alpha'], temperature=r['config']['temperature'], wer=round(r['wer'], 3), cer=round(r['cer'], 3)) for r in results]
    check = dict(clips=len(dev), scoring='v2', grid=grid, deployed=dict(config=ref['config'], wer=ref['wer'], cer=ref['cer']),
                 best=dict(config=best['config'], wer=best['wer'], cer=best['cer']),
                 best_minus_deployed_wer_ci95=[float(lo), float(hi)], switch=bool(switch), chosen=chosen['config'])
    (out / 'check.json').write_text(json.dumps(check, indent=1) + '\n')
    (out / 'dev-predictions.json').write_text(json.dumps([dict(r, hypothesis=h) for r, h in zip(dev, chosen['hypotheses'])],
                                                         ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(dict(deployed_wer=ref['wer'], best=grid[results.index(best)], ci95=[round(lo, 3), round(hi, 3)],
                          switch=bool(switch))), flush=True)


if __name__ == '__main__':
    main()
