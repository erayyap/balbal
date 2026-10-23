"""balbal-v1 training loop (the final continuation stage that produced balbal-v1).

balbal-v1 was trained as
  TEL_P=0.5 EDGE_P=0.5 LOADER_WORKERS=8 python training/train.py runs/balbal-v1 \
      --init PREVIOUS_STAGE.pt --tokenizer-dir TOKENIZER_DIR --train train.jsonl --dev dev.jsonl \
      --updates 9000 --lr 1e-4 --eval-every 3000 --soup 0.5 --micro 16
and the released weights are the raw step-6000 checkpoint (checkpoints/step6000.pt).

- Updates of 32 clips (micro clips per forward/backward), sampled by EpochLengthSampler: length-matched 32-clip batches,
  batch order shuffled per epoch (seed + epoch), waveform augmentation drawn per (seed, epoch, row). CTC-infeasible
  rows (label tokens + repeats > 25 frames/s - 2) are dropped up front.
- Optimizer: Muon (hidden linear maps, per head / per split, momentum .95, scale .2*sqrt(max(h, w))) + AdamW
  (everything else, betas .9/.98, weight decay .001; memory value table at 5x LR, no decay), guarded against
  nonfinite/oversized updates. LR: 10-step linear warmup, cosine from 1 to 0.1 over --updates. Gradient clip 1.0.
- Fast path on (byte-exact), cuDNN/MIOpen off during training steps on ROCm (it is ~41 % slower there) and on in evals.
- Every --eval-every updates (and at the end): greedy WER and CER (balbal-normalizer scoring) on --dev, overall and per
  `source`; for each --soup share w also the weight soup w*current + (1-w)*init. The 5 best raw checkpoints by the best
  dev WER among raw and soups are kept in OUT/checkpoints (index: best.json). The original run additionally ranked
  checkpoints with the deployed KenLM decoder on its internal DEV set; that ranking is not part of this release.
- Logs: OUT/metrics.jsonl (+ stdout), a PROGRESS line (step, loss, updates/s, ETA) every 100 updates.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', type=Path)
    p.add_argument('--train', required=True); p.add_argument('--dev', default=None)
    p.add_argument('--tokenizer-dir', required=True); p.add_argument('--init', default=None)
    p.add_argument('--updates', type=int, default=9000)
    p.add_argument('--stop-at', type=int, default=None, help='stop early (schedule still spans --updates)')
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--seed', type=int, default=20261003)
    p.add_argument('--eval-every', type=int, default=3000); p.add_argument('--micro', type=int, default=16)
    p.add_argument('--soup', default='0.5', help='comma list of trained-weight shares w scored at each eval')
    p.add_argument('--tf32', action='store_true'); p.add_argument('--no-save', action='store_true')
    args = p.parse_args()
    if args.micro < 1 or 32 % args.micro: p.error('--micro must divide the 32-clip update')
    out = args.out; out.mkdir(parents=True, exist_ok=False)

    import numpy as np, torch
    from torch.utils.data import DataLoader
    from training import data as D, fastpath
    from training.model import build_model, loss as loss_fn, greedy_decode
    from training.sampling import EpochLengthSampler
    from balbal_model.conformer_extra_objectives import clear_extra
    from normalizer.metrics_v2 import corpus_metrics_v2

    def emit(event, **kw):
        row = dict(event=event, at=time.time(), **kw); print(json.dumps(row), flush=True)
        with (out / 'metrics.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')

    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = args.tf32; torch.backends.cudnn.benchmark = False
    TRAIN_CUDNN = torch.version.hip is None   # MIOpen is ~41% slower in training on ROCm; cuDNN on NVIDIA is fine
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    read = lambda path: [json.loads(l) for l in open(path)]
    train, dev = read(args.train), (read(args.dev) if args.dev else [])
    model = build_model(args.tokenizer_dir, args.init)
    fastpath.install(model)
    H, micro = args.updates, args.micro
    optimizer = fastpath.HybridAdamWMuon(model, lr=args.lr, memory_lr=float(f'{5*args.lr:.10g}'))
    shape = lambda s: .1+.9*(1+math.cos(math.pi*min(1., max(0., (s-10)/max(1, H-10)))))/2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s+1)/10) * shape(s))
    soups = [float(w) for w in args.soup.split(',') if w]
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()} if soups else None
    labels = [model.tokenizer.text_to_ids(r['text']) for r in train]
    keep = [i for i, r in enumerate(train) if D.ctc_frames_needed(labels[i]) <= int(r['duration_seconds'] * 25) - 2]
    infeasible = len(train) - len(keep); train, labels = [train[i] for i in keep], [labels[i] for i in keep]
    epoch_steps = math.ceil(len(train) / 32)
    (out / 'config.json').write_text(json.dumps(dict(vars(args), out=str(out), train_clips=len(train),
        train_hours=sum(r['duration_seconds'] for r in train)/3600, ctc_infeasible_dropped=infeasible,
        tel_p=D.TEL_P, edge_p=D.EDGE_P, edge_max=D.EDGE_MAX, loader_workers=int(os.getenv('LOADER_WORKERS', 4)),
        started=time.time()), indent=1, default=str))

    @torch.inference_mode()
    def greedy(rows_):
        torch.backends.cudnn.enabled = True; model.eval(); preds = []
        loader = DataLoader(D.Rows(rows_, [[] for _ in rows_]), batch_size=8, collate_fn=D.collate, num_workers=2)
        for _, wave, lengths, _, _ in loader:
            probs, encoded, _ = model(input_signal=wave.cuda(), input_signal_length=lengths.cuda())
            preds += greedy_decode(model, probs.argmax(-1), encoded)
            clear_extra(model)
        torch.backends.cudnn.enabled = TRAIN_CUDNN; model.train()
        return preds

    def measure():
        if not dev: return {}
        hyps = greedy(dev); groups = defaultdict(list)
        for r, h in zip(dev, hyps): groups[r.get('source', 'dev')].append((r['text'], h))
        score = lambda items: corpus_metrics_v2([a for a, _ in items], [b for _, b in items])
        m, by = score([(r['text'], h) for r, h in zip(dev, hyps)]), {g: score(items) for g, items in groups.items()}
        return dict(dev_v2=100*m['wer'], dev_by_source_v2={g: 100*x['wer'] for g, x in by.items()},
                    dev_cer_v2=100*m['cer'], dev_cer_by_source_v2={g: 100*x['cer'] for g, x in by.items()})

    best = []   # best-5 checkpoints by dev WER (best of raw and soups)
    def keep_best(step, state, scores):
        if args.no_save or step == 0 or not scores: return
        wer, label = min(scores); d = out / 'checkpoints'; d.mkdir(exist_ok=True)
        best.append(dict(step=step, dev_v2=wer, best_of=label, path=str(d / f'step{step}.pt')))
        best.sort(key=lambda b: (b['dev_v2'], b['step']))
        if any(b['step'] == step for b in best[:5]): torch.save(state, d / f'step{step}.pt')
        for b in best[5:]: Path(b['path']).unlink(missing_ok=True)
        del best[5:]; (d / 'best.json').write_text(json.dumps(best, indent=2) + '\n')

    def evaluate(step, train_seconds, loss_avg):
        t = time.monotonic(); m = measure()
        emit('EVAL', step=step, train_seconds=train_seconds, loss=loss_avg, eval_seconds=time.monotonic() - t, **m)
        scores = [(m['dev_v2'], 'raw')] if m else []
        if step == 0 or not soups or not dev: return keep_best(step, model.state_dict(), scores)   # at step 0 soups == init
        current = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for w in soups:   # same mix as soup.py; eval time is not training time
            t = time.monotonic()
            model.load_state_dict({k: (w*v.float() + (1-w)*init_state[k].float()).to(v.dtype) if v.is_floating_point() else v
                                   for k, v in current.items()}, strict=True)
            m = measure(); scores.append((m['dev_v2'], f'soup{w}'))
            emit('EVAL_SOUP', w=w, step=step, train_seconds=train_seconds, eval_seconds=time.monotonic() - t, **m)
        model.load_state_dict(current, strict=True); keep_best(step, current, scores); del current

    evaluate(0, 0., None)
    sampler = EpochLengthSampler([r['duration_seconds'] for r in train], 32, args.seed, shuffle_batches=True)
    torch.backends.cudnn.enabled = TRAIN_CUDNN; model.train(); optimizer.zero_grad(set_to_none=True)
    step, train_seconds, losses, stop_at = 0, 0., [], (H if args.stop_at is None else args.stop_at)
    recent, prog = [], [time.monotonic(), 0]
    while step < stop_at:
        t0 = time.monotonic()   # loader start-up counts as training time
        epoch = step // epoch_steps; sampler.set_epoch(epoch); indices = list(sampler)[(step % epoch_steps)*32:]
        workers = int(os.getenv('LOADER_WORKERS', 4))
        loader = DataLoader(D.Rows(train, labels, seed=(args.seed % 2**64, epoch)), batch_size=micro, sampler=indices,
                            collate_fn=D.collate, num_workers=workers, prefetch_factor=4 if workers else None,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        count, pending = 0, []
        for batch in loader:
            update_size = min(32, len(train) - (step % epoch_steps)*32)
            weight = len(batch[0]) / update_size
            loss = loss_fn(model, batch, step)
            (loss*weight).backward(); pending.append((loss.detach(), weight)); count += len(batch[0]); clear_extra(model)
            if count < update_size: continue
            values = torch.stack([l for l, _ in pending]).tolist()
            if not all(map(math.isfinite, values)): raise FloatingPointError('Nonfinite objective')
            losses.append(sum(v*w for v, (_, w) in zip(values, pending))); pending.clear(); count = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
            recent.append(losses[-1])
            if step % 100 == 0:   # step/loss/rate/ETA every 100 updates (wall clock, evals included)
                now = time.monotonic(); rate = (step - prog[1]) / max(now - prog[0], 1e-9); prog[:] = [now, step]
                emit('PROGRESS', step=step, of=stop_at, loss=sum(recent) / len(recent), updates_per_s=rate,
                     eta_minutes=(stop_at - step) / max(rate, 1e-9) / 60); recent.clear()
            if step % args.eval_every == 0 or step == stop_at:
                torch.cuda.synchronize(); train_seconds += time.monotonic() - t0
                evaluate(step, train_seconds, sum(losses) / len(losses)); losses.clear()
                t0 = time.monotonic()
            if step >= stop_at: break
        del loader
        torch.cuda.synchronize(); train_seconds += time.monotonic() - t0
    if not args.no_save: torch.save(model.state_dict(), out / 'final.pt')
    emit('DONE', step=step, train_seconds=train_seconds)


if __name__ == '__main__': main()
