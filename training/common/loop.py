"""Training loop shared by training/balbal_v2/train.py and training/manas/train.py (the recipes set the defaults).

Objective per micro-batch: 0.3 x CTC(base encoder output) + 0.7 x mean over the 3 refiner passes of CTC(pass output)
  + lang_w x frame-level language cross-entropy (tr / en / non-speech; unlabelled frames ignored). bf16 autocast on CUDA, losses in fp32.
Updates: BATCH clips (micro x accumulation x world), length-grouped batches (EpochLengthSampler), batch order shuffled per epoch; only
  full updates (the < BATCH-clip tail of an epoch is skipped). CTC-infeasible rows (tokens + repeats > frames - 2) are dropped up front.
Optimizer: Muon (hidden linear maps) + AdamW (the rest; memory value table and learned memory temperature at mem_lr_mult x LR, no
  decay), guarded against nonfinite / oversized updates; gradient clip 1.0. --nonfinite-skip N skips up to N updates whose gradient norm
  is not finite (logged), then raises.
LR: linear warmup over --warmup updates, flat, then linear decay over the last --decay share of the run to --floor x LR.
Evaluation every --eval-epochs (and at the end): greedy WER / CER (balbal-normalizer scoring) on --dev, overall and per `source`; with
  --soup w also the weight soup w x current + (1 - w) x init. The --keep best raw checkpoints by the best dev WER (raw or soup) stay in
  OUT/checkpoints (index: best.json).
Multi-GPU: torchrun --nproc_per_node N (NCCL; gloo on CPU). Each rank takes every world-th micro-batch of the same updates; gradients are
  summed in ~64 MB buckets (GRAD_COMM=bf16: bf16 traffic). Rank 0 evaluates, saves and logs.
Resume: --resume continues OUT from OUT/state.pt (model, optimizer, step, best list), written at every eval and every --state-every
  updates; the world size may change (the update order depends only on the step).
Logs: OUT/metrics.jsonl (+ stdout); a PROGRESS line (step, epoch, loss, language CE, LR, updates/s, audio hours per hour, ETA) every
  --progress-every updates.
"""
import argparse
import datetime
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]


def arguments(description, defaults):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', type=Path, help='run directory')
    p.add_argument('--train', required=True, help='comma-separated train manifests')
    p.add_argument('--dev', default=None, help='dev manifest (greedy WER / CER at every eval)')
    p.add_argument('--drop', default='', help='comma-separated id lists to leave out')
    p.add_argument('--relabel', default='', help='comma-separated {"id", "text"} JSON lines replacing texts')
    p.add_argument('--tokenizer-dir', required=True, help='tokenizer.model, tokenizer.vocab, vocab.txt')
    p.add_argument('--init', default=None, help='weights to start from')
    p.add_argument('--epochs', type=float, default=defaults['epochs'])
    p.add_argument('--updates', type=int, default=None, help='total updates (overrides --epochs)')
    p.add_argument('--stop-at', type=int, default=None, help='stop early (the schedule still spans the whole run)')
    p.add_argument('--batch', type=int, default=defaults['batch'], help='clips per update')
    p.add_argument('--micro', type=int, default=defaults['micro'], help='clips per forward / backward')
    p.add_argument('--lr', type=float, default=defaults['lr'])
    p.add_argument('--warmup', type=int, default=defaults['warmup'])
    p.add_argument('--decay', type=float, default=defaults['decay'], help='share of the run with the linear LR decay')
    p.add_argument('--floor', type=float, default=.1, help='final LR / peak LR')
    p.add_argument('--weight-decay', type=float, default=defaults['weight_decay'])
    p.add_argument('--mem-lr-mult', type=float, default=5., help='LR multiplier of the memory value table (and learned temperature)')
    p.add_argument('--muon-bf16', action='store_true', default=defaults.get('muon_bf16', False), help='Newton-Schulz matmuls in bf16')
    p.add_argument('--lang-w', type=float, default=.1, help='weight of the frame language cross-entropy (0 = off)')
    p.add_argument('--lang-band', type=int, default=5, help='unscored frames on both sides of a language change')
    p.add_argument('--default-lang', default='tr', choices=('tr', 'en'), help='language of rows without "lang"')
    p.add_argument('--cond', action='store_true', help='train the utterance-level language condition (lang_cond)')
    p.add_argument('--cond-p', type=float, default=.7, help='probability of the true language (else auto) for monolingual clips')
    p.add_argument('--step-offset', type=int, default=defaults['step_offset'],
                   help='training step of the first update: the memory read strength ramps in over steps 128-512 (0 = new memory)')
    p.add_argument('--split-padded-seconds', type=float, default=200., help='micro-batches above this padded length run as halves')
    p.add_argument('--eval-epochs', type=float, default=.25)
    p.add_argument('--soup', default='0.5', help='comma list of trained-weight shares w scored at each eval ("" = none)')
    p.add_argument('--keep', type=int, default=5, help='best raw checkpoints kept')
    p.add_argument('--nonfinite-skip', type=int, default=20)
    p.add_argument('--state-every', type=int, default=1000)
    p.add_argument('--progress-every', type=int, default=100)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=defaults['seed'])
    p.add_argument('--threads', type=int, default=4, help='torch CPU threads')
    p.add_argument('--tf32', action='store_true')
    p.add_argument('--no-save', action='store_true')
    return p


def allreduce_grads(params, bucket=64 * 2 ** 20):
    """SUM-all-reduce the gradients of `params` in ~64 MB buckets (missing grads = zeros)."""
    import torch
    import torch.distributed as dist
    grads = [q.grad if q.grad is not None else torch.zeros_like(q) for q in params]
    for q, g in zip(params, grads): q.grad = g
    group, size = [], 0
    for g in grads + [None]:
        if g is not None: group.append(g); size += g.numel()
        if group and (g is None or size >= bucket):
            flat = torch.cat([x.reshape(-1) for x in group])
            if os.getenv('GRAD_COMM', 'fp32') == 'bf16': send = flat.bfloat16(); dist.all_reduce(send); flat.copy_(send)
            else: dist.all_reduce(flat)
            off = 0
            for x in group: x.copy_(flat[off:off + x.numel()].view_as(x)); off += x.numel()
            group, size = [], 0


def polar_bf16(momentum):
    """muon_safe.polar_update with the Newton-Schulz matmuls in bf16 (same normalization, 8 iterations and guard)."""
    import torch
    with torch.autocast(momentum.device.type, enabled=False):
        x = momentum.float(); x = x / x.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        x = x / torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp_min(1e-7)
        t = x.shape[-2] > x.shape[-1]; x = (x.mT if t else x).bfloat16()
        for _ in range(8):
            g = x @ x.mT; x = 3.4445 * x + (-4.7750 * g + 2.0315 * (g @ g)) @ x
        x = x.float(); x = x.mT if t else x
    return x.to(momentum.dtype)


def greedy_text(model, log_probs, lengths):
    blank = log_probs.shape[-1] - 1; out = []
    for ids, k in zip(log_probs.argmax(-1).tolist(), lengths.tolist()):
        toks, prev = [], None
        for t in ids[:k]:
            if t != prev and t != blank: toks.append(t)
            prev = t
        out.append(model.tokenizer.ids_to_text(toks) if toks else '')
    return out


def run(args, name, model, head_prefix, separate_head, eval_bf16):
    """Train `model` (built and initialised by the recipe, language head attached) on args."""
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from balbal.model.conformer_extra_objectives import clear_extra
    from balbal.models import language_head
    from balbal.normalizer.metrics_v2 import corpus_metrics_v2
    from training.common import data as D, fastpath, muon_safe
    from training.common.sampling import EpochLengthSampler

    world, rank, local_rank = int(os.getenv('WORLD_SIZE', 1)), int(os.getenv('RANK', 0)), int(os.getenv('LOCAL_RANK', 0))
    cuda = torch.cuda.is_available()
    if world > 1:
        if cuda: torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl' if cuda else 'gloo', timeout=datetime.timedelta(hours=2))
    device = torch.device(f'cuda:{local_rank}' if cuda else 'cpu')
    out = args.out
    if rank == 0: (out / 'checkpoints').mkdir(parents=True, exist_ok=args.resume)
    if world > 1: dist.barrier()
    def emit(event, **kw):
        if rank: return
        row = dict(event=event, at=round(time.time(), 1), **kw); print(json.dumps(row, ensure_ascii=False), flush=True)
        with (out / 'metrics.jsonl').open('a') as f: f.write(json.dumps(row, ensure_ascii=False) + '\n')

    torch.set_num_threads(args.threads); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = args.tf32; torch.backends.cudnn.benchmark = False
    train_cudnn = torch.version.hip is None   # MIOpen is much slower than the native kernels in training on ROCm
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.muon_bf16: muon_safe.polar_update = polar_bf16
    memory = getattr(model.ctc_extra, 'acoustic_memory', None)
    if memory is not None: fastpath.install(model)
    model = model.to(device).train()
    if world > 1:   # NeMo all-reduces the max sequence length in every forward under torch.distributed: rank 0's solo eval would hang
        model.encoder.set_max_audio_length(6000); model.encoder.sync_max_audio_length = False
        with torch.no_grad():
            for t in list(model.parameters()) + list(model.buffers()): dist.broadcast(t, 0)
    sub = model.encoder.subsampling_factor; hop = 160 * sub

    # data
    split = lambda s: [x for x in s.split(',') if x]
    rows, counts = D.load_manifests(split(args.train), split(args.drop), split(args.relabel))
    labels, keep = [], []
    for r in rows:
        t = model.tokenizer.text_to_ids(r['text']) if r['text'].strip() else []
        if D.ctc_frames_needed(t) > int(r['duration_seconds'] * 16000) // hop - 2: counts['ctc_infeasible'] = counts.get('ctc_infeasible', 0) + 1; continue
        keep.append(r); labels.append(t)
    rows = keep; dev = [json.loads(l) for l in open(args.dev)] if args.dev else []
    batch, micro = args.batch, args.micro
    accum = batch // (micro * world)
    if accum < 1 or accum * micro * world != batch: raise SystemExit('--micro x world size must divide --batch')
    epoch_steps = len(rows) // batch
    if not epoch_steps: raise SystemExit(f'fewer train rows ({len(rows)}) than --batch ({batch})')
    total = args.updates or int(round(args.epochs * epoch_steps)); stop_at = total if args.stop_at is None else args.stop_at
    eval_every = max(1, int(round(args.eval_epochs * epoch_steps)))

    # optimizer + schedule
    optimizer = fastpath.HybridAdamWMuon(model, lr=args.lr, memory_lr=args.lr * args.mem_lr_mult, weight_decay=args.weight_decay)
    lt = getattr(memory, 'log_temp', None)
    if lt is not None:   # the learned temperature joins the memory group (memory LR, no decay)
        g0, g1 = optimizer.param_groups[:2]; j = next(j for j, q in enumerate(g0['params']) if q is lt)
        for key in ('params', 'param_names', 'splits'): g1[key].append(g0[key].pop(j))
    mults = [1., args.mem_lr_mult, 1.]
    lr_at = lambda s: args.lr * min(1., (s + 1) / args.warmup) * (args.floor + (1 - args.floor) * min(1., (total - s) / max(1, args.decay * total)))
    soups = [float(w) for w in split(args.soup)]
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()} if soups else None
    head = language_head(model)
    if rank == 0 and not args.resume:
        (out / 'config.json').write_text(json.dumps(dict(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, model=name, world=world, subsampling=sub,
            train_rows=len(rows), train_hours=round(sum(r['duration_seconds'] for r in rows) / 3600, 2), counts=counts,
            updates=total, updates_per_epoch=epoch_steps, eval_every=eval_every, accumulation=accum,
            parameters=sum(q.numel() for q in model.parameters()), grad_comm=os.getenv('GRAD_COMM', 'fp32')), indent=1))

    # evaluation
    @torch.inference_mode()
    def measure():
        if not dev: return {}
        model.eval(); hyps = []
        if hasattr(model, 'lang_cond'): model._cond_ids = None   # evaluate in auto mode
        torch.backends.cudnn.enabled = True
        for i in range(0, len(dev), 8):
            ws = [torch.from_numpy(D.materialise(r, args.default_lang)[0]) for r in dev[i:i + 8]]
            w = torch.nn.utils.rnn.pad_sequence(ws, batch_first=True).to(device); n = torch.tensor([len(x) for x in ws], device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=eval_bf16 and device.type == 'cuda'):
                lp, en, _ = model(input_signal=w, input_signal_length=n)
            hyps += greedy_text(model, lp.float().cpu(), en.cpu()); clear_extra(model)
        torch.backends.cudnn.enabled = train_cudnn; model.train()
        groups = defaultdict(list)
        for r, h in zip(dev, hyps): groups[r.get('source', 'dev')].append((r['text'], h))
        score = lambda items: corpus_metrics_v2([a for a, _ in items], [b for _, b in items])
        m, by = score([(r['text'], h) for r, h in zip(dev, hyps)]), {g: score(v) for g, v in sorted(groups.items())}
        return dict(dev_wer=round(100 * m['wer'], 3), dev_cer=round(100 * m['cer'], 3),
                    by_source={g: dict(wer=round(100 * x['wer'], 3), cer=round(100 * x['cer'], 3)) for g, x in by.items()})

    def save_weights(state, path):
        """Model weights; with separate_head the language head goes to *.lang.pt (the weights stay loadable by build_model)."""
        if separate_head:
            torch.save({k: v for k, v in state.items() if not k.startswith(head_prefix)}, path)
            if head is not None: torch.save({k: v for k, v in state.items() if k.startswith(head_prefix)}, str(path)[:-3] + '.lang.pt')
        else:
            torch.save(state, path)

    best = []
    def keep_best(step, state, scores):
        if args.no_save or step == 0 or not scores: return
        wer, label = min(scores); d = out / 'checkpoints'
        best.append(dict(step=step, dev_wer=wer, best_of=label, path=str(d / f'step{step}.pt')))
        best.sort(key=lambda b: (b['dev_wer'], b['step']))
        if any(b['step'] == step for b in best[:args.keep]): save_weights(state, d / f'step{step}.pt')
        for b in best[args.keep:]: Path(b['path']).unlink(missing_ok=True); Path(b['path'][:-3] + '.lang.pt').unlink(missing_ok=True)
        del best[args.keep:]; (d / 'best.json').write_text(json.dumps(best, indent=1) + '\n')

    def evaluate(step, train_seconds, loss_avg):
        t = time.monotonic(); m = measure()
        emit('EVAL', step=step, epoch=round(step / epoch_steps, 3), train_seconds=round(train_seconds, 1), loss=loss_avg,
             eval_seconds=round(time.monotonic() - t, 1), **m)
        scores = [(m['dev_wer'], 'raw')] if m else []
        if step == 0 or not soups or not dev: return keep_best(step, model.state_dict(), scores)
        current = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for w in soups:
            t = time.monotonic()
            model.load_state_dict({k: (w * v.float() + (1 - w) * init_state[k].float()).to(v.dtype) if v.is_floating_point() else v
                                   for k, v in current.items()}, strict=True)
            m = measure(); scores.append((m['dev_wer'], f'soup{w}'))
            emit('EVAL_SOUP', w=w, step=step, epoch=round(step / epoch_steps, 3), eval_seconds=round(time.monotonic() - t, 1), **m)
        model.load_state_dict(current, strict=True); keep_best(step, current, scores); del current

    def save_state(step, train_seconds):
        if rank or args.no_save: return
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, train_seconds=train_seconds, best=best,
                        total=total, batch=batch, rows=len(rows), rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                                                        cuda=torch.cuda.get_rng_state_all() if cuda else None)),
                   out / 'state.pt.tmp')
        (out / 'state.pt.tmp').replace(out / 'state.pt')

    def sync():
        if world > 1: dist.barrier()

    # objective
    lang_parts = defaultdict(float)
    def loss_fn(b, step):
        _, wave, lengths, labels_, target_lengths, lang, cond = b
        wave, lengths, labels_, target_lengths, lang, cond = [x.to(device, non_blocking=True) for x in (wave, lengths, labels_, target_lengths, lang, cond)]
        extra = model.ctc_extra; extra.set_step(args.step_offset + step)
        if hasattr(model, 'lang_cond'): model._cond_ids = cond
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            _, encoded, _ = model(input_signal=wave, input_signal_length=lengths)
            logits = head(model._last_encoded.transpose(1, 2)) if head is not None and args.lang_w else None
        probs = [p.float().log_softmax(-1) for p in [extra.last_base, *extra.last_passes]]
        ctc = [model.loss(log_probs=p, targets=labels_, input_lengths=encoded, target_lengths=target_lengths) for p in probs]
        total_ = .3 * ctc[0] + .7 * torch.stack(ctc[1:]).mean()
        if logits is None: return total_
        T = logits.shape[1]; target = torch.full((len(lang), T), -100, dtype=torch.long, device=device)
        n = min(T, lang.shape[1]); target[:, :n] = lang[:, :n]
        target[torch.arange(T, device=device)[None] >= encoded[:, None]] = -100
        valid = target >= 0
        ce = F.cross_entropy(logits.float()[valid], target[valid]) if bool(valid.any()) else logits.float().sum() * 0.
        lang_parts['ce'] += ce.detach(); lang_parts['n'] += 1
        return total_ + args.lang_w * ce

    # resume
    step, train_seconds = 0, 0.
    if args.resume:
        st = torch.load(out / 'state.pt', map_location='cpu', weights_only=False)
        if st['batch'] != batch or st['rows'] != len(rows): raise SystemExit('resume: --batch or the train rows changed')
        model.load_state_dict(st['model'], strict=True); optimizer.load_state_dict(st['optimizer'])
        step, train_seconds, total = st['step'], st['train_seconds'], st['total']; best[:] = st['best']
        random.setstate(st['rng']['python']); np.random.set_state(st['rng']['numpy']); torch.set_rng_state(st['rng']['torch'])
        if cuda and st['rng']['cuda'] is not None and len(st['rng']['cuda']) == torch.cuda.device_count(): torch.cuda.set_rng_state_all(st['rng']['cuda'])
        stop_at = total if args.stop_at is None else args.stop_at
        emit('RESUME', step=step, of=stop_at, world=world); del st
    else:
        if rank == 0: evaluate(0, 0., None)
        sync()
    if rank and not args.resume: torch.manual_seed(args.seed + 1_000_003 * rank)   # own dropout / SpecAugment masks per rank
    elif rank: torch.manual_seed(args.seed + step + 1_000_003 * rank)

    sampler = EpochLengthSampler([r['duration_seconds'] for r in rows], batch, args.seed, shuffle_batches=True)
    torch.backends.cudnn.enabled = train_cudnn; optimizer.zero_grad(set_to_none=True)
    trainable = [q for q in model.parameters() if q.requires_grad]
    losses, recent, lang_recent, skipped = [], [], [], 0
    prog, audio = [time.monotonic(), step], torch.zeros((), dtype=torch.float64)
    while step < stop_at:
        t0 = time.monotonic()
        epoch = step // epoch_steps; sampler.set_epoch(epoch)
        order = list(sampler)[:epoch_steps * batch][(step % epoch_steps) * batch:]   # full updates only
        workers = D.loader_workers()
        loader = DataLoader(D.Rows(rows, labels, hop, args.lang_band, args.cond_p, (args.seed, epoch), args.default_lang),
                            batch_size=micro, sampler=D.RankSlice(order, micro, rank, world), collate_fn=D.collate,
                            num_workers=workers, prefetch_factor=4 if workers else None)
        it = iter(loader)
        while step < stop_at:
            batches = []
            for _ in range(accum):
                try: batches.append(next(it))
                except StopIteration: break
            have = torch.tensor([len(batches)], device=device)
            if world > 1: dist.all_reduce(have, op=dist.ReduceOp.MIN)   # every rank must join each all-reduce
            if int(have) < accum: break
            for g, m in zip(optimizer.param_groups, mults): g['lr'] = lr_at(step) * m
            pending = []
            for b0 in batches:
                for b in D.split_long(b0, args.split_padded_seconds):
                    loss = loss_fn(b, step); (loss * len(b[0]) / batch).backward()
                    pending.append(loss.detach() * len(b[0]) / batch); audio += float(b[2].sum()) / 16000
                    clear_extra(model)
            del batches
            value = torch.stack(pending).sum()
            if world > 1: dist.all_reduce(value); allreduce_grads(trainable)
            value = float(value)
            if not math.isfinite(value): raise FloatingPointError('Nonfinite objective')
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            if lang_parts: lang_recent.append(float(lang_parts['ce'] / lang_parts['n'])); lang_parts.clear()
            if not math.isfinite(norm):   # same norm on every rank (after the all-reduce)
                skipped += 1; emit('SKIP', step=step, norm=str(norm), skipped=skipped)
                if skipped > args.nonfinite_skip: raise FloatingPointError('too many non-finite gradient norms')
                optimizer.zero_grad(set_to_none=True); step += 1; continue   # the update consumed its batches
            optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
            losses.append(value); recent.append(value)
            if step % args.progress_every == 0:
                if cuda: torch.cuda.synchronize()
                total_audio = audio.clone()
                if world > 1: total_audio = total_audio.to(device); dist.all_reduce(total_audio)
                now = time.monotonic(); rate = (step - prog[1]) / max(now - prog[0], 1e-9)
                emit('PROGRESS', step=step, of=stop_at, epoch=round(step / epoch_steps, 3), loss=sum(recent) / max(1, len(recent)),
                     **({'lang_ce': sum(lang_recent) / len(lang_recent)} if lang_recent else {}), lr=lr_at(step), updates_per_s=round(rate, 3),
                     audio_h_per_h=round(float(total_audio) / max(now - prog[0], 1e-9), 1), eta_minutes=round((stop_at - step) / max(rate, 1e-9) / 60, 1),
                     train_seconds=round(train_seconds + now - t0, 1))
                prog[:] = [now, step]; recent.clear(); lang_recent.clear(); audio.zero_()
            if step % eval_every == 0 or step == stop_at:
                if cuda: torch.cuda.synchronize()
                train_seconds += time.monotonic() - t0
                if rank == 0: evaluate(step, train_seconds, sum(losses) / max(1, len(losses))); save_state(step, train_seconds)
                losses.clear(); sync(); t0 = time.monotonic()
            elif step % args.state_every == 0:
                save_state(step, train_seconds + time.monotonic() - t0); sync()
        del it, loader
        if cuda: torch.cuda.synchronize()
        train_seconds += time.monotonic() - t0
    if rank == 0 and not args.no_save: save_weights(model.state_dict(), out / 'final.pt')
    emit('DONE', step=step, train_seconds=round(train_seconds, 1))
    if world > 1:
        try: dist.barrier(); dist.destroy_process_group()
        except Exception as e: print('ddp teardown error (training finished):', type(e).__name__, flush=True)
