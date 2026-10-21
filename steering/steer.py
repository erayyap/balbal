"""Steering of balbal-v1's acoustic-memory reads toward a target sentence.
Per refinement pass an additive offset on the memory read is optimised with Adam for `steps` steps on CTC(target), only on
low-confidence frames (max prob < gate, widened by 2 frames each side); a target much shorter than balbal's own greedy text
(< guard x its words) is refused. Everything else in the model stays frozen; the offset is discarded after each clip.
  s = Steerer(transcriber.model); s.run(wave, target, steps=3, lr=.03, gate=.9, guard=.5)
  -> dict(log_probs (T, 1025) numpy, plain_greedy, steered_greedy, guarded, steered_frames, ctc_before, ctc_after)"""
import numpy as np
import torch
from torch.nn import functional as F


class Steerer:
    def __init__(self, model):
        self.model = model; ex = model.ctc_extra; self.ex, self.mem = ex, ex.acoustic_memory
        self.dev = next(model.parameters()).device
        self.saved, self.ctl = {}, dict(delta=None, mask=None)
        self.blank = model.decoder.num_classes_with_blank - 1
        plain_refine, plain_retrieve = ex.forward, self.mem.retrieve
        self.plain_refine = plain_refine
        def capture(base):
            self.saved.update(encoded=ex.last_encoded, lengths=ex.last_lengths, base=base); return plain_refine(base)
        def retrieve(query):
            memory, indices, weights = plain_retrieve(query)
            d = self.ctl['delta']
            if d is not None:
                T = memory.shape[1]; memory = memory + d[self.mem.pass_index][:, :T] * self.ctl['mask'][:, :T, None]
            return memory, indices, weights
        ex.forward = capture; self.mem.retrieve = retrieve   # no offset set -> identical to the plain model

    def greedy(self, log_probs, length):
        ids = torch.unique_consecutive(log_probs.argmax(-1)[:int(length)]).tolist()
        return self.model.tokenizer.ids_to_text([i for i in ids if i != self.blank])

    def refine(self):
        ex = self.ex; ex.last_encoded, ex.last_lengths = self.saved['encoded'], self.saved['lengths']
        out = self.plain_refine(self.saved['base']); ex.clear(); return out

    def ctc(self, lp, target, T):
        ids = self.model.tokenizer.text_to_ids(target)
        return F.ctc_loss(lp.transpose(0, 1), torch.tensor([ids], dtype=torch.long, device=lp.device), T.cpu(), torch.tensor([len(ids)]),
                          blank=lp.shape[-1] - 1, reduction='sum', zero_infinity=True)

    def run(self, wave, target, *, steps=3, lr=.03, gate=.9, guard=.5):
        x = torch.from_numpy(np.asarray(wave, dtype=np.float32))[None].to(self.dev)
        L = torch.tensor([x.shape[1]], device=self.dev); self.ctl['delta'] = None
        with torch.no_grad():
            p0, enc, *_ = self.model(input_signal=x, input_signal_length=L); self.ex.clear()
        T = enc.long(); Tf = p0.shape[1]
        low = (p0.float().exp().max(-1).values < gate).float()[:, None]
        mask = (F.max_pool1d(low, 5, 1, 2)[:, 0] > 0).float()
        mask[:, int(T[0]):] = 0
        plain = self.greedy(p0[0], T[0])
        guarded = (not target.strip()) or len(target.split()) < guard * len(plain.split())
        if guarded: mask.zero_()
        self.ctl['mask'] = mask; l0 = l1 = None
        if not guarded and float(mask.sum()) > 0:
            self.ctl['delta'] = [torch.zeros(1, Tf, self.mem.value_dim, device=self.dev, requires_grad=True) for _ in range(self.ex.inference_loops)]
            opt = torch.optim.Adam(self.ctl['delta'], lr=lr)
            with torch.enable_grad():
                for s in range(steps):
                    loss = self.ctc(self.refine().float().log_softmax(-1), target, T)
                    if s == 0: l0 = float(loss)
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): p1 = self.refine()
            self.ctl['delta'] = None
            with torch.no_grad(): l1 = float(self.ctc(p1.float().log_softmax(-1), target, T))
        else:
            p1 = p0
        return dict(log_probs=p1[0, :int(T[0])].float().cpu().numpy(), plain_log_probs=p0[0, :int(T[0])].float().cpu().numpy(),
                    plain_greedy=plain, steered_greedy=self.greedy(p1[0], T[0]), guarded=bool(guarded),
                    steered_frames=float(mask.sum() / T[0]), ctc_before=l0, ctc_after=l1)
