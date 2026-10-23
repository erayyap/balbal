"""balbal-v1 model construction and training objective.

Model: NeMo EncDecCTCModelBPE from configs/model_config.yaml (Conformer-CTC, 16 layers, d_model 176, SentencePiece 1024
+ blank), plus the loop-refinement head with product-key acoustic memory from configs/refiner.json (2-layer refiner run
for 3 passes; 500x500 sub-keys, 4 heads, top-16). 45.4M stored parameters (the memory value table is most of them),
15.4M active per frame.

Objective (one update = 32 clips; microbatches are weighted by their share of the update):
  total = 0.3 * CTC(base encoder output) + 0.7 * mean over the 3 refiner passes of CTC(pass output)
bf16 autocast on CUDA; CTC with zero_infinity off (infeasible clips are filtered out before training).
"""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs'
SUPERVISED_BASE_WEIGHT = 0.3
# The recipe's earlier stages ran 234441 updates; the memory strength ramp (0 -> 1 over steps 128..512) is long done.
STEP_OFFSET = 234441


def build_model(tokenizer_dir, init=None, device='cuda'):
    """tokenizer_dir: SentencePiece dir (tokenizer.model, tokenizer.vocab, vocab.txt). init: state dict path or None."""
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import EncDecCTCModelBPE
    from nemo.core.classes.mixins import AccessMixin
    from nemo.utils import logging
    from balbal_model.conformer_loop_memory import attach_loop_memory
    logging.set_verbosity(logging.ERROR)
    cfg = OmegaConf.load(CONFIGS / 'model_config.yaml')
    for key, value in list(cfg.tokenizer.items()):
        if isinstance(value, str) and (value == 'tokenizer' or value.startswith('tokenizer/')):
            cfg.tokenizer[key] = str(Path(tokenizer_dir) / value.removeprefix('tokenizer').lstrip('/'))
    model = EncDecCTCModelBPE(cfg=cfg).float()
    model.loss.zero_infinity = False
    for layer in model.encoder.layers:   # NeMo's own attention kernel, not PyTorch SDPA
        layer.self_attn.use_pytorch_sdpa = False
    refiner = json.loads((CONFIGS / 'refiner.json').read_text())
    attach_loop_memory(model, refiner['config'], memory_config=refiner['memory'])
    if init:
        model.load_state_dict(torch.load(init, map_location='cpu', weights_only=True), strict=True)
    model = model.float()
    AccessMixin.set_access_enabled(False, guid=model.model_guid)
    model.ctc_extra.inference_loops = refiner['inference_loops']
    assert model.decoder.num_classes_with_blank == 1025 and all(p.requires_grad for p in model.parameters())
    return model.to(device)


def loss(model, batch, step):
    """Training loss of one microbatch (the model must be in train mode)."""
    extra = model.ctc_extra
    extra.set_step(STEP_OFFSET + step)
    _, wave, lengths, labels, target_lengths = batch
    device = next(model.parameters()).device
    wave, lengths, labels, target_lengths = [x.to(device, non_blocking=True)
                                            for x in (wave, lengths, labels, target_lengths)]
    extra.loops = 3
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        features, feature_lengths = model.preprocessor(input_signal=wave, length=lengths)
        _, encoded, _ = model(processed_signal=features, processed_signal_length=feature_lengths)
    outputs, ctc_loss = [extra.last_base, *extra.last_passes], model.loss
    probs = [p.float().log_softmax(-1) for p in outputs]
    supervised = [ctc_loss(log_probs=p, targets=labels, input_lengths=encoded, target_lengths=target_lengths) for p in probs]
    return SUPERVISED_BASE_WEIGHT * supervised[0] + (1 - SUPERVISED_BASE_WEIGHT) * torch.stack(supervised[1:]).mean()


def greedy_decode(model, predictions, lengths):
    # CTC order matters: collapse adjacent duplicates BEFORE removing blank.
    blank = model.decoder.num_classes_with_blank-1
    result = []
    for ids, length in zip(predictions.cpu(), lengths.cpu()):
        ids = torch.unique_consecutive(ids[:int(length)]).tolist()
        result.append(model.tokenizer.ids_to_text([i for i in ids if i != blank]))
    return result
