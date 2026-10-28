"""The two balbal-v2 models and how their PyTorch weights load.

  balbal-v2        Conformer-CTC (16 layers, d_model 176, 4x subsampling) + 3-pass loop refiner + product-key acoustic memory
                   (500x500 sub-keys, 4 heads, top-16). SentencePiece 1024 + blank. Same architecture as balbal-v1.
  balbal-v2-manas  FastConformer-CTC (18 layers, d_model 512, depthwise-striding stem run at 4x) + 3-pass loop refiner + product-key
                   acoustic memory (1000x1000 sub-keys, 8 heads, top-32, learned per-head read temperature) + frame-level language head
                   (tr / en / non-speech). SentencePiece 2048 (Turkish + English) + blank.

configs/<name>/model_config.yaml (NeMo), refiner.json (refiner + memory), decoder.json (KenLM beam search).
A tokenizer directory holds tokenizer.model, tokenizer.vocab and vocab.txt.

Optional training-time modules (not part of the released inference graph):
  language head  balbal-v2: model.decoder.lang_head, saved next to the weights as *.lang.pt; balbal-v2-manas: model.lang_head, in the weights
  condition      model.lang_cond: utterance-level language condition (tr 0 / en 1 / auto 2), zero-initialised, added (x scale) to the
                 encoder's subsampling output; model._cond_ids (B,) selects it, None = auto
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs'
MODELS = ('balbal-v2', 'balbal-v2-manas')
LANGUAGES = ('tr', 'en', 'non-speech')
COND_IDS = {'tr': 0, 'en': 1, 'auto': 2}
LANG_HEAD = {'balbal-v2': 'decoder.lang_head.', 'balbal-v2-manas': 'lang_head.'}


def check_name(name):
    if name not in MODELS:
        raise ValueError(f'unknown model {name!r} (one of {", ".join(MODELS)})')
    return name


def refiner_config(name):
    return json.loads((CONFIGS / check_name(name) / 'refiner.json').read_text())


def decoder_config(name):
    return json.loads((CONFIGS / check_name(name) / 'decoder.json').read_text())


def set_subsampling(model, factor):
    """4x <-> 8x on a depthwise-striding stem without new weights: the third stride-2 conv (conv[5]) gets time stride 1 (frequency
    stride stays 2, so the output projection is unchanged). Lengths follow the conv strides (MaskedConvSequential)."""
    pre = model.encoder.pre_encode
    assert pre._subsampling == 'dw_striding' and factor in (4, 8)
    pre.conv[5].stride = (2 if factor == 8 else 1, 2); pre._sampling_num = 3 if factor == 8 else 2
    pre.subsampling_factor = model.encoder.subsampling_factor = factor


def learn_temperature(memory):
    """One learned log-temperature per memory head (start = the configured temperature); read weights = softmax(score / exp(log_temp))."""
    import torch
    memory.log_temp = torch.nn.Parameter(torch.full((memory.heads, 1), math.log(memory.temperature)))
    memory.register_forward_pre_hook(lambda m, _: setattr(m, 'temperature', m.log_temp.exp()))


def attach_language_head(model, name):
    """Frame-level language head (tr / en / non-speech) on the encoder output; model._last_encoded holds that output after a forward."""
    import torch
    d = model.encoder.d_model
    head = torch.nn.Linear(d, len(LANGUAGES)).to(next(model.parameters()).device)
    if name == 'balbal-v2': model.decoder.lang_head = head
    else: model.lang_head = head
    def capture(_m, _a, output): model._last_encoded = output[0]
    model.encoder.register_forward_hook(capture)
    return head


def language_head(model):
    return model.lang_head if hasattr(model, 'lang_head') else getattr(model.decoder, 'lang_head', None)


def attach_condition(model, scale=10.):
    """Utterance-level language condition: Embedding(3, d_model), zero-init (an exact no-op at the start), times `scale` on the
    subsampling output. model._cond_ids (B,) long; None = auto for every clip (evaluation, plain inference)."""
    import torch
    model.lang_cond = torch.nn.Embedding(len(COND_IDS), model.encoder.d_model).to(next(model.parameters()).device)
    torch.nn.init.zeros_(model.lang_cond.weight); model._cond_ids = None
    def add(_m, _a, output):
        x, n = output
        ids = model._cond_ids if model._cond_ids is not None else torch.full((x.shape[0],), COND_IDS['auto'], dtype=torch.long, device=x.device)
        return x + scale * model.lang_cond(ids.to(x.device))[:, None, :].to(x.dtype), n
    model.encoder.pre_encode.register_forward_hook(add)


def load_state(path):
    """A state dict from a weights file, a training state (state.pt) or a DDP checkpoint ('module.' prefixes removed)."""
    import torch
    s = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(s, dict) and 'model' in s and not hasattr(s['model'], 'shape'): s = s['model']
    return {k.removeprefix('module.'): v for k, v in s.items()}


def split_state(state, name):
    """(acoustic model weights, language head weights, condition weights) of a state dict; head / condition are {} when absent.
    balbal-v2-manas keeps its language head among the acoustic model weights (it is built in)."""
    head_prefix = LANG_HEAD[check_name(name)]
    cond = {k.split('.', 1)[1]: v for k, v in state.items() if k.startswith('lang_cond.')}
    head = {k[len(head_prefix):]: v for k, v in state.items() if k.startswith(head_prefix)}
    keep = {k: v for k, v in state.items() if not k.startswith('lang_cond.') and (name == 'balbal-v2-manas' or not k.startswith(head_prefix))}
    return keep, head, cond


def build_model(name, tokenizer_dir, weights=None, *, device='cpu', loops=None):
    """NeMo EncDecCTCModelBPE from configs/<name>/model_config.yaml with the tokenizer in tokenizer_dir, plus the loop refiner and
    acoustic memory (and for balbal-v2-manas: 4x stem, learned memory temperature, language head). weights: a state dict path (or dict),
    loaded strictly; balbal-v2 language-head / condition entries in it are ignored (use split_state for training).
    Returns the float32 model in eval mode on `device`."""
    import torch
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import EncDecCTCModelBPE
    from nemo.core.classes.mixins import AccessMixin
    from nemo.utils import logging
    from balbal.model.conformer_loop_memory import attach_loop_memory
    logging.set_verbosity(logging.ERROR)
    check_name(name)
    cfg = OmegaConf.load(CONFIGS / name / 'model_config.yaml')
    for key, value in list(cfg.tokenizer.items()):
        if isinstance(value, str) and (value == 'tokenizer' or value.startswith('tokenizer/')):
            cfg.tokenizer[key] = str(Path(tokenizer_dir) / value.removeprefix('tokenizer').lstrip('/'))
    model = EncDecCTCModelBPE(cfg=cfg).float()
    model.loss.zero_infinity = False
    refiner = refiner_config(name)
    if model.encoder.subsampling_factor != refiner['subsampling_factor']: set_subsampling(model, refiner['subsampling_factor'])
    if name == 'balbal-v2':
        for layer in model.encoder.layers:   # NeMo's own attention kernel, not PyTorch SDPA (as trained)
            layer.self_attn.use_pytorch_sdpa = False
    attach_loop_memory(model, refiner['config'], memory_config=refiner['memory'])
    if refiner.get('learned_temperature'): learn_temperature(model.ctc_extra.acoustic_memory)
    if refiner.get('language_head'): attach_language_head(model, name)
    assert model.decoder.num_classes_with_blank == refiner['config']['vocabulary'] + 1
    if weights is not None:
        state = weights if isinstance(weights, dict) else load_state(weights)
        model.load_state_dict(split_state(state, name)[0], strict=True)
    AccessMixin.set_access_enabled(False, guid=model.model_guid)
    model.ctc_extra.inference_loops = loops or refiner['inference_loops']
    return model.float().to(device).eval()
