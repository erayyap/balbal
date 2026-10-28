"""Initial weights for the trainers: --init (model weights, a training state.pt, or a released checkpoint) into a built model, plus the
optional language head (balbal-v2: entries decoder.lang_head.* or a sibling *.lang.pt) and language condition (lang_cond.*)."""
from pathlib import Path


def load_init(model, name, init, cond=False, cond_scale=10.):
    """Loads `init` strictly (acoustic model) and the head / condition when present; attaches the condition when cond. Returns a report."""
    from balbal import models as M
    report = dict(init=str(init) if init else None, language_head='new', condition=None)
    state = M.load_state(init) if init else {}
    acoustic, head, condition = M.split_state(state, name) if init else ({}, {}, {})
    if init: model.load_state_dict(acoustic, strict=True)
    if name == 'balbal-v2':
        M.attach_language_head(model, name)
        sibling = Path(str(init)[:-3] + '.lang.pt') if init and str(init).endswith('.pt') else None
        if not head and sibling is not None and sibling.exists():
            head = {k.removeprefix('decoder.lang_head.'): v for k, v in M.load_state(sibling).items()}
        if head: model.decoder.lang_head.load_state_dict(head); report['language_head'] = 'loaded'
    elif init:
        report['language_head'] = 'loaded'   # balbal-v2-manas: part of the model weights
    if cond:
        M.attach_condition(model, cond_scale)
        if condition: model.lang_cond.load_state_dict(condition)
        report['condition'] = 'loaded' if condition else 'zero-init'
    return report
