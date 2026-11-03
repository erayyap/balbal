"""Initial weights for the trainers: --init (model weights, a training state.pt, or a released checkpoint) into a built model, plus the
optional language head (balbal-v2: entries decoder.lang_head.* or a sibling *.lang.pt) and language condition (lang_cond.*).
A language head or condition of another size than the model's (a model trained for other languages) is left out: it starts fresh."""
from pathlib import Path


def _same_shapes(module, state):
    own = module.state_dict()
    return bool(state) and all(k in own and own[k].shape == v.shape for k, v in state.items())


def load_init(model, name, init, cond=False, cond_scale=10.):
    """Loads `init` strictly (acoustic model) and the head / condition when present; attaches the condition when cond. Returns a report."""
    from balbal import models as M
    languages = tuple(getattr(model, 'balbal_languages', M.HEAD_LANGUAGES))
    report = dict(init=str(init) if init else None, language_head='new', condition=None, languages=list(languages))
    state = M.load_state(init, trusted=True) if init else {}
    acoustic, head, condition = M.split_state(state, name) if init else ({}, {}, {})
    if init and name != 'balbal-v2':   # the manas head is part of the acoustic weights: keep it only when it has the model's size
        prefix = M.LANG_HEAD[name]; inside = {k[len(prefix):]: v for k, v in acoustic.items() if k.startswith(prefix)}
        own = M.language_head(model)
        if inside and (own is None or not _same_shapes(own, inside)):
            acoustic = {k: v for k, v in acoustic.items() if not k.startswith(prefix)}; report['language_head'] = 'new (size differs)'
            missing, unexpected = model.load_state_dict(acoustic, strict=False)
            if unexpected or any(not k.startswith(prefix) for k in missing): raise RuntimeError(f'init mismatch: {missing} {unexpected}')
        else:
            model.load_state_dict(acoustic, strict=True)
            if inside: report['language_head'] = 'loaded'
    elif init:
        model.load_state_dict(acoustic, strict=True)
    if name == 'balbal-v2':
        M.attach_language_head(model, name, languages)
        sibling = Path(str(init)[:-3] + '.lang.pt') if init and str(init).endswith('.pt') else None
        if not head and sibling is not None and sibling.exists():
            head = {k.removeprefix('decoder.lang_head.'): v for k, v in M.load_state(sibling, trusted=True).items()}
        if head and _same_shapes(model.decoder.lang_head, head):
            model.decoder.lang_head.load_state_dict(head); report['language_head'] = 'loaded'
        elif head:
            report['language_head'] = 'new (size differs)'
    if cond:
        M.attach_condition(model, cond_scale, languages)
        ok = _same_shapes(model.lang_cond, condition)
        if ok: model.lang_cond.load_state_dict(condition)
        report['condition'] = 'loaded' if ok else 'new (size differs)' if condition else 'zero-init'
    return report
