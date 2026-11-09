"""Unit tests without a model."""
from pathlib import Path
import json

import numpy as np
import pytest

from balbal.formats import render, segments_from_words, to_srt, to_vtt, two_lines
from balbal.itn import itn_text, itn_words
from balbal.streaming import pcm


@pytest.mark.parametrize('spoken, digits', [
    ('bin dokuz yüz doksan beş', '1995'),
    ('sıfır doksan kırk beş altmış yedi', '0 90 45 67'),
    ('dokuz sıfır bir', '9 0 1'),
    ('bir gün', 'bir gün'),                       # a lone ambiguous word stays a word
    ('saat on yedide geldi', "saat 17'de geldi"),
    ('Kırk beş.', '45.'),
])
def test_itn_text(spoken, digits):
    assert itn_text(spoken) == digits


def test_itn_words_merges_runs_with_lowest_confidence():
    words = [dict(word='yirmi', start=0.0, end=0.4, confidence=0.9), dict(word='beş', start=0.4, end=0.7, confidence=0.6),
             dict(word='lira', start=0.8, end=1.1, confidence=0.95)]
    out = itn_words(words)
    assert [w['word'] for w in out] == ['25', 'lira']
    assert out[0]['start'] == 0.0 and out[0]['end'] == 0.7 and out[0]['confidence'] == 0.6


def words(spec):
    return [dict(word=w, start=s, end=e, confidence=c) for w, s, e, c in spec]


def test_segments_split_on_pause_sentence_end_and_length():
    ws = words([('Merhaba.', 0.0, 0.5, 1.0), ('Nasılsın', 0.6, 1.0, 0.5), ('bugün', 2.0, 2.4, 1.0)])
    segs = segments_from_words(ws)
    assert [s['text'] for s in segs] == ['Merhaba.', 'Nasılsın', 'bugün']   # sentence end, then a 1 s pause
    assert segs[1]['avg_logprob'] == pytest.approx(np.log(0.5), abs=1e-4)
    long = words([(f'kelime{i}', i * 0.3, i * 0.3 + 0.25, 1.0) for i in range(40)])
    for s in segments_from_words(long):
        assert s['end'] - s['start'] <= 7.0 + 1e-9 and len(s['text']) <= 84


def test_segments_keep_itn_numbers_whole():
    from balbal.formats import group_words
    ws = words([('Numara', 0.0, 0.5, 1.0), ('0 532 45 67', 0.5, 2.5, 0.8), ('tamam.', 2.6, 3.0, 0.9)])
    assert [len(g) for g in group_words(ws)] == [3] and segments_from_words(ws)[0]['text'] == 'Numara 0 532 45 67 tamam.'


def test_subtitles():
    segs = segments_from_words(words([('Merhaba', 0.5, 1.0, 1.0), ('dünya.', 1.0, 1.6, 1.0)]))
    assert to_srt(segs) == '1\n00:00:00,500 --> 00:00:01,600\nMerhaba dünya.\n\n'
    assert to_vtt(segs) == 'WEBVTT\n\n00:00:00.500 --> 00:00:01.600\nMerhaba dünya.\n\n'
    assert two_lines('a' * 30 + ' ' + 'b' * 30) == 'a' * 30 + '\n' + 'b' * 30
    result = dict(text='Merhaba dünya.', words=words([('Merhaba', 0.5, 1.0, 1.0), ('dünya.', 1.0, 1.6, 1.0)]))
    assert render(result, 'txt') == 'Merhaba dünya.\n' and json.loads(render(result, 'json'))['text'] == 'Merhaba dünya.'


def test_pcm_inputs():
    x = np.array([0, 16384, -32768], np.int16)
    assert np.allclose(pcm(x), [0, 0.5, -1]) and np.allclose(pcm(x.tobytes()), [0, 0.5, -1])
    assert pcm(np.zeros(3, np.float64)).dtype == np.float32
    with pytest.raises(ValueError): pcm(np.zeros((2, 2)))


def test_decimator_is_block_invariant():
    pytest.importorskip('scipy')
    from balbal.live import Decimator
    x = np.sin(2 * np.pi * 440 * np.arange(96000) / 48000).astype(np.float32)
    whole = Decimator(48000)(x)
    d = Decimator(48000); pieces = np.concatenate([d(x[i:i + 4801]) for i in range(0, len(x), 4801)])
    assert len(whole) == 32000 and np.allclose(whole, pieces, atol=1e-6)


def test_cli_version(capsys):
    from balbal import __version__
    from balbal.cli import main
    with pytest.raises(SystemExit): main(['--version'])
    assert __version__ in capsys.readouterr().out


def test_batch_expand(tmp_path):
    from balbal.batch import expand
    (tmp_path / 'a').mkdir(); (tmp_path / 'a' / 'x.wav').write_bytes(b''); (tmp_path / 'a' / 'notes.txt').write_text('', encoding='utf-8')
    (tmp_path / 'y.mp3').write_bytes(b'')
    got = expand([tmp_path])
    assert [rel.as_posix() for _, rel in got] == ['a/x.wav', 'y.mp3']


def test_language_rules():
    from balbal import lang
    tr, az, en = lang.get('tr-TR'), lang.get('az'), lang.get('en')
    assert tr.lower('IŞIK İzmir') == 'ışık izmir' and az.lower('İlham') == 'ilham' and en.lower('Istanbul') == 'istanbul'
    assert tr.capitalize('istanbul') == 'İstanbul' and en.capitalize('istanbul') == 'Istanbul'
    assert lang.itn_functions('tr')[0]('bin dokuz yüz doksan beş') == '1995'
    assert lang.itn_functions('az') is None and lang.itn_functions('kk') is None and lang.itn_functions(None) is None


def test_resolve_and_model_info(tmp_path):
    from balbal import hub
    from balbal.engine import model_info
    assert hub.resolve('manas') == ('registry', 'balbal-v2-manas')
    assert hub.resolve('someone/azerbaijani-model') == ('repo', 'someone/azerbaijani-model')
    assert hub.resolve(str(tmp_path)) == ('local', tmp_path)
    with pytest.raises(ValueError): hub.resolve('not a model')
    assert model_info(tmp_path) == dict(model='balbal-v2', name='balbal-v2', languages=['tr'], itn='tr', long_windows=(30., 10.), frame_samples=640)
    (tmp_path / 'model.json').write_text(json.dumps({'model': 'balbal-v2-manas', 'languages': ['az'], 'name': 'az-model'}), encoding='utf-8')
    info = model_info(tmp_path)
    assert info['model'] == 'balbal-v2-manas' and info['name'] == 'az-model' and info['languages'] == ['az'] and info['itn'] == 'az'
    assert model_info(tmp_path, name='owner/x')['name'] == 'az-model'


def test_repo_without_onnx_says_how_to_export(tmp_path, monkeypatch):
    import huggingface_hub
    from balbal import hub
    listed = ['model.json', 'model.pt', 'lm/model.binary', 'lm/tokenizer.model']   # a torch checkpoint and an LM, no onnx/
    monkeypatch.setattr(huggingface_hub, 'list_repo_files', lambda repo, revision=None: listed)
    def no_download(**kwargs): raise AssertionError('nothing may be downloaded')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', no_download)
    with pytest.raises(FileNotFoundError, match='export_onnx'):
        hub.download('owner/torch-only', 'fast', cache_dir=tmp_path)


def test_other_scripts_punctuation_and_sentence_ends():
    from balbal.word_timing import _content
    assert not _content('؟') and not _content('▁«') and _content('▁%') and _content('/')
    words = [dict(word='مرحبا؟', start=0., end=.5), dict(word='كيف', start=.6, end=1.)]
    assert len(segments_from_words(words)) == 2


def test_wyoming_routes_by_language():
    from types import SimpleNamespace
    from balbal.wyoming_server import choose
    loaded = {'balbal-v2': SimpleNamespace(languages=['tr'], source='balbal-v2'),
              'balbal-v2-manas': SimpleNamespace(languages=['tr', 'en'], source='balbal-v2-manas'),
              'az-model': SimpleNamespace(languages=['az'], source='/models/az')}
    pool = SimpleNamespace(models=list(loaded), loaded=loaded)
    from balbal.server import Pool
    pool.lookup = lambda name: Pool.lookup(pool, name)
    assert choose(pool, None, 'en-US') == 'balbal-v2-manas' and choose(pool, None, 'az') == 'az-model'
    assert choose(pool, None, 'tr') == 'balbal-v2' and choose(pool, None, 'de') == 'balbal-v2' and choose(pool, None, None) == 'balbal-v2'
    assert choose(pool, 'manas', 'tr') == 'balbal-v2-manas' and choose(pool, 'az-model') == 'az-model'


def test_zellige_registry_and_info(tmp_path):
    from balbal import hub
    from balbal.engine import model_info, ONNX_ONLY
    assert hub.resolve('zellige') == ('registry', 'balbal-v2e-zellige') == hub.resolve('erayyapagci/balbal-v2e-zellige')
    for old in ('balbal-v2.1-zellige', 'v2.1', 'v2.1-zellige', 'v2e', 'v2e-zellige', 'erayyapagci/balbal-v2.1-zellige'):   # renamed 2026-11-08
        assert hub.resolve(old) == ('registry', 'balbal-v2e-zellige')
    assert hub.resolve('v2') == ('registry', 'balbal-v2') and hub.resolve('manas') == ('registry', 'balbal-v2-manas')
    assert 'NC' in hub.LICENSES['balbal-v2e-zellige'] and hub.graph_files('balbal-v2e-zellige', 'fast') == ['onnx/model.onnx']
    assert hub.graph_files('balbal-v2', 'fast') == hub.GRAPHS['fast'] and 'balbal-v2e-zellige' in ONNX_ONLY
    info = model_info(tmp_path, 'balbal-v2e-zellige')
    assert info['languages'] == ['ary'] and info['frame_samples'] == 1280 and info['itn'] == 'ary'
    from balbal import lang
    assert lang.itn_functions('ary') is None   # Moroccan Darija: no number writer, no casing


def test_calliope_registry_and_info(tmp_path):
    from balbal import hub
    from balbal.engine import model_info, ONNX_ONLY
    for name in ('calliope', 'v2e-calliope', 'balbal-v2e-calliope', 'erayyapagci/balbal-v2e-calliope'):
        assert hub.resolve(name) == ('registry', 'balbal-v2e-calliope')
    assert hub.resolve('zellige') == ('registry', 'balbal-v2e-zellige')
    assert 'NC' not in hub.LICENSES['balbal-v2e-calliope'] and 'balbal-v2e-calliope' in ONNX_ONLY
    assert hub.graph_files('balbal-v2e-calliope', 'onnx') == hub.graph_files('balbal-v2e-calliope', 'fast') == ['onnx/model.onnx', 'onnx/model.onnx.data']
    info = model_info(tmp_path, 'balbal-v2e-calliope')
    assert info['languages'] == ['grc'] and info['frame_samples'] == 640
    from balbal import lang
    assert lang.itn_functions(info['itn']) is None and lang.NAMES['grc'] == 'ancient greek'


def test_version_matches_package_metadata():
    import balbal
    from importlib import metadata
    try: installed = metadata.version('balbal')
    except metadata.PackageNotFoundError: pytest.skip('balbal is not installed (source tree on sys.path)')
    if Path(balbal.__file__).resolve().parents[1].name == 'src': pytest.skip('source tree imported, not the installed package')
    assert balbal.__version__ == installed


def test_registry_models_have_packaged_decoder_settings():
    import balbal
    from balbal import hub
    for name in hub.MODELS:   # engine reads configs/<registry name>/decoder.json (the folder must follow a model rename)
        assert (Path(balbal.__file__).parent / 'configs' / name / 'decoder.json').is_file(), name


def test_license_notice(tmp_path, capsys):
    from balbal import hub
    (tmp_path / 'model.json').write_text(json.dumps({'license': 'cc-by-nc-sa-4.0', 'license_name': 'CC BY-NC-SA 4.0'}), encoding='utf-8')
    assert hub.license_notice(tmp_path, remember=True) is True
    assert 'CC BY-NC-SA 4.0' in capsys.readouterr().err
    assert hub.license_notice(tmp_path, remember=True) is False and capsys.readouterr().err == ''   # once
    free = tmp_path / 'free'; free.mkdir()
    (free / 'model.json').write_text(json.dumps({'license': 'other'}), encoding='utf-8')
    assert hub.license_notice(free) is False and capsys.readouterr().err == ''


def test_word_timing_frame_seconds():
    import sentencepiece  # noqa: F401  (only to skip where the tokenizer package is missing)
    from balbal.word_timing import words_from_spans

    class Sp:
        pieces = {0: '\u2581a', 1: '\u2581b'}
        def id_to_piece(self, t): return self.pieces[t]
        def decode(self, ids): return ''.join(self.pieces[t] for t in ids).replace('\u2581', ' ')
    spans = [(0, 10, 10), (1, 30, 30)]
    a = words_from_spans(spans, Sp()); b = words_from_spans(spans, Sp(), frame_seconds=.08)
    assert a[0]['start'] == pytest.approx(.36) and b[0]['start'] == pytest.approx(.72)   # (frame 10 - LEAD 1 frame) x frame length
    assert b[1]['end'] > b[1]['start']
