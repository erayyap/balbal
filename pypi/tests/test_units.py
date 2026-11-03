"""Unit tests without a model."""
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
    (tmp_path / 'a').mkdir(); (tmp_path / 'a' / 'x.wav').write_bytes(b''); (tmp_path / 'a' / 'notes.txt').write_text('')
    (tmp_path / 'y.mp3').write_bytes(b'')
    got = expand([tmp_path])
    assert [str(rel) for _, rel in got] == ['a/x.wav', 'y.mp3']
