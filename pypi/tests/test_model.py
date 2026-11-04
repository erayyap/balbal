"""Model tests (BALBAL_TEST_MODELS=1): offline output shape, streaming == offline decoding on the same frames, domain LM, server
endpoints."""
import io
import json

import numpy as np
import pytest

pytestmark = pytest.mark.model


def test_silence_and_noise_are_no_speech(model):
    # the models can write a short word over digital silence; no_speech flags it (no word reaches NO_SPEECH_CONFIDENCE)
    rng = np.random.default_rng(0)
    for wave in (np.zeros(16000 * 3, np.float32), np.zeros(16000 * 10, np.float32), (0.003 * rng.standard_normal(48000)).astype(np.float32)):
        r = model.transcribe(wave)
        assert r['no_speech'] and r['duration_seconds'] == len(wave) / 16000


def test_result_shape(model, speech):
    for wave in speech:
        r = model.transcribe(wave)
        assert set(r) >= {'text', 'words', 'no_speech', 'duration_seconds', 'inference_seconds', 'realtime_factor', 'model', 'runtime'}
        for w in r['words']:
            assert set(w) == {'word', 'start', 'end', 'confidence'} and 0 <= w['confidence'] <= 1 and w['start'] <= w['end']


def test_stream_equals_offline_on_its_frames(model, speech):
    """With endpointing off, the streamed text equals the offline decoder on the frames the stream decoded, for any piece size."""
    from balbal.streaming import simulate
    off = dict(endpoint=1e9, max_segment=1e9, trace=True, itn=False)
    for wave in speech:
        a, _, s = simulate(model, wave, mode='window', first_right=.64, second_pass=False, **off)
        assert a == model.decoder.decode(np.concatenate(s._stream.trace))
        b, _, _ = simulate(model, wave, piece=.05, mode='window', first_right=.64, second_pass=False, **off)
        assert a == b
        c, _, s = simulate(model, wave, mode='segment', **off)
        assert c == model.decoder.decode(np.concatenate(s._stream.trace))


def test_stream_events(model, speech):
    stream = model.stream()
    events = [e for piece in np.array_split(speech[0], 40) for e in stream.accept(piece)] + stream.finish()
    assert all(e['type'] in ('partial', 'final') for e in events)
    finals = [e['text'] for e in events if e['type'] == 'final']
    assert stream.transcript().replace(' ', '') == ''.join(finals).replace(' ', '')
    assert stream.stats['audio_seconds'] == pytest.approx(len(speech[0]) / 16000, abs=1e-3)


def test_domain_lm(model, speech, tmp_path):
    """The model's own LM as the domain LM mixes to the same distribution (same text as without it); weight 0 loads no domain LM;
    a domain LM with another tokenizer is refused."""
    from balbal import Model
    from balbal._native.ctc_lm_bonus import MixLM
    lm = model.model_dir / 'lm'
    load = lambda **settings: Model(model.source, model.runtime, model.threads, settings=settings)
    same = load(domain_lm=str(lm), domain_weight=0.5)
    assert isinstance(same.decoder.lm, MixLM)
    assert same.decoder.lm.mix(-1., -3.) == pytest.approx(np.log10(.5e-1 + .5e-3), abs=1e-6)
    for wave in speech:
        assert same.transcribe(wave)['text'] == model.transcribe(wave)['text']
    assert not isinstance(load(domain_lm=str(lm), domain_weight=0.).decoder.lm, MixLM)
    (tmp_path / 'model.binary').symlink_to(lm / 'model.binary'); (tmp_path / 'tokenizer.model').write_bytes(b'other')
    with pytest.raises(ValueError, match='another tokenizer'):
        load(domain_lm=str(tmp_path))


def test_server(model, speech):
    pytest.importorskip('fastapi'); pytest.importorskip('httpx')
    import soundfile as sf
    from fastapi.testclient import TestClient
    from balbal.server import create_app
    client = TestClient(create_app([model.source], threads=2, api_key='k'))
    h = {'Authorization': 'Bearer k'}
    buf = io.BytesIO(); sf.write(buf, speech[0], 16000, format='WAV'); data = buf.getvalue()
    assert client.post('/v1/audio/transcriptions', files={'file': ('a.wav', data)}).status_code == 401
    r = client.post('/v1/audio/transcriptions', headers=h, files={'file': ('a.wav', data)},
                    data={'model': 'whisper-1', 'response_format': 'verbose_json', 'timestamp_granularities[]': ['word', 'segment']})
    assert r.status_code == 200 and r.json()['text'] == model.transcribe(speech[0])['text']
    for seg in r.json()['segments']: assert 0 <= seg['no_speech_prob'] <= 1 and seg['avg_logprob'] <= 0
    assert client.post('/v1/audio/transcriptions', headers=h, files={'file': ('a.wav', data)},
                       data={'response_format': 'nope'}).json()['error']['code'] == 'invalid_value'
    pcm = (np.clip(speech[0], -1, 1) * 32767).astype('<i2').tobytes()
    with client.websocket_connect('/v1/stream?api_key=k') as ws:
        assert ws.receive_json()['type'] == 'ready'
        for i in range(0, len(pcm), 3200): ws.send_bytes(pcm[i:i + 3200])
        ws.send_text(json.dumps({'type': 'end'}))
        events = []
        while not events or events[-1]['type'] != 'done': events.append(ws.receive_json())
    assert events[-1]['type'] == 'done' and 'transcript' in events[-1]
