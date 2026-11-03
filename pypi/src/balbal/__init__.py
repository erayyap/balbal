"""balbal: open Turkish speech recognition on CPU.

    import balbal
    print(balbal.transcribe('speech.wav')['text'])              # balbal-v2, downloaded from Hugging Face on first use
    model = balbal.load('balbal-v2-manas')                       # or 'manas'; reuse the loaded model
    result = model.transcribe('call.wav', hotwords=['Eray'])     # text + per-word timestamps and confidences
    stream = model.stream()                                      # live audio: stream.accept(piece) -> partial / final events

Command line: balbal run speech.wav [--model manas] [--json | --format srt -o out/]; balbal live; balbal serve; balbal wyoming
"""
from .engine import Model, load, transcribe
from .hub import MODELS, download
from .streaming import LiveStream

__version__ = '0.3.1'
__all__ = ['Model', 'LiveStream', 'load', 'transcribe', 'download', 'MODELS', '__version__']
