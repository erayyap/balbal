"""balbal: open Turkish speech recognition on CPU.

    import balbal
    print(balbal.transcribe('speech.wav')['text'])              # balbal-v2, downloaded from Hugging Face on first use
    model = balbal.load('balbal-v2-manas')                       # or 'manas'; reuse the loaded model
    result = model.transcribe('call.wav', hotwords=['Eray'])     # text + per-word timestamps

Command line: balbal run speech.wav [--model manas] [--json]
"""
from .engine import Model, load, transcribe
from .hub import MODELS, download

__version__ = '0.1.0'
__all__ = ['Model', 'load', 'transcribe', 'download', 'MODELS', '__version__']
