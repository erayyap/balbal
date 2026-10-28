"""Whisper-large-v3-turbo teacher for steering (transformers), language tr, beam 5, max 220 new tokens. fp16 on a GPU, fp32 on CPU (slow)."""
import os
os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')
import numpy as np
import torch


class TurboTeacher:
    def __init__(self, path, device, beam=5):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.device, self.beam = device, beam; self.dtype = torch.float16 if device.startswith('cuda') else torch.float32
        self.proc = WhisperProcessor.from_pretrained(path)
        self.model = WhisperForConditionalGeneration.from_pretrained(path, torch_dtype=self.dtype, attn_implementation='sdpa').to(device).eval()

    def transcribe(self, wave):
        from transformers.modeling_outputs import BaseModelOutput
        feats = self.proc.feature_extractor([np.asarray(wave, dtype=np.float32)[:30 * 16000]], sampling_rate=16000,
                                            return_tensors='pt').input_features.to(self.device, self.dtype)
        with torch.inference_mode():
            enc = self.model.model.encoder(feats).last_hidden_state
            ids = self.model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=enc), language='tr', task='transcribe',
                                      num_beams=self.beam, max_new_tokens=220)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
