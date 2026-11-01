#!/usr/bin/env python3
"""PhoWhisper-small ASR — Vietnamese audio → text."""
from __future__ import annotations
import torch

_model = None
_proc = None


def load(model_id="vinai/PhoWhisper-small"):
    global _model, _proc
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    _proc = AutoProcessor.from_pretrained(model_id)
    _model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")
    _model.eval()
    print(f"ASR loaded: {model_id} ({sum(p.numel() for p in _model.parameters())/1e6:.0f}M params)")


def transcribe(wav_path: str) -> str:
    import librosa
    audio, _ = librosa.load(wav_path, sr=16000)
    inputs = _proc(audio, sampling_rate=16000, return_tensors="pt")
    feats = inputs["input_features"].to(_model.device, dtype=torch.float16)
    with torch.no_grad():
        ids = _model.generate(feats, max_new_tokens=256)
    return _proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def unload():
    global _model, _proc
    del _model, _proc
    _model = _proc = None
    torch.cuda.empty_cache()
