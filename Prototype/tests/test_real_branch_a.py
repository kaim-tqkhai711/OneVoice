from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tonebridge.config import PipelineConfig
from tonebridge.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
WAV = ROOT / "models/asr/zipformer-vi-int8/test_wavs/1.wav"
need = [WAV, ROOT / "models/silero_vad.onnx", ROOT / "models/nmt/vi-en-int8-arm64/encoder_model_quantized.onnx",
        ROOT / "models/tts/vits-piper-en_US-ljspeech-medium/tokens.txt"]


@pytest.mark.skipif(not all(p.exists() for p in need), reason="models not downloaded")
def test_real_branch_a_wav_to_wav():
    from tonebridge.stages.factory import real_branch_a

    cfg = PipelineConfig()
    x, sr = sf.read(WAV, dtype="float32")
    res = Pipeline(cfg, real_branch_a(cfg)).run(x, "t1")
    assert res.record.asr_text and res.record.nmt_text.isascii()
    assert res.out_wav.size > 0 and np.isfinite(res.out_wav).all()
    assert res.record.endpoint_to_first_audio_ms is not None
