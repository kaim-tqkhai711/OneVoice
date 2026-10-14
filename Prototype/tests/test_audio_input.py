"""Audio boundary regressions for PCM, malformed input and silent/long turns."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from tonebridge.audio import AudioInputError, load_wav
from tonebridge.config import PipelineConfig
from tonebridge.contracts import GateAction
from tonebridge.pipeline import Pipeline
from tonebridge.stages.factory import build_stub_stages


@pytest.mark.parametrize("data,expected", [
    (np.array([0, 128, 255], np.uint8), [-1, 0, 127 / 128]),
    (np.array([-32768, 0, 32767], np.int16), [-1, 0, 32767 / 32768]),
    (np.array([-2147483648, 0, 2147483647], np.int32), [-1, 0, 1]),
    (np.array([-1, 0, 1], np.float32), [-1, 0, 1]),
    (np.array([-.25, 0, .75], np.float64), [-.25, 0, .75]),
])
def test_wav_normalization(tmp_path, data, expected):
    path = tmp_path / "input.wav"
    wavfile.write(path, 16000, data)
    x = load_wav(path)
    assert x.dtype == np.float32 and x.ndim == 1
    np.testing.assert_allclose(x, expected, atol=1e-7)
    assert np.isfinite(x).all() and np.max(np.abs(x)) <= 1


@pytest.mark.parametrize("sr,data", [
    (44100, np.zeros(20, np.int16)),
    (8000, np.zeros(20, np.int16)),
    (16000, np.zeros((20, 2), np.int16)),
    (16000, np.zeros(0, np.int16)),
    (16000, np.array([np.nan], np.float32)),
    (16000, np.array([np.inf], np.float32)),
    (16000, np.array([-1.01, 0], np.float32)),
])
def test_invalid_wav_rejected(tmp_path, sr, data):
    path = tmp_path / "input.wav"
    wavfile.write(path, sr, data)
    with pytest.raises(AudioInputError):
        load_wav(path)


def test_malformed_and_missing_file(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not a RIFF file")
    for p in (path, tmp_path / "missing.wav"):
        with pytest.raises(AudioInputError):
            load_wav(p)


def test_duration_limit_and_exact_boundary(tmp_path):
    path = tmp_path / "input.wav"
    wavfile.write(path, 16000, np.zeros(1600, np.int16))
    assert len(load_wav(path, max_utterance_s=.1)) == 1600
    wavfile.write(path, 16000, np.zeros(1601, np.int16))
    with pytest.raises(AudioInputError, match="max_utterance_s"):
        load_wav(path, max_utterance_s=.1)


def _forbidden(*args, **kwargs):
    pytest.fail("invalid or silent input reached an inference/TTS stage")


@pytest.mark.parametrize("wav", [
    np.array([np.nan], np.float32), np.array([np.inf], np.float32),
    np.array([2], np.float32), np.array([2147483647], np.int32),
    np.array([255], np.uint8), np.zeros((10, 2), np.float32),
    np.zeros(0, np.float32), np.zeros(1601, np.float32),
])
def test_invalid_direct_turn_abstains_without_tts(monkeypatch, wav):
    cfg = PipelineConfig(max_utterance_s=.1)
    stages = build_stub_stages(cfg)
    for stage, method in ((stages.asr, "transcribe"), (stages.nmt, "translate"), (stages.tts, "stream")):
        monkeypatch.setattr(stage, method, _forbidden)
    result = Pipeline(cfg, stages).run(wav, "invalid")
    assert result.record.status == "invalid_audio"
    assert result.record.gate.action == GateAction.ABSTAIN
    assert not result.out_wav.size


@pytest.mark.parametrize("dtype,silence", [(np.uint8, 128), (np.int32, 0), (np.int16, 0)])
def test_pcm_silence_short_circuits_before_models(tmp_path, monkeypatch, dtype, silence):
    path = tmp_path / "silence.wav"
    wavfile.write(path, 16000, np.full(16000, silence, dtype=dtype))
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg)
    monkeypatch.setattr(stages.asr, "transcribe", _forbidden)
    monkeypatch.setattr(stages.nmt, "translate", _forbidden)
    monkeypatch.setattr(stages.tts, "stream", _forbidden)
    result = Pipeline(cfg, stages).run(load_wav(path), "silent")
    assert result.record.status == "no_speech"
    assert result.record.gate.action == GateAction.REPEAT
    assert not result.out_wav.size


@pytest.mark.parametrize("samples", [np.array([0, 255], np.uint8),
                                    np.array([-2147483648, 2147483647], np.int32)])
def test_tiny_pcm_input_never_invokes_tts(tmp_path, monkeypatch, samples):
    path = tmp_path / "tiny.wav"
    wavfile.write(path, 16000, samples)
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg)
    monkeypatch.setattr(stages.asr, "transcribe", _forbidden)
    monkeypatch.setattr(stages.tts, "stream", _forbidden)
    result = Pipeline(cfg, stages).run(load_wav(path), "tiny-pcm")
    assert result.record.status == "no_speech"
    assert result.record.gate.action == GateAction.REPEAT and not result.out_wav.size
