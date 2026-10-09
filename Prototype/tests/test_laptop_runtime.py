import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from tonebridge.audio import load_wav
from tonebridge.cli import build_stub_stages, public_row
from tonebridge.config import PipelineConfig
from tonebridge.contracts import AsrResult, GateAction, MtResult, SafetyReport
from tonebridge.gate import Gate, GateConfig, decide
from tonebridge.pipeline import Pipeline
from tonebridge.stages.factory import build_text_stages


def signal():
    return (0.1 * np.sin(np.arange(16000) * 0.1)).astype(np.float32)


@pytest.mark.parametrize("direction", ["vi-en", "en-vi", "en-ko", "ko-en"])
def test_routing_all_directions(direction):
    cfg = PipelineConfig(direction=direction)
    stages = build_stub_stages(cfg, None)
    seen = []

    class Asr:
        def transcribe(self, wav, lang):
            seen.append(lang)
            return AsrResult(text="hello there", lang=lang, confidence=0.9)

    stages.asr = Asr()
    result = Pipeline(cfg, stages).run(signal(), "routing")
    assert seen == [direction.split("-")[0]]
    assert result.record.direction == direction
    assert result.record.status == "ok" and result.out_wav.size


@pytest.mark.parametrize("stage", ["frontend", "vad", "denoise", "asr", "nmt", "safety", "gate", "tts"])
def test_stage_failure_discards_all_audio(stage):
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg, None)

    def fail(*args):
        raise RuntimeError("sensitive text must never enter error log")

    if stage == "gate":
        stages.gate = fail
    elif stage == "tts":
        def broken_stream(*args):
            yield np.ones(100, dtype=np.float32)
            fail()
        stages.tts.stream = broken_stream
    else:
        method = {"frontend": "process", "denoise": "process", "vad": "segment", "asr": "transcribe", "nmt": "translate", "safety": "check"}[stage]
        setattr(getattr(stages, "denoiser" if stage == "denoise" else stage), method, fail)
    result = Pipeline(cfg, stages).run(signal(), "failure")
    assert result.record.status == "error" and result.record.error_stage == stage
    assert result.record.gate.action == GateAction.ABSTAIN
    assert not result.out_wav.size and result.record.endpoint_to_first_audio_ms is None
    assert "sensitive" not in json.dumps(public_row(result.record))


@pytest.mark.parametrize("audio,reason", [(np.zeros(16000, np.float32), "no_speech_or_empty_asr"), (np.zeros(0, np.float32), "no_speech_or_empty_asr"), (np.ones(16000 * 16, np.float32), "utterance_too_long")])
def test_invalid_turn_does_not_call_models(audio, reason):
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg, None)
    stages.asr.transcribe = lambda *args: pytest.fail("ASR must not run")
    result = Pipeline(cfg, stages).run(audio, "rejected")
    assert result.record.gate.action == GateAction.REPEAT
    assert reason in result.record.gate.reasons and not result.out_wav.size


@pytest.mark.parametrize("evidence", [{"truncated": True}, {"terminated_by_eos": False}, {"constraints_satisfied": False}])
def test_incomplete_translation_never_reaches_tts(evidence):
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg, None)
    stages.nmt.translate = lambda text, src, tgt: MtResult(src_text=text, tgt_text="Hello there", src_lang=src, tgt_lang=tgt, **evidence)
    result = Pipeline(cfg, stages).run(signal(), "incomplete")
    assert result.record.gate.action == GateAction.CONFIRM and not result.out_wav.size


def test_missing_new_direction_fails_explicitly():
    with pytest.raises(ValueError, match="not registered"):
        build_text_stages(PipelineConfig(direction="en-ko"))


def test_source_override_cannot_disagree_with_direction():
    cfg = PipelineConfig(direction="en-vi")
    result = Pipeline(cfg, build_stub_stages(cfg, None)).run(signal(), "bad", src="vi")
    assert result.record.error_stage == "input" and not result.out_wav.size


def test_default_logging_strips_all_spoken_content():
    cfg = PipelineConfig()
    result = Pipeline(cfg, build_stub_stages(cfg, None)).run(signal(), "private")
    row = public_row(result.record)
    assert "asr_text" not in row and "nmt_text" not in row and "speak_text" not in row["gate"]
    assert public_row(result.record, True)["asr_text"]


@pytest.mark.parametrize("dtype,sr", [(np.uint8, 8000), (np.int16, 16000), (np.int32, 48000), (np.float32, 44100)])
def test_audio_encodings_downmix_and_resample(tmp_path, dtype, sr):
    x = 0.2 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
    if dtype == np.uint8:
        data = (x * 128 + 128).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        data = (x * 2 ** (np.dtype(dtype).itemsize * 8 - 1)).astype(dtype)
    else:
        data = x.astype(dtype)
    path = tmp_path / "stereo.wav"
    wavfile.write(path, sr, np.stack([data, data], axis=1))
    audio = load_wav(path)
    assert audio.shape == (16000,) and audio.dtype == np.float32
    assert 0.18 < np.max(audio) < 0.22


def test_nonfinite_audio_is_rejected(tmp_path):
    path = tmp_path / "invalid.wav"
    wavfile.write(path, 16000, np.array([np.nan], dtype=np.float32))
    with pytest.raises(ValueError):
        load_wav(path)


def test_unspaced_korean_is_not_rejected_by_word_ratio():
    from tonebridge.contracts import UrgencyResult
    decision = decide(AsrResult(text="그는괜찮은척하려고애쓰는것같았다", lang="ko", confidence=0.9),
                      SafetyReport(passed=True), UrgencyResult(label="UNKNOWN"),
                      "He seemed to be trying to pretend that he was all right", GateConfig())
    assert decision.action == GateAction.SPEAK


def test_microphone_capture_uses_native_rate_and_does_not_save(monkeypatch):
    import sys
    from types import SimpleNamespace
    from tonebridge.audio import record_microphone
    calls = []
    def record(frames, **kwargs):
        calls.append((frames, kwargs))
        return np.zeros((frames, 1), np.float32)
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(query_devices=lambda **kwargs: {"default_samplerate": 48000}, rec=record))
    x = record_microphone(1)
    assert x.shape == (16000,)
    assert calls[0][0] == 48000 and calls[0][1]["samplerate"] == 48000


def test_output_is_pcm16_and_never_overwrites(tmp_path):
    from tonebridge.audio import write_wav_new
    path = tmp_path / "approved.wav"
    write_wav_new(path, 22050, signal())
    before = path.read_bytes()
    sr, data = wavfile.read(path)
    assert sr == 22050 and data.dtype == np.int16
    with pytest.raises(FileExistsError):
        write_wav_new(path, 22050, -signal())
    assert path.read_bytes() == before and len(list(tmp_path.iterdir())) == 1


def test_partner_factory_handoff(monkeypatch):
    import sys
    from types import SimpleNamespace
    seen = []
    cfg = PipelineConfig(direction="en-ko", text_factory="partner_text:build")
    stages = build_stub_stages(cfg, None)
    def build(configuration, threads):
        seen.append((configuration.direction, threads))
        return stages.nmt, stages.safety
    monkeypatch.setitem(sys.modules, "partner_text", SimpleNamespace(build=build))
    nmt, safety = build_text_stages(cfg, 2)
    assert seen == [("en-ko", 2)] and nmt is stages.nmt and safety is stages.safety


def test_adapter_wrong_return_is_safe_error():
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg, None)
    stages.asr.transcribe = lambda *args: None
    result = Pipeline(cfg, stages).run(signal(), "invalid-contract")
    assert result.record.status == "error" and not result.out_wav.size


def test_watchdog_kills_worker_before_output(tmp_path):
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "tools/run_laptop.py"), "--timeout", "0.001", "--",
                             "--wav", str(tmp_path / "unused.wav"), "--out", str(tmp_path / "output.wav"), "--stub"],
                            capture_output=True, text=True, encoding="utf-8", timeout=10)
    assert result.returncode == 124
    assert json.loads(result.stdout)["error_type"] == "Timeout"
    assert not (tmp_path / "output.wav").exists()
