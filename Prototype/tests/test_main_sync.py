"""Regression at the A-C/upstream merge boundaries; no real-model claims."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile

from tonebridge.config import ROOT, PipelineConfig
from tonebridge.contracts import GateAction, GateDecision
from tonebridge.pipeline import Pipeline
from tonebridge.stages.factory import build_stub_stages
from tonebridge.telemetry import metadata_only


@pytest.mark.parametrize("direction", ["vi-en", "en-vi", "en-ko", "ko-en"])
def test_registered_laptop_cli_stub_keeps_mode_direction_and_privacy(tmp_path, direction):
    audio = tmp_path / "input.wav"
    wavfile.write(audio, 16000, (np.sin(np.arange(16000) * .1) * 3000).astype(np.int16))
    secret = "sensitive synthetic conversation for privacy regression"
    audio.with_suffix(".txt").write_text(secret, encoding="utf8")
    log = tmp_path / "turn.jsonl"
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "tonebridge.cli", "--mode", "stub",
        "--config", str(ROOT / "configs" / ("laptop_" + direction + ".json")),
        "--wav", str(audio), "--out", str(tmp_path / "out.wav"), "--log", str(log)],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True,
        encoding="utf8", timeout=20)
    assert result.returncode == 0, result.stderr
    row = json.loads(log.read_text(encoding="utf8"))
    assert row["mode"] == "stub" and row["direction"] == direction
    assert secret not in result.stdout + result.stderr + log.read_text(encoding="utf8")
    assert (tmp_path / "out.wav").exists()


def test_custom_gate_cannot_bypass_full_acoustic_policy():
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg)
    stages.mode = "full"
    stages.gate = lambda asr, safety, urgency, text, *_: GateDecision(action=GateAction.SPEAK, speak_text=text)
    stages.tts.stream = lambda *a: pytest.fail("uncalibrated full output reached TTS")
    with Pipeline(cfg, stages) as pipe:
        result = pipe.run(np.ones(16000, np.float32) * .1, "quality")
    assert result.record.gate.action in (GateAction.REPEAT, GateAction.CONFIRM)
    assert not result.out_wav.size


def test_restoration_nmt_pack_does_not_shadow_historical_pins():
    registry = json.loads((ROOT / "configs/nmt/models_laptop.json").read_text(encoding="utf8"))
    restored = registry["models"]["vi-en"]["assets"]
    historical = PipelineConfig.load(ROOT / "configs/pipeline.json").models["nmt_vi_en"].path
    assert restored == "models/nmt-laptop/vi-en" and restored != historical
    manifest = json.loads((ROOT / "configs/laptop_assets_manifest.json").read_text(encoding="utf8"))
    assert any(r["path"].startswith(restored + "/") for r in manifest["files"])
    assert not any(r["path"].startswith(historical + "/") for r in manifest["files"])


def test_rich_nmt_metadata_keeps_only_numeric_and_boolean_evidence():
    row = metadata_only({"nmt_evidence": {"source_tokens": 2, "output_tokens": 3, "eos_reached": True,
        "model_id": "sensitive", "tokenizer_issues": ["sensitive"], "unsatisfied_constraints": ["sensitive"]},
        "safety_reasons": ["sensitive"], "safety_status": "FAIL"})
    assert row["nmt_evidence"]["source_tokens"] == 2
    assert row["nmt_evidence"]["eos_reached"] is True and row["safety_status"] == "FAIL"
    assert "sensitive" not in json.dumps(row)


def test_laptop_registry_snapshot_is_effective_and_hash_checked(tmp_path):
    from tonebridge.laptop_config import prepare_laptop
    from tonebridge.stages.audio_factory import audio_specs
    configs = tmp_path / "configs"; configs.mkdir()
    for name in ("safety_lexicon_vi_en.json", "clinical_lexicon.json"):
        (configs / name).write_bytes((ROOT / "configs" / name).read_bytes())
    paths = ["models/asr/" + n for n in ("encoder", "decoder", "joiner", "tokens")]
    paths += ["models/tts/model", "models/tts/tokens", "models/tts/data/dictionary", "models/silero_vad.onnx"]
    paths += ["models/nmt/" + n for n in ("encoder", "decoder", "config", "source", "target", "vocab")]
    rows = []
    for name in paths:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"pin fixture")
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    audio = {"asr": {"vi": dict(zip(("encoder", "decoder", "joiner", "tokens"), paths[:4]))},
        "tts": {"en": {"kind": "vits", "model": paths[4], "tokens": paths[5], "data_dir": "models/tts/data"}}}
    (configs / "audio.json").write_text(json.dumps(audio))
    (configs / "nmt.json").write_text(json.dumps({"models": {"vi-en": {"assets": "models/nmt", "backend": "onnx"}}}))
    (configs / "laptop_assets_manifest.json").write_text(json.dumps({"files": rows}))
    (configs / "pipeline.json").write_text(json.dumps({"audio_config": "configs/audio.json",
        "nmt_config": "configs/nmt.json", "urgency_enabled": False}))
    cfg = PipelineConfig.load(configs / "pipeline.json")
    frozen = prepare_laptop(cfg)
    before = frozen.config_hash()
    audio["asr"]["vi"]["encoder"] = "models/changed"
    (configs / "audio.json").write_text(json.dumps(audio))
    assert audio_specs(frozen)["asr"]["vi"]["encoder"] == paths[0]
    assert frozen.config_hash() == before
    (configs / "audio.json").write_text(json.dumps(frozen._laptop_audio_specs))
    (tmp_path / paths[0]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="preflight"):
        prepare_laptop(cfg)


def test_laptop_direction_override_preserves_custom_config_root(tmp_path):
    configs = tmp_path / "configs"; configs.mkdir()
    (configs / "safety_lexicon_vi_en.json").write_bytes((ROOT / "configs/safety_lexicon_vi_en.json").read_bytes())
    config = configs / "laptop.json"
    config.write_text(json.dumps({"audio_config": "explicit_registry.json", "models": {}, "direction": "vi-en"}))
    audio = tmp_path / "in.wav"
    wavfile.write(audio, 16000, np.ones(16000, np.int16) * 1000)
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "tonebridge.cli", "--mode", "stub",
        "--config", str(config), "--direction", "en-vi", "--wav", str(audio),
        "--out", str(tmp_path / "out.wav"), "--log", str(tmp_path / "log.jsonl")],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True,
        text=True, encoding="utf8", timeout=20)
    assert result.returncode == 0, result.stderr
    cfg = PipelineConfig.load(config); cfg.direction = "en-vi"
    expected = build_stub_stages(cfg).effective_config.config_hash()
    assert json.loads(result.stdout)["config_hash"] == expected


def test_laptop_log_io_failure_does_not_leak_paths_or_publish_audio(tmp_path):
    audio = tmp_path / "in.wav"
    wavfile.write(audio, 16000, np.ones(16000, np.int16) * 1000)
    secret = "sensitive_dialogue_phrase"
    bad_log = tmp_path / secret; bad_log.mkdir()
    output = tmp_path / "out.wav"
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "tonebridge.cli", "--mode", "stub",
        "--config", str(ROOT / "configs/laptop_vi-en.json"), "--wav", str(audio),
        "--out", str(output), "--log", str(bad_log)],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True,
        text=True, encoding="utf8", timeout=20)
    assert result.returncode == 2 and not output.exists()
    assert secret not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr
