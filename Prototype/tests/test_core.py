import json

import numpy as np
import pytest

from tonebridge.config import ModelSpec, PipelineConfig
from tonebridge.contracts import AudioBuffer, GateAction, GateDecision, TurnRecord
from tonebridge.telemetry import JsonlLogger, TurnClock, rtf, summarize_latency


def test_config_hash_deterministic_and_sensitive():
    a, b = PipelineConfig(), PipelineConfig()
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != PipelineConfig(denoise_mode="on").config_hash()
    assert a.config_hash() != PipelineConfig(seed=1).config_hash()


def test_seed_everything_reproducible():
    cfg = PipelineConfig(seed=7)
    cfg.seed_everything()
    x = np.random.rand(4)
    cfg.seed_everything()
    y = np.random.rand(4)
    assert np.array_equal(x, y)


def test_model_spec_requires_license():
    with pytest.raises(Exception):
        ModelSpec(name="x", version="1")  # license missing


def test_audio_duration():
    assert AudioBuffer(wav=np.zeros(16000, np.float32)).duration_s == 1.0


def test_turn_clock_and_jsonl(tmp_path):
    clk = TurnClock()
    with clk.stage("asr"):
        pass
    clk.mark("endpoint")
    clk.mark("first_audio")
    assert clk.stage_ms["asr"] >= 0 and clk.between_ms("endpoint", "first_audio") >= 0
    assert clk.between_ms("endpoint", "missing") is None
    rec = TurnRecord(session_id="s", utterance_id="u", config_hash="h", direction="vi-ko", audio_s=1.0,
                     stage_ms={"asr": 1.0}, endpoint_to_first_audio_ms=2.0, total_ms=3.0, rtf=rtf(3.0, 1.0),
                     peak_rss_mb=1.0, gate=GateDecision(action=GateAction.SPEAK), asr_text="a", nmt_text="b")
    log = JsonlLogger(tmp_path / "t.jsonl")
    log.write(json.loads(rec.model_dump_json()))
    row = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[0])
    assert row["gate"]["action"] == "SPEAK" and abs(row["rtf"] - 0.003) < 1e-9


def test_summarize_latency_has_ci():
    s = summarize_latency(list(range(1, 101)))
    assert s["n"] == 100 and s["p50_ci95"][0] <= s["p50"] <= s["p50_ci95"][1]
