"""Fault injection at real control boundaries using the existing skeleton stages."""
import multiprocessing as mp
import os
import time
from threading import Event

import numpy as np
import pytest

from tonebridge.config import PipelineConfig
from tonebridge.contracts import AsrResult, GateAction, MtResult
from tonebridge.gate import Gate
from tonebridge.pipeline import Pipeline
from tonebridge.stages.factory import build_stub_stages


def _wav():
    return (.1 * np.sin(np.arange(16000) * .1)).astype(np.float32)


def _fail(*args, **kwargs):
    raise RuntimeError("injected stage failure")


def _forbidden(*args, **kwargs):
    pytest.fail("failed turn called a forbidden downstream stage")


@pytest.mark.parametrize("stage,method", [("frontend", "process"), ("vad", "segment"),
    ("denoiser", "process"), ("asr", "transcribe"), ("nmt", "translate"),
    ("safety", "check"), ("branch_b", "analyze")])
def test_stage_exception_is_safe_turn(monkeypatch, stage, method):
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg)
    monkeypatch.setattr(getattr(stages, stage), method, _fail)
    monkeypatch.setattr(stages.tts, "stream", _forbidden)
    result = Pipeline(cfg, stages).run(_wav(), "failed")
    assert result.record.status == "error"
    assert result.record.gate.action == GateAction.ABSTAIN
    assert result.record.error_stage in (stage, "denoise")
    assert not result.out_wav.size and result.record.gate.speak_text is None


def test_gate_exception_and_partial_tts_error(monkeypatch):
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg)
    stages.gate = _fail
    result = Pipeline(cfg, stages).run(_wav(), "gate-error")
    assert result.record.error_stage == "gate" and not result.out_wav.size

    def broken_tts(*args):
        yield np.zeros(10, np.float32)
        raise ValueError("injected failure after first chunk")

    stages = build_stub_stages(cfg)
    monkeypatch.setattr(stages.tts, "stream", broken_tts)
    result = Pipeline(cfg, stages).run(_wav(), "tts-error")
    assert result.record.error_stage == "tts" and not result.out_wav.size
    assert result.record.endpoint_to_first_audio_ms is None
    assert result.record.gate.speak_text is None


def test_vad_empty_segment_skips_asr_nmt_tts(monkeypatch):
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg)
    monkeypatch.setattr(stages.vad, "segment", lambda x: (0.0, 0.0))
    monkeypatch.setattr(stages.asr, "transcribe", _forbidden)
    monkeypatch.setattr(stages.nmt, "translate", _forbidden)
    monkeypatch.setattr(stages.tts, "stream", _forbidden)
    result = Pipeline(cfg, stages).run(_wav(), "no-speech")
    assert result.record.gate.action == GateAction.REPEAT
    assert result.record.status == "no_speech" and not result.out_wav.size


@pytest.mark.parametrize("span", [(-.1, .5), (.7, .2), (0, 2), (0, float("nan"))])
def test_invalid_vad_span_abstains(monkeypatch, span):
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg)
    monkeypatch.setattr(stages.vad, "segment", lambda x: span)
    monkeypatch.setattr(stages.tts, "stream", _forbidden)
    result = Pipeline(cfg, stages).run(_wav(), "invalid-span")
    assert result.record.error_stage == "vad" and not result.out_wav.size


@pytest.mark.parametrize("branch,method", [("asr", "transcribe"), ("branch_b", "analyze")])
def test_branch_timeout_returns_promptly_and_retires_pipeline(monkeypatch, branch, method):
    cfg = PipelineConfig(branch_timeout_s=.05, turn_timeout_s=.5)
    stages = build_stub_stages(cfg)
    release = Event()
    original = getattr(getattr(stages, branch), method)

    def blocked(*args):
        release.wait(2)
        return original(*args)

    monkeypatch.setattr(getattr(stages, branch), method, blocked)
    monkeypatch.setattr(stages.tts, "stream", _forbidden)
    pipeline = Pipeline(cfg, stages)
    start = time.monotonic()
    try:
        result = pipeline.run(_wav(), "timeout")
        assert time.monotonic() - start < .5
        assert result.record.status == "timeout" and not result.out_wav.size
        frozen = dict(result.record.stage_ms)
        next_turn = pipeline.run(_wav(), "next")
        assert next_turn.record.error_type == "RetiredAfterTimeout"
        assert next_turn.record.gate.action == GateAction.ABSTAIN
    finally:
        release.set()
    time.sleep(.03)
    assert result.record.stage_ms == frozen


def test_tts_timeout_discards_partial_audio(monkeypatch):
    cfg = PipelineConfig(turn_timeout_s=.08)
    stages = build_stub_stages(cfg)
    release = Event()

    def blocked_tts(*args):
        yield np.ones(20, np.float32) * .1
        release.wait(2)

    monkeypatch.setattr(stages.tts, "stream", blocked_tts)
    try:
        result = Pipeline(cfg, stages).run(_wav(), "tts-timeout")
        assert result.record.status == "timeout" and result.record.error_stage == "tts"
        assert result.record.endpoint_to_first_audio_ms is None and not result.out_wav.size
    finally:
        release.set()


def test_empty_asr_and_truncated_translation_never_reach_tts(monkeypatch):
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg)
    stages.gate = Gate()
    monkeypatch.setattr(stages.asr, "transcribe", lambda *a: AsrResult(text="", lang="vi", confidence=0))
    monkeypatch.setattr(stages.nmt, "translate", _forbidden)
    monkeypatch.setattr(stages.tts, "stream", _forbidden)
    result = Pipeline(cfg, stages).run(_wav(), "empty-asr")
    assert result.record.gate.action == GateAction.REPEAT

    stages = build_stub_stages(cfg)
    monkeypatch.setattr(stages.nmt, "translate", lambda *a: MtResult(src_text=a[0], tgt_text="truncated",
                                                                      src_lang="vi", tgt_lang="en", truncated=True))
    monkeypatch.setattr(stages.tts, "stream", _forbidden)
    result = Pipeline(cfg, stages).run(_wav(), "truncated")
    assert result.record.error_type == "TruncatedTranslation" and not result.out_wav.size


def _hanging_worker(connection, *args):
    """Inject a non-returning native-style process for the hard watchdog test."""
    connection.send(("ready", "fault-injection"))
    while True:
        time.sleep(60)


def _crashed_worker(connection, *args):
    os._exit(7)


@pytest.mark.parametrize("target,expected", [(_hanging_worker, "timeout"), (_crashed_worker, "error")])
def test_process_watchdog_reaps_hung_or_crashed_worker(monkeypatch, target, expected):
    import tonebridge.runtime as runtime
    before = {p.pid for p in mp.active_children()}
    monkeypatch.setattr(runtime, "_worker", target)
    cfg = PipelineConfig(model_load_timeout_s=10, turn_timeout_s=.05).prepare("stub")
    result, audit = runtime.run_isolated(cfg, "stub", _wav(), "process-failure")
    assert result.record.status == expected and not result.out_wav.size
    assert audit is None
    assert {p.pid for p in mp.active_children()} == before


def test_worker_start_failure_returns_safe_record(monkeypatch):
    import tonebridge.runtime as runtime
    monkeypatch.setattr(mp.context.SpawnProcess, "start", _fail)
    cfg = PipelineConfig().prepare("stub")
    result, audit = runtime.run_isolated(cfg, "stub", _wav(), "start-failure")
    assert result.record.error_stage == "worker_start"
    assert result.record.status == "error" and not result.out_wav.size
