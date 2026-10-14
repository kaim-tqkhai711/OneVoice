"""First-clause readiness/delivery while retaining full-translation approval."""
from threading import Event, Thread
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from tonebridge.config import PipelineConfig
from tonebridge.contracts import GateAction, GateDecision
from tonebridge.pipeline import Pipeline
from tonebridge.stages.factory import build_stub_stages
from tonebridge.stages.tts_piper import PiperEn, split_clauses


def test_piper_yields_first_clause_before_synthesizing_remainder(monkeypatch):
    tts = PiperEn.__new__(PiperEn)
    tts._chunk, tts.first_clause = 2, True
    calls = []
    def synth(text):
        calls.append(text)
        return np.full(4, .1 * len(calls), np.float32)
    monkeypatch.setattr(tts, "synth", synth)
    text = "Take 0.5 mg of aspirin, and call the doctor if needed."
    stream = tts.stream(text, "en")
    first = next(stream)
    assert calls == ["Take 0.5 mg of aspirin,"] and np.allclose(first, .1)
    next(stream)
    assert len(calls) == 1
    chunks = [first, np.full(2, .1, np.float32), *list(stream)]
    assert calls == split_clauses(text) and np.concatenate(chunks).size == 8


def test_whole_utterance_baseline_is_explicit(monkeypatch):
    tts = PiperEn.__new__(PiperEn)
    tts._chunk, tts.first_clause = 2, False
    calls = []
    monkeypatch.setattr(tts, "synth", lambda text: calls.append(text) or np.ones(4, np.float32) * .1)
    text = "Take two tablets, and call the doctor."
    assert len(list(tts.stream(text, "en"))) == 2 and calls == [text]


def test_pipeline_delivers_before_remaining_audio_finishes(monkeypatch):
    cfg = PipelineConfig(turn_timeout_s=3)
    stages = build_stub_stages(cfg)
    delivered, release, finished = Event(), Event(), Event()
    def stream(*args):
        yield np.full(10, .1, np.float32)
        release.wait(2)
        yield np.full(10, .2, np.float32)
    monkeypatch.setattr(stages.tts, "stream", stream)
    chunks, results = [], []
    def consume(chunk, sr):
        chunks.append(chunk)
        assert sr == 22050
        delivered.set()
    def run():
        results.append(Pipeline(cfg, stages).run(np.ones(16000, np.float32) * .1, "stream", on_audio=consume))
        finished.set()
    thread = Thread(target=run)
    thread.start()
    try:
        assert delivered.wait(1) and not finished.is_set()
    finally:
        release.set(); thread.join(timeout=3)
    assert not thread.is_alive() and len(chunks) == 2
    assert results[0].out_wav.size == 20
    assert results[0].record.endpoint_to_first_delivery_ms is not None


def test_gate_blocks_all_live_delivery(monkeypatch):
    stages = build_stub_stages(PipelineConfig())
    stages.gate = lambda *args: GateDecision(action=GateAction.CONFIRM)
    monkeypatch.setattr(stages.tts, "stream", lambda *args: pytest.fail("unsafe text reached TTS"))
    result = Pipeline(PipelineConfig(), stages).run(np.ones(16000, np.float32) * .1, "blocked",
        on_audio=lambda *args: pytest.fail("unsafe audio delivered"))
    assert not result.out_wav.size and result.record.endpoint_to_first_delivery_ms is None


def test_late_error_discards_buffered_audio_and_reports_live_partial_delivery(monkeypatch):
    stages = build_stub_stages(PipelineConfig())
    def broken(*args):
        yield np.full(10, .1, np.float32)
        raise ValueError("failure after approved first clause")
    monkeypatch.setattr(stages.tts, "stream", broken)
    delivered = []
    result = Pipeline(PipelineConfig(), stages).run(np.ones(16000, np.float32) * .1, "late",
        on_audio=lambda chunk, sr: delivered.append(chunk))
    assert result.record.status == "error" and not result.out_wav.size
    assert len(delivered) == 1  # live audio cannot be retrospectively unplayed
    assert result.record.endpoint_to_first_audio_ms is None


def test_playback_sink_drains_success_and_aborts_failure(monkeypatch):
    from tonebridge.playback import StreamingPlayer
    calls = []
    class Device:
        def __init__(self, **kwargs): calls.append(("open", kwargs))
        def start(self): calls.append("start")
        def write(self, data):
            assert data.shape == (10, 1) and data.dtype == np.float32
            calls.append("write"); return False
        def stop(self): calls.append("stop")
        def abort(self): calls.append("abort")
        def close(self): calls.append("close")
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=Device))
    for abort in (False, True):
        player = StreamingPlayer()
        assert player.stream is None
        player.write(np.ones(10, np.float32) * .1, 22050)
        player.close(abort=abort)
        assert calls[-2:] == ["abort" if abort else "stop", "close"]


def test_first_clause_setting_changes_effective_fingerprint():
    cfg = PipelineConfig().prepare("stub")
    before = cfg.config_hash()
    cfg.decode.tts_first_clause = False
    assert cfg.config_hash() != before


def test_empty_first_clause_cannot_silently_speak_only_the_remainder(monkeypatch):
    tts = PiperEn.__new__(PiperEn)
    tts._chunk, tts.first_clause = 2, True
    calls = []
    def synth(text):
        calls.append(text)
        return np.empty(0, np.float32) if len(calls) == 1 else np.ones(4, np.float32) * .1
    monkeypatch.setattr(tts, "synth", synth)
    with pytest.raises(ValueError, match="first-clause"):
        list(tts.stream("Take two tablets, and call the doctor.", "en"))
    assert calls == ["Take two tablets,"]
