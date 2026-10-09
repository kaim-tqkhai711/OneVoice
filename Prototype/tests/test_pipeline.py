import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from tonebridge.config import PipelineConfig
from tonebridge.contracts import GateAction, UrgencyResult
from tonebridge.pipeline import Pipeline
from tonebridge.cli import build_stub_stages


class MarkerDenoiser:
    """Adds a constant so we can tell whether downstream stages saw denoised audio."""

    def process(self, wav):
        return wav + 0.25


class SpyBranchB:
    def __init__(self):
        self.seen = None

    def analyze(self, wav):
        self.seen = wav.copy()
        return UrgencyResult(label="UNKNOWN")


class SpyAsr:
    def __init__(self):
        self.seen = None

    def transcribe(self, wav, lang):
        from tonebridge.contracts import AsrResult
        self.seen = wav.copy()
        return AsrResult(text="x", lang=lang, confidence=0.9)


def _wav(n=16000):
    return (0.1 * np.sin(np.linspace(0, 200, n))).astype(np.float32)  # [n]


def test_branch_b_never_sees_denoised_audio():
    cfg = PipelineConfig()
    st = build_stub_stages(cfg, None)
    st.denoiser, st.branch_b, st.asr = MarkerDenoiser(), SpyBranchB(), SpyAsr()
    wav = _wav()
    Pipeline(cfg, st).run(wav, "u1")
    assert np.allclose(st.branch_b.seen, wav)  # raw-ish
    assert np.allclose(st.asr.seen, wav + 0.25)  # Branch A got the denoised audio


def test_turn_record_has_stages_and_latency():
    cfg = PipelineConfig()
    res = Pipeline(cfg, build_stub_stages(cfg, None)).run(_wav(), "u2")
    r = res.record
    assert {"vad", "asr", "nmt", "safety", "branch_b", "gate", "tts"} <= set(r.stage_ms)
    assert r.endpoint_to_first_audio_ms is not None and r.endpoint_to_first_audio_ms >= 0
    assert r.gate.action == GateAction.SPEAK and res.out_wav.size > 0
    assert r.peak_rss_mb > 0 and r.rtf >= 0


def test_low_asr_confidence_gives_repeat_and_no_audio():
    cfg = PipelineConfig()
    st = build_stub_stages(cfg, None)

    class WeakAsr:
        def transcribe(self, wav, lang):
            from tonebridge.contracts import AsrResult
            return AsrResult(text="?", lang=lang, confidence=0.1)

    st.asr = WeakAsr()
    res = Pipeline(cfg, st).run(_wav(), "u3")
    assert res.record.gate.action == GateAction.REPEAT
    assert res.out_wav.size == 0 and res.record.endpoint_to_first_audio_ms is None


def test_cli_end_to_end(tmp_path):
    inp = tmp_path / "in.wav"
    wavfile.write(inp, 16000, (_wav() * 32767).astype(np.int16))
    inp.with_suffix(".txt").write_text("Bạn có bị đau ngực không?", encoding="utf-8")
    out, log = tmp_path / "out.wav", tmp_path / "turns.jsonl"
    root = Path(__file__).resolve().parents[1]
    env = {"PYTHONPATH": str(root / "src"), "PYTHONIOENCODING": "utf-8", "SYSTEMROOT": "C:\\Windows"}
    p = subprocess.run([sys.executable, "-m", "tonebridge.cli", "--wav", str(inp), "--out", str(out),
                       "--config", str(root / "configs" / "pipeline.json"), "--log", str(log), "--stub", "--log-content"],
                       capture_output=True, text=True, encoding="utf-8", env=env, cwd=root)
    assert p.returncode == 0, p.stderr
    sr, y = wavfile.read(out)
    assert sr == 22050 and y.size > 0
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert row["asr_text"].startswith("Bạn") and row["gate"]["action"] == "SPEAK"
