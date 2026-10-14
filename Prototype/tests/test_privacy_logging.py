"""PHI sentinels must not cross default serialization, JSONL or CLI boundaries."""
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
from tonebridge.telemetry import JsonlLogger, PrivacyLogError

SECRET = "PHI_SENTINEL_patient_8675309"


def record():
    result = Pipeline(PipelineConfig(), build_stub_stages(PipelineConfig())).run(
        (.1 * np.sin(np.arange(16000) * .1)).astype(np.float32), SECRET, session_id=SECRET)
    row = result.record
    row.asr_text = row.nmt_text = SECRET
    row.gate = GateDecision(action=GateAction.CONFIRM, reasons=[SECRET], speak_text=SECRET)
    return row


def test_default_dump_and_logger_allowlist_drop_all_text(tmp_path):
    row = record()
    assert SECRET not in row.model_dump_json()
    assert "asr_text" not in row.model_dump() and "speak_text" not in row.model_dump()["gate"]
    raw = row.model_dump(mode="json", context={"eval_mode": True})
    raw.update(error_type=SECRET, error_stage=SECRET, extra={"transcript": SECRET}, capabilities={SECRET: SECRET})
    raw["stage_ms"][SECRET] = 1
    raw["offline_audit"] = {"net_attempts_python": 1, "net_conns_os_seen": 0, "attempt_detail": [SECRET]}
    log = tmp_path / "default.jsonl"
    projected = JsonlLogger(log).write(raw)
    text = log.read_text(encoding="utf8")
    assert SECRET not in text and SECRET not in json.dumps(projected)
    assert projected["error_type"] == "StageError"
    assert set(projected["gate"]) == {"action"}
    assert projected["offline_audit"]["net_attempts_python"] == 1


def test_eval_mode_requires_explicit_opt_in_and_logs_cannot_mix(tmp_path):
    log = tmp_path / "eval.jsonl"
    JsonlLogger(log, eval_mode=True).write(record().model_dump(mode="json", context={"eval_mode": True}))
    assert SECRET in log.read_text(encoding="utf8")
    before = log.read_bytes()
    with pytest.raises(PrivacyLogError):
        JsonlLogger(log)
    assert log.read_bytes() == before
    safe = tmp_path / "runtime.jsonl"
    JsonlLogger(safe).write(record().model_dump())
    with pytest.raises(PrivacyLogError):
        JsonlLogger(safe, eval_mode=True)


def test_safe_milestone_b_metadata_log_remains_appendable(tmp_path):
    row = record().model_dump(mode="json")
    for key in ("peak_rss_mib", "rss_unit", "endpoint_to_first_delivery_ms", "urgency_status", "cue_enabled"):
        row.pop(key, None)
    path = tmp_path / "old-metadata.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf8")
    JsonlLogger(path).write(record().model_dump())
    assert len(path.read_text(encoding="utf8").splitlines()) == 2
    assert SECRET not in path.read_text(encoding="utf8")


@pytest.mark.parametrize("old", ['{"asr_text":"' + SECRET + '"}\n', SECRET, '{"logging_policy":"metadata-v1","asr_text":"' + SECRET + '"}\n'])
def test_existing_unsafe_or_corrupt_log_fails_closed(tmp_path, old):
    log = tmp_path / "legacy.jsonl"
    log.write_text(old, encoding="utf8")
    with pytest.raises(PrivacyLogError):
        JsonlLogger(log)


@pytest.mark.parametrize("stage,method", [("asr", "transcribe"), ("nmt", "translate"), ("safety", "check"), ("branch_b", "analyze"), ("tts", "stream")])
def test_stage_error_messages_do_not_leak_in_default_log(monkeypatch, tmp_path, stage, method):
    stages = build_stub_stages(PipelineConfig())
    def broken(*args):
        raise ValueError(SECRET)
    monkeypatch.setattr(getattr(stages, stage), method, broken)
    result = Pipeline(PipelineConfig(), stages).run((.1 * np.sin(np.arange(16000) * .1)).astype(np.float32), SECRET)
    assert result.record.status == "error" and not result.out_wav.size
    log = tmp_path / "errors.jsonl"
    JsonlLogger(log).write(result.record.model_dump(mode="json", context={"eval_mode": True}))
    assert SECRET not in log.read_text(encoding="utf8")


def invoke(tmp_path, *options):
    inp = tmp_path / (SECRET + ".wav")
    wavfile.write(inp, 16000, (.1 * np.sin(np.arange(16000) * .1)).astype(np.float32))
    inp.with_suffix(".txt").write_text(SECRET, encoding="utf8")
    log = tmp_path / "turns.jsonl"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf8"}
    result = subprocess.run([sys.executable, "-m", "tonebridge.cli", "--mode", "stub", "--wav", str(inp),
        "--out", str(tmp_path / "out.wav"), "--log", str(log), *options], cwd=ROOT, env=env,
        capture_output=True, text=True, encoding="utf8", timeout=30)
    return result, log


def test_cli_default_stdout_stderr_and_file_are_metadata_only(tmp_path):
    result, log = invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    assert SECRET not in result.stdout + result.stderr + log.read_text(encoding="utf8")
    row = json.loads(result.stdout)
    assert row == json.loads(log.read_text(encoding="utf8"))
    assert "asr_text" not in row and "nmt_text" not in row and "speak_text" not in row["gate"]


def test_cli_eval_mode_keeps_text_only_when_requested(tmp_path):
    result, log = invoke(tmp_path, "--eval-mode")
    assert result.returncode == 0, result.stderr
    assert SECRET in result.stdout and SECRET in log.read_text(encoding="utf8")
    assert SECRET not in result.stderr


@pytest.mark.parametrize("failure", ["config", "argument", "invalid_audio", "log"])
def test_cli_errors_never_echo_private_values(tmp_path, failure):
    flags = []
    if failure == "config":
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"direction": SECRET}), encoding="utf8")
        flags = ["--config", str(bad)]
    elif failure == "argument":
        flags = ["--mode", SECRET]
    elif failure == "invalid_audio":
        flags = ["--wav", str(tmp_path / (SECRET + "missing.wav"))]
    else:
        log = tmp_path / "legacy.jsonl"
        log.write_text(json.dumps({"asr_text": SECRET}), encoding="utf8")
        flags = ["--log", str(log)]
    result, log = invoke(tmp_path, *flags)
    assert result.returncode != 0
    assert SECRET not in result.stdout + result.stderr
    if log.exists():
        assert SECRET not in log.read_text(encoding="utf8")


def test_native_and_python_worker_output_suppressed():
    script = """
import os,sys
from tonebridge.runtime import quiet_worker_output
with quiet_worker_output():
    print('PHI_SENTINEL_patient_8675309',flush=True)
    print('PHI_SENTINEL_patient_8675309',file=sys.stderr,flush=True)
    os.write(1,b'PHI_SENTINEL_patient_8675309')
    os.write(2,b'PHI_SENTINEL_patient_8675309')
    if os.name=='nt':
        import ctypes
        k=ctypes.WinDLL('kernel32');k.GetStdHandle.restype=ctypes.c_void_p
        k.WriteFile.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong,ctypes.c_void_p,ctypes.c_void_p]
        data=ctypes.create_string_buffer(b'PHI_SENTINEL_patient_8675309')
        k.WriteFile(k.GetStdHandle(0xFFFFFFF5),data,len(data)-1,None,None)
print('safe-restored',flush=True)
"""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert SECRET.encode() not in result.stdout + result.stderr
    assert b"safe-restored" in result.stdout
