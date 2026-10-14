"""CLI entry-point acceptance, including the spawned offline worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
import pytest
from scipy.io import wavfile

from tonebridge.config import ROOT


def invoke(tmp_path, *flags, data=None):
    inp = tmp_path / "input.wav"
    if data is None:
        data = (.1 * np.sin(np.arange(16000) * .1) * 32767).astype(np.int16)
    wavfile.write(inp, 16000, data)
    out, log = tmp_path / "output.wav", tmp_path / "turns.jsonl"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf8"}
    result = subprocess.run([sys.executable, "-m", "tonebridge.cli", "--wav", str(inp),
                             "--out", str(out), "--log", str(log), *flags],
                            cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf8", timeout=30)
    return result, out, log


def test_mode_required_and_old_flags_rejected(tmp_path):
    result, out, log = invoke(tmp_path)
    assert result.returncode == 2 and "--mode" in result.stderr
    assert not out.exists() and not log.exists()


def test_missing_config_stops_before_inference(tmp_path):
    result, out, log = invoke(tmp_path, "--mode", "stub", "--config", str(tmp_path / "missing.json"))
    assert result.returncode == 2 and "configuration error" in result.stderr
    assert not out.exists() and not log.exists()


def test_stub_startup_and_turn_are_explicit(tmp_path):
    result, out, log = invoke(tmp_path, "--mode", "stub")
    assert result.returncode == 0, result.stderr
    startup = json.loads(result.stderr.splitlines()[0])
    assert startup["mode"] == "stub" and "stub" in startup["capabilities"]["tts"]
    row = json.loads(log.read_text(encoding="utf8"))
    assert row["mode"] == "stub" and row["status"] == "ok" and out.exists()


def test_full_missing_assets_fails_without_fallback(tmp_path):
    # Keep testing fail-closed startup after the real checkout is provisioned.
    configs = tmp_path / "configs"
    configs.mkdir()
    for name in ("pipeline.json", "runtime_files.json", "safety_lexicon_vi_en.json"):
        shutil.copyfile(ROOT / "configs" / name, configs / name)
    result, out, log = invoke(tmp_path, "--mode", "full", "--config", str(configs / "pipeline.json"))
    assert result.returncode == 2 and "missing asset" in result.stderr
    assert not out.exists() and not log.exists()
    assert '"mode": "full"' in result.stderr


def test_silent_and_overlong_cli_never_write_audio(tmp_path):
    result, out, log = invoke(tmp_path, "--mode", "stub", data=np.full(16000, 128, np.uint8))
    assert result.returncode == 0 and not out.exists()
    assert json.loads(log.read_text())["status"] == "no_speech"
    result, out, log = invoke(tmp_path, "--mode", "stub", data=np.zeros(16 * 16000, np.int32))
    assert result.returncode == 1 and not out.exists()
    assert json.loads(log.read_text().splitlines()[-1])["status"] == "invalid_audio"


def test_spawned_stub_offline_has_no_live_children_or_network(tmp_path):
    result, out, log = invoke(tmp_path, "--mode", "stub", "--assert-offline")
    assert result.returncode == 0, result.stderr + result.stdout
    row = json.loads(log.read_text(encoding="utf8"))
    assert row["mode"] == "stub" and out.exists()
    assert row["offline_audit"]["net_attempts_python"] == 0
    assert row["offline_audit"]["net_conns_os_seen"] == 0


def test_output_io_failure_is_controlled_exit(tmp_path):
    result, out, log = invoke(tmp_path, "--mode", "stub", "--log", str(tmp_path))
    assert result.returncode == 2 and "cannot write turn output" in result.stderr
    assert "Traceback" not in result.stderr and not out.exists()
