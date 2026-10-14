"""Preflight must distinguish environment failures from missing model packs."""
import json
import os
import subprocess
import sys

from tonebridge.config import ROOT
from tonebridge.preflight import check_dependencies, requirements


def test_requirement_profiles_include_missing_imports():
    runtime = requirements(ROOT / "requirements-runtime.txt")
    dev = requirements(ROOT / "requirements-dev.txt")
    bench = requirements(ROOT / "requirements-bench.txt")
    assert {"sherpa-onnx", "swift-f0", "sentencepiece"} <= runtime.keys()
    assert {"librosa", "soundfile", "jiwer", "soxr", "pytest", "onnx", "torch"} <= dev.keys()
    assert {"optimum", "transformers", "sacremoses"} <= bench.keys()
    assert "torch" not in runtime and "librosa" not in runtime


def test_version_drift_and_import_failure_reported(monkeypatch):
    import tonebridge.preflight as preflight
    monkeypatch.setattr(preflight.importlib.metadata, "version", lambda name: "0.0")
    result = check_dependencies("stub")
    assert len(result["errors"]) == 4
    assert all("version mismatch" in error for error in result["errors"])


def test_preflight_script_stub_and_missing_config_exit_codes(tmp_path):
    env = {**os.environ, "PYTHONIOENCODING": "utf8"}
    command = [sys.executable, str(ROOT / "tools/check_runtime.py"), "--mode", "stub", "--allow-version-drift"]
    good = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert good.returncode == 0 and json.loads(good.stdout)["assets_checked"] == 0
    bad = subprocess.run(command + ["--config", str(tmp_path / "missing.json")], cwd=tmp_path,
                         env=env, capture_output=True, text=True, timeout=30)
    assert bad.returncode == 1 and not json.loads(bad.stdout)["ok"]
