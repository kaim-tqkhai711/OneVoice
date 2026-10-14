"""Benchmark distributions, persistent stages, missing assets and process faults."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

from tonebridge.benchmark import distribution, run_benchmark, TreeRssSampler
from tonebridge.config import PipelineConfig, ROOT


def test_latency_distribution_and_missing_samples():
    summary = distribution(range(1, 101))
    assert summary == {"n": 100, "min": 1., "max": 100., "p50": 50.5, "p95": 95.05}
    assert distribution([None])["p95"] is None
    with pytest.raises(ValueError): distribution([float("nan")])


def test_rss_is_measured_in_binary_units():
    sampler = TreeRssSampler()
    sampler.start(); result = sampler.finish()
    assert result["peak_bytes"] > 0 and result["samples"] > 0
    assert result["peak_mib"] == result["peak_bytes"] / 2**20
    assert result["peak_gib"] == result["peak_mib"] / 1024


def test_persistent_stub_benchmark_separates_cold_and_warm_and_keeps_privacy():
    cfg = PipelineConfig().prepare("stub")
    before = {process.pid for process in mp.active_children()}
    report = run_benchmark(cfg, "stub", [np.ones(16000, np.float32) * .1], turns=3)
    assert report["status"] == "COMPLETE" and report["audio_turns"] == 3
    assert report["cold_turn"]["index"] == 0 and report["warm"]["total_ms"]["n"] == 2
    assert report["load"]["startup_ms"] >= report["load"]["model_load_ms"]
    assert report["cold_turn"]["cold_e2e_ms"] >= report["load"]["startup_ms"]
    assert report["audible_playback_ms"] is None
    assert all("asr_text" not in row and "nmt_text" not in row for row in report["rows"])
    assert {process.pid for process in mp.active_children()} == before


def _hang(connection, cfg, *args):
    connection.send(("ready", {"config_hash": cfg.config_hash(), "model_load_ms": 0, "capabilities": {}}))
    while True: time.sleep(10)


def _crash(connection, *args):
    os._exit(7)


@pytest.mark.parametrize("worker,error", [(_hang, "BenchmarkTimeout"), (_crash, "BenchmarkWorkerExited")])
def test_failed_benchmark_is_incomplete_and_worker_reaped(monkeypatch, worker, error):
    import tonebridge.benchmark as benchmark
    monkeypatch.setattr(benchmark, "_worker", worker)
    cfg = PipelineConfig(model_load_timeout_s=10, turn_timeout_s=.05).prepare("stub")
    before = {process.pid for process in mp.active_children()}
    report = run_benchmark(cfg, "stub", [np.ones(16000, np.float32) * .1], turns=3)
    assert report["status"] == "INCOMPLETE" and report["error"] == error
    assert report["processed_turns"] == 0 and report["warm"]["total_ms"]["n"] == 0
    assert {process.pid for process in mp.active_children()} == before


def test_invalid_turn_is_not_counted_as_success():
    report = run_benchmark(PipelineConfig().prepare("stub"), "stub", [np.empty(0, np.float32)], turns=3)
    assert report["status"] == "INCOMPLETE" and report["successful_turns"] == 0 and report["audio_turns"] == 0


def test_real_benchmark_fails_preflight_without_fallback(tmp_path):
    configs = tmp_path / "configs"; configs.mkdir()
    for name in ("pipeline.json", "runtime_files.json", "safety_lexicon_vi_en.json"):
        shutil.copyfile(ROOT / "configs" / name, configs / name)
    out = tmp_path / "blocked.json"
    result = subprocess.run([sys.executable, str(ROOT / "tools/bench_e2e.py"), "--mode", "branch-a-bench",
        "--config", str(configs / "pipeline.json"), "--turns", "100", "--out", str(out)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    report = json.loads(out.read_text())
    assert report["status"] == "BLOCKED_PREFLIGHT" and report["processed_turns"] == 0
    assert report["missing_assets"] == report["assets_checked"] == 368
    assert report["latency"] is None and report["rss"] is None and report["mode"] == "branch-a-bench"


def test_cli_rejects_too_small_benchmark(tmp_path):
    out = tmp_path / "small.json"
    result = subprocess.run([sys.executable, str(ROOT / "tools/bench_e2e.py"), "--mode", "stub", "--turns", "5", "--out", str(out)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and not out.exists()
