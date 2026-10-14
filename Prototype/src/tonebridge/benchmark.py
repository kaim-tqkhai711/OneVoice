"""Measured E2E turns in one persistent, killable worker; metadata-only reports.

Cold start includes spawn/imports/session load and first inference. Warm samples
reuse exactly the same stages. RSS is sampled for the parent + descendants, not
estimated by adding isolated stage peaks; shared resident pages can be counted
in multiple processes. No readiness timing is called audible playback latency.
"""
from __future__ import annotations

from collections import Counter
import multiprocessing as mp
import platform
from threading import Event, Thread
import time

import numpy as np
import psutil

from tonebridge.config import PipelineConfig


def distribution(values) -> dict:
    """Finite samples only, always carry n and explicit min/max/p50/p95."""
    data = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if data.size and not np.isfinite(data).all():
        raise ValueError("nonfinite benchmark measurements")
    if not data.size:
        return {"n": 0, "min": None, "max": None, "p50": None, "p95": None}
    return {"n": int(data.size), "min": float(data.min()), "max": float(data.max()),
            "p50": float(np.percentile(data, 50)), "p95": float(np.percentile(data, 95))}


class TreeRssSampler:
    """Sample the actual resident bytes of the benchmark parent and its workers."""
    def __init__(self, interval_s: float = .01):
        self.interval_s = interval_s
        self.peak_bytes = 0
        self.samples = 0
        self.stop_event = Event()
        self.thread = Thread(target=self._monitor, daemon=True)
        self.process = psutil.Process()

    def sample(self):
        total = 0
        for process in [self.process, *self.process.children(recursive=True)]:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.peak_bytes = max(self.peak_bytes, total)
        self.samples += 1

    def _monitor(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval_s)

    def start(self):
        self.sample()
        self.thread.start()

    def finish(self) -> dict:
        self.stop_event.set()
        self.thread.join(timeout=2)
        return {"scope": "parent_plus_descendants_resident_sum", "peak_bytes": self.peak_bytes,
            "peak_mib": self.peak_bytes / 2**20, "peak_gib": self.peak_bytes / 2**30,
            "sample_interval_ms": self.interval_s * 1000, "samples": self.samples,
            "caveat": "Sampling may miss short peaks; shared resident pages may be counted per process."}


def _worker(connection, cfg, mode, inputs, turns):
    from tonebridge.runtime import quiet_worker_output
    with quiet_worker_output():
        try:
            from tonebridge.stages.factory import build_stages
            from tonebridge.pipeline import Pipeline
            began = time.perf_counter()
            stages = build_stages(cfg, mode)
            pipeline = Pipeline(cfg, stages)
            connection.send(("ready", {"model_load_ms": (time.perf_counter() - began) * 1000,
                "config_hash": pipeline.cfg.config_hash(), "capabilities": stages.capabilities}))
            for index in range(turns):
                first = True
                def on_audio(chunk, sample_rate):
                    nonlocal first
                    if first:
                        first = False
                        connection.send(("first_buffer", index))
                connection.send(("started", index))
                result = pipeline.run(inputs[index % len(inputs)], str(index), on_audio=on_audio)
                row = result.record.model_dump(mode="json")
                row.update(index=index, input_index=index % len(inputs),
                    output_samples=int(result.out_wav.size), worker_rss_mib=psutil.Process().memory_info().rss / 2**20)
                connection.send(("turn", row))
                if row["status"] in ("error", "timeout", "invalid_audio"):
                    break
            connection.send(("finished",))
        except BaseException:
            connection.send(("error", "BenchmarkWorkerError"))
        finally:
            connection.close()


def run_benchmark(cfg: PipelineConfig, mode: str, inputs: list[np.ndarray], turns: int = 100) -> dict:
    """Measure, report every completed/blocked/failing turn, and reap the worker.

    Public CLI enforces >=50 turns; smaller calls exist for fault tests. A run is
    complete only if all requested turns succeed; stub/branch-a-bench additionally
    require synthesized output for every turn. Full quality-blocked turns are
    counted explicitly, with no fabricated first-audio samples.
    """
    from tonebridge.runtime import _stop_process
    if not inputs or turns < 1:
        raise ValueError("nonempty inputs and positive turn count required")
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    worker = context.Process(target=_worker, args=(child, cfg, mode, inputs, turns), name="tonebridge-e2e")
    rss = TreeRssSampler()
    rows, load, first_buffers = [], None, {}
    error = None
    started = time.perf_counter()
    deadline = started + cfg.model_load_timeout_s
    rss.start()
    launched = False
    try:
        worker.start()
        launched = True
        child.close()
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or not parent.poll(max(0., remaining)):
                error = "BenchmarkTimeout"
                break
            message = parent.recv()
            event = message[0]
            now = time.perf_counter()
            if event == "ready":
                load = message[1]
                load["startup_ms"] = (now - started) * 1000
                if load["config_hash"] != cfg.config_hash():
                    error = "FingerprintChangedAtLoad"
                    break
                deadline = now + cfg.turn_timeout_s + 1
            elif event == "started":
                deadline = now + cfg.turn_timeout_s + 1
            elif event == "first_buffer":
                first_buffers[message[1]] = (now - started) * 1000
            elif event == "turn":
                row = message[1]
                if not rows:
                    row["cold_e2e_ms"] = (now - started) * 1000
                    row["cold_first_buffer_received_ms"] = first_buffers.get(0)
                rows.append(row)
                deadline = now + cfg.turn_timeout_s + 1
            elif event == "error":
                error = message[1]
                break
            elif event == "finished":
                break
    except (EOFError, OSError, RuntimeError):
        error = "BenchmarkWorkerExited"
    finally:
        parent.close()
        child.close()
        try:
            if launched:
                _stop_process(worker)
            else:
                worker.close()
        finally:
            memory = rss.finish()
    success = [row for row in rows if row["status"] == "ok"]
    audio = [row for row in success if row["output_samples"] > 0]
    complete = error is None and len(success) == turns and (mode == "full" or len(audio) == turns)
    keys = ("total_ms", "endpoint_to_first_audio_ms", "endpoint_to_first_delivery_ms", "rtf", "worker_rss_mib")
    warm = [row for row in success if row["index"] > 0]
    return {"schema_version": "tonebridge-e2e-v1", "status": "COMPLETE" if complete else "INCOMPLETE",
        "mode": mode, "config_hash": cfg.config_hash(), "requested_turns": turns, "processed_turns": len(rows),
        "successful_turns": len(success), "audio_turns": len(audio), "input_count": len(inputs),
        "gate_counts": dict(Counter(row["gate"]["action"] for row in rows)), "error": error,
        "load": load, "cold_turn": rows[0] if rows else None,
        "warm": {key: distribution(row[key] for row in warm) for key in keys},
        "all_turns": {key: distribution(row[key] for row in success) for key in keys},
        "rss": memory, "hardware": {"os": platform.system(), "architecture": platform.machine(),
            "processor": platform.processor(), "logical_cpus": psutil.cpu_count(), "physical_memory_gib": psutil.virtual_memory().total / 2**30},
        "audible_playback_ms": None, "playback": "not_measured_no_audio_device_or_loopback",
        "measurement": "wall clock E2E; first-buffer readiness and IPC submission, not audible onset",
        "rows": rows}
