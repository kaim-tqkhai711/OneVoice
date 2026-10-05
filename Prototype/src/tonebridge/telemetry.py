"""Latency / RSS / RTF instrumentation. Writes one JSONL row per turn.

Latency definition (fixed, Proposal section 1.3): PTT release -> first TTS audio sample.
``TurnClock.mark("endpoint")`` is called when the PTT is released (audio fully buffered);
``mark("first_audio")`` when the first TTS chunk is ready.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import psutil


class TurnClock:
    """Per-turn stage timer. One writer per stage key (stages run in separate threads)."""

    def __init__(self) -> None:
        self.stage_ms: dict[str, float] = {}
        self._marks: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stage_ms[name] = (time.perf_counter() - t0) * 1000.0

    def mark(self, name: str) -> None:
        self._marks[name] = time.perf_counter()

    def between_ms(self, a: str, b: str) -> float | None:
        if a not in self._marks or b not in self._marks:
            return None
        return (self._marks[b] - self._marks[a]) * 1000.0


def peak_rss_mb() -> float:
    """Peak resident set of this process in MB (Windows peak_wset, Linux/Android ru_maxrss)."""
    mi = psutil.Process(os.getpid()).memory_info()
    if hasattr(mi, "peak_wset"):
        return mi.peak_wset / 2**20
    import resource  # POSIX only

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB -> MB on Linux


def rtf(processing_ms: float, audio_s: float) -> float:
    """Real-time factor = processing time / audio duration."""
    return (processing_ms / 1000.0) / max(audio_s, 1e-9)


def bootstrap_ci(x: np.ndarray, q: float, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap 95% CI of the q-th percentile. x: [N] -> (lo, hi)."""
    rng = np.random.default_rng(seed)
    stats = [np.percentile(rng.choice(x, size=len(x), replace=True), q) for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarize_latency(ms: list[float]) -> dict:
    """p50/p95 with bootstrap CI. ms: [N] -> dict. Always report ``n`` next to the percentiles."""
    x = np.asarray(ms, dtype=np.float64)  # [N]
    return {"n": int(x.size), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "p50_ci95": list(bootstrap_ci(x, 50)), "p95_ci95": list(bootstrap_ci(x, 95))}


class JsonlLogger:
    """Append-only JSONL writer."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
