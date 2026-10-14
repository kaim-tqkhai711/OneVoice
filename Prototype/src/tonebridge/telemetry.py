"""Latency / RSS / RTF instrumentation. Writes one JSONL row per turn.

Latency definition (fixed, Proposal section 1.3): PTT release -> first TTS audio sample.
``TurnClock.mark("endpoint")`` is called when the PTT is released (audio fully buffered);
``mark("first_audio")`` when the first TTS chunk is ready.
"""
from __future__ import annotations

import json
import os
import time
import math
import re
import sys
from threading import Lock
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
        self._lock = Lock()
        self._frozen = False

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            with self._lock:
                if not self._frozen:
                    self.stage_ms[name] = (time.perf_counter() - t0) * 1000.0

    def mark(self, name: str) -> None:
        with self._lock:
            if not self._frozen:
                self._marks[name] = time.perf_counter()

    def freeze(self) -> None:
        """Prevent late timeout workers from changing a completed turn record."""
        with self._lock:
            self._frozen = True

    def between_ms(self, a: str, b: str) -> float | None:
        if a not in self._marks or b not in self._marks:
            return None
        return (self._marks[b] - self._marks[a]) * 1000.0


def peak_rss_mb() -> float:
    """Legacy API name, returns MiB (binary 2**20), never decimal MB."""
    mi = psutil.Process(os.getpid()).memory_info()
    if hasattr(mi, "peak_wset"):
        return mi.peak_wset / 2**20
    import resource  # POSIX only

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (2**20 if sys.platform == "darwin" else 1024.0)


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


_STAGES = {"denoise", "vad", "asr", "nmt", "safety", "branch_b", "gate", "tts", "frontend",
           "input", "config", "pipeline", "model_load", "turn", "worker_start", "offline"}
_ERROR_TYPES = {"AudioInputError", "ValueError", "RuntimeError", "TypeError", "OSError", "TimeoutError",
                "WorkerExited", "LengthMismatch", "InvalidSegment", "TruncatedTranslation", "NetworkActivity",
                "ConcurrentTurn", "RetiredAfterTimeout", "ChangedAfterInitialization", "ChangedDuringTurn",
                "FingerprintChangedAtLoad", "StageError"}
_ACTIONS = {"SPEAK", "SPEAK_CUE", "REPEAT", "CONFIRM", "ABSTAIN"}


def metadata_only(row: dict) -> dict:
    """Project untrusted turn dictionaries onto a finite metadata schema.

    No identifiers, arbitrary reason strings, error messages, capability strings
    or network details cross this boundary. Numeric fields reject NaN/Inf/text.
    """
    def number(value):
        return value if type(value) in (int, float) and math.isfinite(value) else None

    out = {"logging_policy": "metadata-v1"}
    for key in ("audio_s", "endpoint_to_first_audio_ms", "endpoint_to_first_delivery_ms", "endpoint_to_text_ms", "total_ms", "rtf", "peak_rss_mb"):
        out[key] = number(row.get(key))
    out["peak_rss_mib"] = number(row.get("peak_rss_mib", row.get("peak_rss_mb")))
    out["rss_unit"] = "MiB"  # legacy *_mb values are explicitly documented aliases
    out["stage_ms"] = {k: number(v) for k, v in row.get("stage_ms", {}).items() if k in _STAGES}
    out["config_hash"] = row.get("config_hash") if re.fullmatch(r"[a-f0-9]{16,64}", str(row.get("config_hash", ""))) else None
    out["direction"] = row.get("direction") if row.get("direction") in ("vi-en", "vi-ko", "en-vi", "en-ko", "ko-en") else None
    out["mode"] = row.get("mode") if row.get("mode") in ("stub", "full", "branch-a-bench", "custom") else None
    out["status"] = row.get("status") if row.get("status") in ("ok", "no_speech", "invalid_audio", "error", "timeout") else "error"
    gate = row.get("gate", {})
    action = gate.get("action") if isinstance(gate, dict) else None
    out["gate"] = {"action": action if action in _ACTIONS else "ABSTAIN"}
    urgency = gate.get("urgency") if isinstance(gate, dict) else None
    urgency_status = urgency.get("classifier_status") if isinstance(urgency, dict) else row.get("urgency_status")
    out["urgency_status"] = urgency_status if urgency_status in ("READY", "UNCONFIGURED", "DISABLED", "ERROR", "LEGACY") else None
    cue_enabled = urgency.get("cue_enabled") if isinstance(urgency, dict) else row.get("cue_enabled")
    out["cue_enabled"] = cue_enabled is True and out["urgency_status"] in ("READY", "LEGACY")
    stage, error = row.get("error_stage"), row.get("error_type")
    out["error_stage"] = stage if stage in _STAGES else None
    out["error_type"] = error if error in _ERROR_TYPES else "StageError" if error is not None else None
    out["safety_status"] = row.get("safety_status") if row.get("safety_status") in ("PASS", "CONFIRM", "FAIL") else None
    if re.fullmatch(r"[a-f0-9]{64}", str(row.get("runtime_manifest_sha256", ""))):
        out["runtime_manifest_sha256"] = row["runtime_manifest_sha256"]
    if isinstance(row.get("nmt_evidence"), dict):
        evidence = row["nmt_evidence"]
        out["nmt_evidence"] = {k: number(evidence.get(k)) for k in ("source_tokens", "output_tokens",
            "source_unknown_tokens", "output_unknown_tokens", "constraints_requested", "constraints_satisfied")}
        for k in ("eos_reached", "truncated", "input_truncated"):
            out["nmt_evidence"][k] = evidence.get(k) if type(evidence.get(k)) is bool else None
    if type(row.get("output_written")) is bool:
        out["output_written"] = row["output_written"]
    if isinstance(row.get("offline_audit"), dict):
        out["offline_audit"] = {k: number(row["offline_audit"].get(k))
                                for k in ("net_attempts_python", "net_conns_os_seen")}
    return out


class PrivacyLogError(ValueError):
    """The selected existing log has incompatible or unverifiable privacy policy."""


class JsonlLogger:
    """Append-only writer with a default metadata allowlist and explicit eval opt-in.

    Refuse mixed-policy logs; never delete or silently reuse old text-bearing logs.
    The returned row is the exact projection suitable for stdout as well.
    """

    def __init__(self, path: Path, *, eval_mode: bool = False) -> None:
        self.path = Path(path)
        self.eval_mode = eval_mode
        self.policy = "evaluation-v1" if eval_mode else "metadata-v1"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open(encoding="utf8") as stream:
                for line in stream:
                    try:
                        old = json.loads(line)
                        projection = metadata_only(old) if not eval_mode else old
                        # Safe older metadata rows can omit newer optional
                        # measurements. Every existing field must still match
                        # the allowlist projection, including nested gate data.
                        unsafe = any(key not in projection or value != projection[key] for key, value in old.items())
                        if old.get("logging_policy") != self.policy or (not eval_mode and unsafe):
                            raise PrivacyLogError("incompatible log privacy policy")
                    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
                        raise PrivacyLogError("unverifiable log privacy policy") from exc

    def write(self, row: dict) -> dict:
        safe = dict(row) if self.eval_mode else metadata_only(row)
        safe["logging_policy"] = self.policy
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=False, allow_nan=False) + "\n")
        return safe
