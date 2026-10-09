"""One-direction laptop E2E benchmark; use a separate process for each active language pack."""
import argparse
from collections import Counter
import json
import platform
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonebridge.audio import load_wav
from tonebridge.cli import public_row
from tonebridge.config import PipelineConfig
from tonebridge.offline_guard import OfflineGuard
from tonebridge.pipeline import Pipeline
from tonebridge.stages.factory import real_full
from tonebridge.telemetry import summarize_latency


def benchmark(config, manifest, destination, warmup=3, repeats=1):
    cfg = PipelineConfig.load(config)
    samples = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not samples or repeats < 1 or warmup < 0:
        raise ValueError("nonempty manifest, repeats>=1 and warmup>=0 required")
    if any(s.get("direction", cfg.direction) != cfg.direction for s in samples):
        raise ValueError("one direction per benchmark process")
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    with OfflineGuard() as guard:
        start = time.perf_counter()
        pipe = Pipeline(cfg, real_full(cfg))
        load_ms = (time.perf_counter() - start) * 1000
        for i in range(warmup):
            sample = samples[i % len(samples)]
            path = manifest.parent / sample["wav"]
            pipe.run(load_wav(path), "warmup")
        with (destination / "turns.jsonl").open("w", encoding="utf-8") as log:
            for repeat in range(repeats):
                for i, sample in enumerate(samples):
                    path = manifest.parent / sample["wav"]
                    result = pipe.run(load_wav(path), str(sample.get("id", i)), session_id=f"repeat-{repeat}")
                    row = public_row(result.record, cfg.log_content)
                    row["rss_mb"] = psutil.Process().memory_info().rss / 2**20
                    rows.append(row)
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                print("completed", len(rows), "turns", flush=True)
        pipe.close()
    def stats(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return summarize_latency(values) if values else None
    report = {"direction": cfg.direction, "platform": platform.platform(), "threads_per_model": 2,
              "config_hash": cfg.config_hash(), "model_load_ms": load_ms, "warmup_turns": warmup,
              "runtime_manifest_sha256": pipe.manifest_hash,
              "n_turns": len(rows), "n_manifest_entries": len(samples),
              "n_distinct_wavs": len({str((manifest.parent / s['wav']).resolve()) for s in samples}), "repeats": repeats,
              "label": "Repeated-sample runtime stability smoke" if repeats > 1 else "Local WAV runtime benchmark; quality not assessed",
              "gate_actions": dict(Counter(r["gate"]["action"] for r in rows)), "statuses": dict(Counter(r["status"] for r in rows)),
              "endpoint_to_first_sample_ms": stats("endpoint_to_first_audio_ms"), "endpoint_to_text_ms": stats("endpoint_to_text_ms"),
              "first_audible_ms": None, "peak_rss_mb": max(r["peak_rss_mb"] for r in rows),
              "rss_first_mb": rows[0]["rss_mb"], "rss_last_mb": rows[-1]["rss_mb"], "offline_audit": guard.report()}
    (destination / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if report["statuses"].get("error", 0) else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True, help="JSONL: id, wav (relative to manifest), optional direction")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    raise SystemExit(benchmark(a.config, a.manifest, a.out_dir, a.warmup, a.repeats))
