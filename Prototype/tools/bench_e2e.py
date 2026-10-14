"""Measured 100-turn laptop benchmark, persistent stages, cold/warm and MiB/GiB.

python tools/bench_e2e.py --mode stub --turns 100 --out results/bench_e2e_stub.json
python tools/bench_e2e.py --mode branch-a-bench --wav input.wav --turns 100 --out results/bench_e2e_real.json
Real modes require pinned packs AND genuine mono 16k WAV inputs. Blocked runs
write metadata-only diagnostics with zero measured samples, never stub fallback.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stub", "branch-a-bench", "full"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.json")
    parser.add_argument("--wav", type=Path, nargs="+")
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.turns < 50:
        parser.error("at least 50 turns are required")
    if args.out.exists():
        parser.error("choose a fresh result path")
    from tonebridge.preflight import check_runtime
    preflight = check_runtime(args.config, args.mode)
    if not preflight["ok"]:
        result = {"schema_version": "tonebridge-e2e-v1", "status": "BLOCKED_PREFLIGHT", "mode": args.mode,
            "requested_turns": args.turns, "processed_turns": 0, "audio_turns": 0,
            "config_hash": preflight.get("config_hash"), "assets_checked": preflight["assets_checked"],
            "missing_assets": sum(e.startswith("missing asset:") for e in preflight["errors"]),
            "preflight_error_count": len(preflight["errors"]), "latency": None, "rss": None,
            "note": "No model inference executed; no measured latency/RSS or automatic fallback."}
    else:
        import numpy as np
        from tonebridge.audio import load_wav
        from tonebridge.config import PipelineConfig
        from tonebridge.benchmark import run_benchmark
        cfg = PipelineConfig.load(args.config).prepare(args.mode)
        if not args.wav and args.mode != "stub":
            parser.error("real benchmark requires --wav with genuine speech inputs")
        try:
            inputs = [load_wav(path, cfg.sample_rate, cfg.max_utterance_s) for path in args.wav] if args.wav else [
                (.1 * np.sin(2 * np.pi * 220 * np.arange(int(duration * 16000)) / 16000)).astype(np.float32)
                for duration in (.5, 1., 2., 3.) if duration <= cfg.max_utterance_s]
            result = run_benchmark(cfg, args.mode, inputs, args.turns)
        except Exception:
            result = {"schema_version": "tonebridge-e2e-v1", "status": "FAILED", "mode": args.mode,
                "requested_turns": args.turns, "error": "BenchmarkInputOrRuntimeError"}
        result["input_kind"] = "provided_wav" if args.wav else "synthetic_tones_stub_control_flow_only"
        result["input_sha256"] = [hashlib.sha256(x.tobytes()).hexdigest() for x in inputs] if "inputs" in locals() else []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(args.out)
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "cold_turn", "input_sha256")}, indent=2))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
