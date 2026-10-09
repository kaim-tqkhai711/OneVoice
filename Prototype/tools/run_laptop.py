"""Process watchdog. Native inference hangs/crashes cannot be cancelled safely in threads."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def worker_error(arguments, stage, error_type, **extra):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log", type=Path, default=ROOT / "results/laptop_current/turns.jsonl")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--direction")
    args, _ = parser.parse_known_args(arguments)
    row = {"status": "error", "error_stage": stage, "error_type": error_type,
           "direction": args.direction, "gate": {"action": "ABSTAIN"},
           "output_written": bool(args.out and args.out.exists()), **extra}
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row))


def supervise(arguments, timeout):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        result = subprocess.run([sys.executable, "-X", "utf8", "-m", "tonebridge.cli", *arguments], env=env, timeout=timeout)
        if result.returncode < 0 or result.returncode > 2:
            worker_error(arguments, "worker", "WorkerExit", worker_exit_code=result.returncode)
        return result.returncode
    except subprocess.TimeoutExpired:
        # subprocess.run has killed and reaped the entire inference process, including its threads.
        worker_error(arguments, "watchdog", "Timeout")
        return 124


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("arguments", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.timeout <= 0:
        ap.error("timeout must be positive")
    raise SystemExit(supervise(a.arguments[1:] if a.arguments[:1] == ["--"] else a.arguments, a.timeout))
