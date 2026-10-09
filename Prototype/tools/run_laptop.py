"""Process watchdog. Native inference hangs/crashes cannot be cancelled safely in threads."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def supervise(arguments, timeout):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), "PYTHONIOENCODING": "utf-8"}
    try:
        result = subprocess.run([sys.executable, "-m", "tonebridge.cli", *arguments], env=env, timeout=timeout)
        if result.returncode < 0 or result.returncode > 2:
            print(json.dumps({"status": "error", "error_stage": "worker", "error_type": "WorkerExit",
                              "worker_exit_code": result.returncode, "gate": {"action": "ABSTAIN"}, "output_written": False}))
        return result.returncode
    except subprocess.TimeoutExpired:
        # subprocess.run has killed and reaped the entire inference process, including its threads.
        print(json.dumps({"status": "error", "error_stage": "watchdog", "error_type": "Timeout",
                          "gate": {"action": "ABSTAIN"}, "output_written": False}))
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
