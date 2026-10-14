"""Inspect dependencies and pinned assets; exit 0 only when selected mode is ready.

Example: python tools/check_runtime.py --mode full --profile runtime
No models are downloaded, created or loaded by this command.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tonebridge.preflight import check_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("stub", "branch-a-bench", "full"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.json")
    parser.add_argument("--profile", choices=("runtime", "dev", "bench", "playback"), default="runtime")
    parser.add_argument("--allow-version-drift", action="store_true",
                        help="diagnostic only: still check imports and asset hashes")
    args = parser.parse_args()
    result = check_runtime(args.config, args.mode, args.profile, strict_versions=not args.allow_version_drift)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
