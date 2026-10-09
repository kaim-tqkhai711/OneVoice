"""Independent local text -> NMT -> safety command. Never downloads assets."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tonebridge.clinical_safety import ClinicalSafetyChecker
from tonebridge.nmt_evidence import DIRECTIONS
from tonebridge.stages.nmt_marian import MarianTextAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=DIRECTIONS, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()
    src, tgt = args.direction.split("-")
    try:
        nmt = MarianTextAdapter(args.model_dir, src, tgt, max_new_tokens=args.max_new_tokens)
        result = nmt.translate(args.text, src, tgt)
        safety = ClinicalSafetyChecker(src, tgt).check(result)
        print(json.dumps({"translation": result.model_dump(), "safety": safety.model_dump()}, ensure_ascii=False))
        return 0 if safety.status == "PASS" else 2
    except Exception as exc:
        print(json.dumps({"direction": args.direction, "status": "ERROR", "error_type": type(exc).__name__,
                          "reason": str(exc), "automatic_speech_allowed": False}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
