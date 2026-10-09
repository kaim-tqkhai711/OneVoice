"""Local pretrained-model benchmark, one direction per process for meaningful peak RSS."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tonebridge.nmt_evidence import DIRECTIONS
from tonebridge.clinical_safety import ClinicalSafetyChecker
from tonebridge.stages.nmt_marian import MarianTextAdapter


def main():
    import numpy as np
    import psutil
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=DIRECTIONS, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("configs/nmt/text_references_v1.jsonl"))
    ap.add_argument("--split", choices=("dev", "holdout"), default="dev")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hyp-out", type=Path, help="One output per ID from repeat 0, for independent evaluation")
    args = ap.parse_args()
    if args.repeats < 1:
        raise ValueError("positive repeat count required")
    items = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [r for r in items if r["direction"] == args.direction and r["split"] == args.split]
    if not items:
        raise ValueError("empty_benchmark")
    src, tgt = args.direction.split("-")
    proc = psutil.Process()
    before = proc.memory_info().rss
    t = time.perf_counter()
    nmt = MarianTextAdapter(args.model_dir, src, tgt, threads=args.threads)
    load_ms = (time.perf_counter()-t)*1000
    after = proc.memory_info().rss
    checker = ClinicalSafetyChecker(src, tgt)
    nmt.translate(items[0]["source"], src, tgt)
    rows, counts = [], Counter()
    for repeat in range(args.repeats):
        for item in items:
            start = time.perf_counter()
            result = nmt.translate(item["source"], src, tgt)
            nmt_ms = (time.perf_counter()-start)*1000
            start = time.perf_counter()
            safety = checker.check(result)
            safety_ms = (time.perf_counter()-start)*1000
            counts[safety.status] += 1
            rows.append({"id": item["id"], "repeat": repeat, "nmt_ms": nmt_ms,
                         "safety_ms": safety_ms, "translation": result.model_dump(), "safety": safety.model_dump()})
    memory = proc.memory_info()
    ms = [r["nmt_ms"] for r in rows]
    report = {"direction": args.direction, "n_utterances": len(items), "n_runs": len(rows),
              "split": args.split, "load_ms": load_ms, "rss_before_bytes": before, "rss_loaded_bytes": after,
              "process_peak_rss_bytes": getattr(memory, "peak_wset", None),
              "process_current_rss_bytes": memory.rss, "nmt_p50_ms": float(np.percentile(ms, 50)),
              "nmt_p95_ms": float(np.percentile(ms, 95)), "gate_counts": dict(counts),
              "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              "env": {"platform": platform.platform(), "python": sys.version, "threads": args.threads},
              "model_id": nmt.model_id, "revision": nmt.revision,
              "measurement": "text-only; excludes ASR/TTS/playback; no clinical quality claim",
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.hyp_out:
        args.hyp_out.parent.mkdir(parents=True, exist_ok=True)
        args.hyp_out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows if r["repeat"] == 0) + "\n",
                                encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
