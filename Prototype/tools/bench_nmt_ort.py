"""Lean NMT bench (ORT + sentencepiece + numpy only; no torch). Reports whole-sentence latency, encoder ms, per-decoder-step ms, peak RSS.
Composed estimate for phone: encoder_ms + n_tokens * decoder_step_ms (label "composed (est.)").
python tools/bench_nmt_ort.py [--threads 2] [--repeats 3]"""
import argparse, json, platform, statistics, sys, time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tonebridge.stages.nmt_ort import DEFAULT_DIR, OrtMarianNmt

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=2); ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--out", default="results/nmt_vi_en_ort_bench.json")
a = ap.parse_args()
n = OrtMarianNmt(threads=a.threads)
sents = [l.strip() for l in open(Path(__file__).resolve().parents[1] / "configs/sentences_vi.txt", encoding="utf8") if l.strip()]
for s in sents[:3]: n.greedy(n.encode_ids(s))  # warm-up
tot, enc, step, ntok = [], [], [], []
for _ in range(a.repeats):
    for s in sents:
        t = time.perf_counter(); ids = n.encode_ids(s); out, m = n.greedy(ids); n.decode_ids(out); tot.append((time.perf_counter() - t) * 1000)
        enc.append(m["encoder_ms"]); ntok.append(m["decoder_steps"])
        step.append((tot[-1] - m["encoder_ms"]) / m["decoder_steps"])
q = lambda x, p: float(np.percentile(x, p))
mi = psutil.Process().memory_info()
res = {"model": "opus-mt-vi-en INT8 arm64-config, greedy", "threads": a.threads, "n_runs": len(tot), "host": platform.processor(),
       "total_ms_p50": round(q(tot, 50), 1), "total_ms_p95": round(q(tot, 95), 1), "encoder_ms_p50": round(q(enc, 50), 1),
       "decoder_step_ms_p50": round(q(step, 50), 2), "decoder_steps_mean": round(statistics.mean(ntok), 1),
       "composed_est_ms_p50": round(q(enc, 50) + statistics.mean(ntok) * q(step, 50), 1),
       "peak_rss_mb": round(getattr(mi, "peak_wset", mi.rss) / 2**20), "rss_mb": round(mi.rss / 2**20),
       "int8_mb": round(sum(f.stat().st_size for f in DEFAULT_DIR.glob("*_quantized.onnx")) / 2**20, 1)}
print(json.dumps(res, indent=1)); Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf8")
