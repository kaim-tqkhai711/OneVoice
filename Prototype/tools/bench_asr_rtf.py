"""RTF + size of the exact ASR files that ship. Usage: python tools/bench_asr_rtf.py [--threads 2] [--repeats 10]

RTF = decode wall time / audio duration, offline transducer, full-utterance decode (Proposal deviation #2).
The same script and the same files are run on the phone for k calibration (RTF_phone / RTF_laptop).
"""
import argparse, json, os, platform, statistics, time
from pathlib import Path

import numpy as np
import psutil
import sherpa_onnx
import soundfile as sf

D = Path(__file__).resolve().parents[1] / "models/asr/zipformer-vi-int8"
FILES = ["encoder-epoch-12-avg-8.int8.onnx", "decoder-epoch-12-avg-8.onnx", "joiner-epoch-12-avg-8.int8.onnx", "tokens.txt", "bpe.model"]

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--repeats", type=int, default=10)
ap.add_argument("--out", default="results/asr_vi_rtf.json")
a = ap.parse_args()

rss0 = psutil.Process().memory_info().rss / 2**20
t0 = time.perf_counter()
rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=str(D / FILES[0]), decoder=str(D / FILES[1]), joiner=str(D / FILES[2]), tokens=str(D / "tokens.txt"),
    num_threads=a.threads, sample_rate=16000, feature_dim=80, decoding_method="greedy_search", provider="cpu")
load_s = time.perf_counter() - t0

rows = []
for w in sorted((D / "test_wavs").glob("*.wav")):
    x, sr = sf.read(w, dtype="float32")
    assert sr == 16000, sr
    dur = len(x) / sr
    ts = []
    for i in range(a.repeats + 2):  # 2 warm-up
        t = time.perf_counter()
        s = rec.create_stream(); s.accept_waveform(sr, x); rec.decode_stream(s)
        ts.append(time.perf_counter() - t)
    ts = ts[2:]
    rows.append({"wav": w.name, "audio_s": round(dur, 3), "text": s.result.text, "decode_ms_p50": round(1000 * statistics.median(ts), 1),
                 "decode_ms_max": round(1000 * max(ts), 1), "rtf_p50": round(statistics.median(ts) / dur, 4)})
rss1 = psutil.Process().memory_info().rss / 2**20
tot_a = sum(r["audio_s"] for r in rows)
tot_d = sum(r["decode_ms_p50"] for r in rows) / 1000
res = {"model": "sherpa-onnx-zipformer-vi-int8-2025-04-20", "sherpa_onnx": sherpa_onnx.__version__, "threads": a.threads, "repeats": a.repeats,
       "host": platform.processor(), "files_mb": {f: round(os.path.getsize(D / f) / 2**20, 2) for f in FILES},
       "model_mb_total": round(sum(os.path.getsize(D / f) for f in FILES[:3]) / 2**20, 2), "load_s": round(load_s, 2),
       "rss_mb_before": round(rss0), "rss_mb_after": round(rss1), "rtf_pooled": round(tot_d / tot_a, 4), "rows": rows}
Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf8")
print(json.dumps(res, indent=1, ensure_ascii=False))
