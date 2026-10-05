"""WER/CER of the shipped ASR on an eval set, split by tag (NEU/URG for recordings). Writes JSON.
python tools/run_asr_eval.py --set fleurs [--limit N] [--decoder decoder-epoch-12-avg-8.onnx]
python tools/run_asr_eval.py --set kit --kit-dir <dir with spkXX_NEU_s01.wav ...>
"""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tonebridge.evalkit import datasets
from tonebridge.evalkit.metrics import ErrorCounts
from tonebridge.stages.asr_sherpa import SherpaZipformerVi

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["fleurs", "kit"], required=True)
ap.add_argument("--kit-dir")
ap.add_argument("--limit", type=int)
ap.add_argument("--decoder", default="decoder-epoch-12-avg-8.onnx")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--out")
a = ap.parse_args()

items = datasets.fleurs_vi(limit=a.limit) if a.set == "fleurs" else datasets.recording_kit(Path(a.kit_dir))
asr = SherpaZipformerVi(decoder=a.decoder, threads=a.threads)
by = defaultdict(ErrorCounts); allc = ErrorCounts(); dec_s = aud_s = 0.0; confs = []
for uid, path, ref, tag in items:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    if sr != 16000:
        import soxr; x = soxr.resample(x, sr, 16000)
    t = time.perf_counter(); r = asr.transcribe(x, "vi"); dec_s += time.perf_counter() - t; aud_s += len(x) / 16000
    by[tag].add(ref, r.text); allc.add(ref, r.text); confs.append(r.confidence)
def row(c): return {"utts": c.n_utts, "words": c.words, "WER": round(c.wer, 4), "CER": round(c.cer, 4)}
res = {"set": a.set, "decoder": a.decoder, "threads": a.threads, "all": row(allc), "by_tag": {k: row(v) for k, v in by.items()},
       "rtf": round(dec_s / aud_s, 4), "audio_s": round(aud_s, 1), "mean_conf": round(float(np.mean(confs)), 3)}
print(json.dumps(res, ensure_ascii=False, indent=1))
if a.out: Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf8")
