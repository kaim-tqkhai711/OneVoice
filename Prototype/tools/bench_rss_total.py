"""Total process RSS with ASR + NMT + TTS (+ Silero VAD, optional GTCRN) all resident, after real utterances. x86 laptop proxy, 2 threads.
python tools/bench_rss_total.py [--gtcrn] [--out results/rss_total.json]"""
import argparse, json, sys, time
from pathlib import Path

import psutil, soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser(); ap.add_argument("--gtcrn", action="store_true"); ap.add_argument("--out", default="results/rss_total.json"); ap.add_argument("--max-s", type=float, default=None, help="only utterances up to this many seconds (PTT-like)")
a = ap.parse_args()
p = psutil.Process()
mb = lambda: round(p.memory_info().rss / 2**20, 1)
peak = lambda: round(getattr(p.memory_info(), "peak_wset", p.memory_info().rss) / 2**20, 1)
res = {"threads": 2, "rss_baseline_mb": mb()}
from tonebridge.stages.asr_sherpa import SherpaZipformerVi
from tonebridge.stages.nmt_ort import OrtMarianNmt
from tonebridge.stages.tts_piper import PiperEn
from tonebridge.stages.vad_silero import SileroVad
res["rss_after_imports_mb"] = mb()
asr = SherpaZipformerVi(decoder="decoder-epoch-12-avg-8.int8.onnx", threads=2); res["rss_after_asr_load_mb"] = mb()
nmt = OrtMarianNmt(threads=2); res["rss_after_nmt_load_mb"] = mb()
tts = PiperEn(threads=2); res["rss_after_tts_load_mb"] = mb()
vad = SileroVad(); res["rss_after_vad_load_mb"] = mb()
if a.gtcrn:
    from tonebridge.stages.denoise import GtcrnDenoiser
    g = GtcrnDenoiser(threads=2); res["rss_after_gtcrn_load_mb"] = mb()
from tonebridge.evalkit import datasets
items = [it for it in datasets.fleurs_vi() if a.max_s is None or sf.info(it[1]).duration <= a.max_s][:20]
for uid, path, ref, _ in items:
    x, _ = sf.read(path, dtype="float32")
    if a.gtcrn: x = g.process(x)
    vad.segment(x); r = asr.transcribe(x, "vi"); m = nmt.translate(r.text, "vi", "en"); list(tts.stream(m.tgt_text or "ok", "en"))
res["rss_after_20_utts_mb"] = mb(); res["peak_rss_mb"] = peak(); res["n_utts"] = len(items); res["max_s"] = a.max_s; res["gtcrn_loaded"] = a.gtcrn
res["note"] = "x86 Windows proxy; phone RSS differs (different allocator/ORT build). Phone threshold for total pipeline: 1.0 GB (owner, 2026-10-07)."
print(json.dumps(res, indent=1)); Path(a.out).write_text(json.dumps(res, indent=1))
