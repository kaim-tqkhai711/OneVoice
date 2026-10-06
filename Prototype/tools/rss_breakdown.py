"""Per-stage RSS growth: each stage alone in a fresh process, 20 real FLEURS dev utterances (x86 proxy, 2 threads). Shows where the resident set grows.
python tools/rss_breakdown.py -> results/rss_breakdown.json"""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = r'''
import sys, json, psutil, soundfile as sf
sys.path.insert(0, "src")
from tonebridge.evalkit import datasets
p = psutil.Process(); mb = lambda: round(p.memory_info().rss / 2**20, 1)
split = json.load(open("configs/splits/fleurs_vi_dev_test.json")); by = {u: pa for u, pa, _, _ in datasets.fleurs_vi()}
xs = [sf.read(by[u], dtype="float32")[0] for u in split["dev"][:20]]
stage = sys.argv[1]; r = {"stage": stage, "rss_baseline_mb": mb()}
if stage == "asr":
    from tonebridge.stages.asr_sherpa import SherpaZipformerVi
    a = SherpaZipformerVi(decoder="decoder-epoch-12-avg-8.int8.onnx", threads=2); r["after_load"] = mb()
    texts = [a.transcribe(x, "vi").text for x in xs]
elif stage == "nmt":
    from tonebridge.stages.nmt_ort import OrtMarianNmt
    n = OrtMarianNmt(threads=2); r["after_load"] = mb()
    for s in [l.strip() for l in open("configs/sentences_vi.txt", encoding="utf8") if l.strip()][:20]: n.translate(s, "vi", "en")
elif stage == "tts":
    from tonebridge.stages.tts_piper import PiperEn
    t = PiperEn(threads=2); r["after_load"] = mb()
    for s in ["Take two tablets of paracetamol every six hours after the meal, and call the doctor if the pain gets worse."] * 20: t.synth(s)
elif stage == "vad":
    from tonebridge.stages.vad_silero import SileroVad
    v = SileroVad(); r["after_load"] = mb()
    for x in xs: v.segment(x)
r["after_20_utts"] = mb(); r["peak_wset_mb"] = round(getattr(p.memory_info(), "peak_wset", 0) / 2**20, 1)
print(json.dumps(r))
'''
out = []
for st in ("asr", "nmt", "tts", "vad"):
    res = subprocess.run([sys.executable, "-I", "-c", CODE, st], cwd=ROOT, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONUTF8": "1"})
    out.append(json.loads(res.stdout.strip().splitlines()[-1]) if res.returncode == 0 else {"stage": st, "error": res.stderr[-300:]})
    print(out[-1], flush=True)
(ROOT / "results/rss_breakdown.json").write_text(json.dumps(out, indent=1))
