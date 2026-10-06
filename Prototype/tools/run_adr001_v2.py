"""ADR-001 grid v2 (2026-10-07). SNR is over the VAD speech mask. noise types: demand (4 envs) | babble | alarm; arms off/on/oa<b>/cns.
One process per noise type (clean is run by the process that gets --with-clean). Resumable: finished cells are skipped.
python tools/run_adr001_v2.py --split dev --noise demand --with-clean --out results/adr001_v2_dev_demand.json"""
import argparse, json, sys, time, zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonebridge.evalkit import datasets
from tonebridge.evalkit.metrics import ErrorCounts
from tonebridge.evalkit.noise import measured_snr_masked_db, mix_at_snr_masked, speech_mask
from tonebridge.evalkit.noisegen import make_alarm, make_babble
from tonebridge.stages.asr_sherpa import SherpaZipformerVi
from tonebridge.stages.denoise import GtcrnDenoiser, oa_mix
from tonebridge.stages.denoise_classic import ClassicalNs
from tonebridge.stages.vad_silero import SileroVad

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["dev", "test"], required=True); ap.add_argument("--noise", choices=["demand", "babble", "alarm"], required=True)
ap.add_argument("--snrs", nargs="+", default=["10", "5", "0", "-5"]); ap.add_argument("--with-clean", action="store_true")
ap.add_argument("--arms", nargs="+", default=["off", "on", "oa0.25", "oa0.5", "oa0.75", "cns"]); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--limit", type=int, default=None); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--asr-method", default="greedy_search"); ap.add_argument("--chunked", action="store_true", help="decode per Silero VAD segment and join")
a = ap.parse_args()

split = json.loads((ROOT / "configs/splits/fleurs_vi_dev_test.json").read_text())
byid = {u: (p, r) for u, p, r, _ in datasets.fleurs_vi()}
ids = split[a.split][: a.limit]
SEED = split["seed"]
vad = SileroVad()

def load(uid):
    x, sr = sf.read(byid[uid][0], dtype="float32"); assert sr == 16000; return x

noise_tracks: list[tuple[str, np.ndarray]] = []
if a.noise == "demand":
    for z in sorted((ROOT / "data/noise/demand").glob("*_16k.zip")):
        with zipfile.ZipFile(z) as zf:
            with zf.open([n for n in zf.namelist() if n.endswith("ch01.wav")][0]) as f: x, sr = sf.read(f, dtype="float32")
        noise_tracks.append((z.name.split("_")[0], x))
elif a.noise == "babble":
    pool = [load(u) for u in split["babble_pool"]]
    noise_tracks.append(("babble6", make_babble(pool, 90.0, 6, SEED + 1)))
else:
    noise_tracks.append(("alarm", make_alarm(90.0, SEED + 2)))

asr = SherpaZipformerVi(decoder="decoder-epoch-12-avg-8.int8.onnx", threads=a.threads, decoding_method=a.asr_method)

def transcribe(y):
    if not a.chunked:
        return asr.transcribe(y, "vi").text
    sp = vad.segments(y)
    return " ".join(asr.transcribe(y[int(max(0, s0 - 0.1) * 16000): int((e0 + 0.1) * 16000)], "vi").text for s0, e0 in sp) if sp else asr.transcribe(y, "vi").text
gt = GtcrnDenoiser(threads=a.threads); cns = ClassicalNs()
res = json.loads(a.out.read_text()) if a.out.exists() else {"split": a.split, "ids": ids, "noise": a.noise, "tracks": [n for n, _ in noise_tracks], "cells": {}, "snr_check": {}}
clips = []
for u in ids:
    x = load(u); t0, t1 = vad.segment(x)  # VAD span on the CLEAN clip -> speech mask
    clips.append((u, x, byid[u][1], speech_mask(len(x), [(t0, t1)] if t1 > t0 else [])))
t_start = time.time()
conds = (["clean"] if a.with_clean else []) + a.snrs
for snr in conds:
    if all(f"{a.noise}|{snr}|{arm}" in res["cells"] for arm in a.arms):
        continue
    mrng = np.random.default_rng(SEED + (0 if snr == "clean" else int(snr) + 100) + {"demand": 0, "babble": 1000, "alarm": 2000}[a.noise])
    noisy, chk = [], []
    for k, (u, x, ref, m) in enumerate(clips):
        if snr == "clean":
            y = x
        else:
            y = mix_at_snr_masked(x, noise_tracks[k % len(noise_tracks)][1], float(snr), m, mrng)
            chk.append(measured_snr_masked_db(x, y, m))
        noisy.append(y)
    if chk:
        res["snr_check"][f"{a.noise}|{snr}"] = {"target": float(snr), "mean": round(float(np.mean(chk)), 3), "max_abs_dev": round(float(np.max(np.abs(np.array(chk) - float(snr)))), 3)}
    enh = [gt.process(y) for y in noisy] if any(r in ("on",) or r.startswith("oa") for r in a.arms) else None
    cn = [cns.process(y) for y in noisy] if "cns" in a.arms else None
    for arm in a.arms:
        key = f"{a.noise}|{snr}|{arm}"
        if key in res["cells"]: continue
        c = ErrorCounts(); per = []; hyps = []
        for k, (u, x, ref, m) in enumerate(clips):
            y = noisy[k] if arm == "off" else enh[k] if arm == "on" else cn[k] if arm == "cns" else oa_mix(noisy[k], enh[k], float(arm[2:]))
            b = (c.words, c.word_errors, c.chars, c.char_errors)
            h = transcribe(y); hyps.append(h)
            c.add(ref, h)
            per.append([c.words - b[0], c.word_errors - b[1], c.chars - b[2], c.char_errors - b[3]])
        res["cells"][key] = {"WER": round(c.wer, 4), "CER": round(c.cer, 4), "words": c.words, "per_utt": per, "hyp": hyps}
        print(f"{key:<24} WER={c.wer:.4f} CER={c.cer:.4f} ({time.time()-t_start:.0f}s)", flush=True)
        a.out.write_text(json.dumps(res), encoding="utf8")
print("DONE", time.time() - t_start, flush=True)
