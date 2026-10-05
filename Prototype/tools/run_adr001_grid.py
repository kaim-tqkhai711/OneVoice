"""ADR-001 grid: WER/CER x SNR (clean/10/5/0) x arm (off / on=GTCRN / oa=beta mix), Branch A front-end only.
Noise: DEMAND environments (CC-BY-4.0), one per utterance round-robin with a fixed seed. Per-utterance error counts are saved so
paired bootstrap CIs on the arm differences can be computed. Rule (ADR-001): an arm wins only by >= 2 points absolute WER.
python tools/run_adr001_grid.py --n 200 --betas 0.5 --out results/adr001_dev.json"""
import argparse, json, sys, time, zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonebridge.evalkit import datasets
from tonebridge.evalkit.metrics import ErrorCounts
from tonebridge.evalkit.noise import measured_snr_db, mix_at_snr
from tonebridge.stages.asr_sherpa import SherpaZipformerVi
from tonebridge.stages.denoise import GtcrnDenoiser, oa_mix

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--betas", type=float, nargs="+", default=[0.5]); ap.add_argument("--snrs", nargs="+", default=["clean", "10", "5", "0"])
ap.add_argument("--noise-dir", type=Path, default=ROOT / "data/noise/demand"); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

noises = {}
for z in sorted(a.noise_dir.glob("*_16k.zip")):
    with zipfile.ZipFile(z) as zf:
        name = [n for n in zf.namelist() if n.endswith("ch01.wav")][0]
        with zf.open(name) as f: x, sr = sf.read(f, dtype="float32")
    assert sr == 16000; noises[z.name.split("_")[0]] = x
print("noise envs:", {k: round(len(v) / 16000) for k, v in noises.items()}, flush=True)
env_names = sorted(noises)

items = list(datasets.fleurs_vi())
rng = np.random.default_rng(a.seed)
sel = sorted(rng.choice(len(items), size=min(a.n, len(items)), replace=False).tolist())
asr = SherpaZipformerVi(decoder="decoder-epoch-12-avg-8.int8.onnx", threads=2)
gt = GtcrnDenoiser(threads=2)
arms = ["off", "on"] + [f"oa{b}" for b in a.betas]
res = {"n_utts": len(sel), "seed": a.seed, "noise_envs": env_names, "betas": a.betas, "cells": {}}
clips = []
for i in sel:
    uid, p, ref, _ = items[i]
    x, sr = sf.read(p, dtype="float32"); assert sr == 16000
    clips.append((uid, x, ref))
t0 = time.time()
for snr in a.snrs:
    mrng = np.random.default_rng(a.seed + (0 if snr == "clean" else int(snr) + 100))
    noisy_set = []
    for k, (uid, x, ref) in enumerate(clips):
        y = x if snr == "clean" else mix_at_snr(x, noises[env_names[k % len(env_names)]], float(snr), mrng)
        noisy_set.append(y)
    enh = [gt.process(y) for y in noisy_set]
    for arm in arms:
        c = ErrorCounts(); per = []
        for k, (uid, x, ref) in enumerate(clips):
            y = noisy_set[k] if arm == "off" else enh[k] if arm == "on" else oa_mix(noisy_set[k], enh[k], float(arm[2:]))
            before = (c.words, c.word_errors, c.chars, c.char_errors)
            c.add(ref, asr.transcribe(y, "vi").text)
            per.append([c.words - before[0], c.word_errors - before[1], c.chars - before[2], c.char_errors - before[3]])
        res["cells"][f"{snr}|{arm}"] = {"WER": round(c.wer, 4), "CER": round(c.cer, 4), "words": c.words, "per_utt": per}
        print(f"snr={snr:>5} arm={arm:<6} WER={c.wer:.4f} CER={c.cer:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if snr != "clean":
        res.setdefault("measured_snr_db_mean", {})[snr] = round(float(np.mean([measured_snr_db(clips[k][1], noisy_set[k]) for k in range(len(clips))])), 2)
    a.out.write_text(json.dumps(res), encoding="utf8")
