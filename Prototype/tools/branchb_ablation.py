"""Branch B ablation (E): same utterance, input = raw noisy audio vs GTCRN(noisy). SwiftF0 on both (and on the clean clip as pitch reference).
Per SNR x noise type: median / mean |F0 deviation| in cents between raw and denoised on frames voiced in BOTH, deviation of each from the clean
reference, change in voiced_frac (denoised - raw). Dev split only. python tools/branchb_ablation.py --n 100"""
import argparse, json, sys, zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swift_f0 import SwiftF0
from tonebridge.evalkit import datasets
from tonebridge.evalkit.noise import mix_at_snr_masked, speech_mask
from tonebridge.evalkit.noisegen import make_alarm, make_babble
from tonebridge.stages.denoise import GtcrnDenoiser
from tonebridge.stages.vad_silero import SileroVad

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--out", default="results/branchb_ablation.json")
a = ap.parse_args()
split = json.loads((ROOT / "configs/splits/fleurs_vi_dev_test.json").read_text()); byid = {u: p for u, p, _, _ in datasets.fleurs_vi()}
ids = split["dev"][: a.n]; SEED = split["seed"]
vad, gt, f0 = SileroVad(), GtcrnDenoiser(threads=1), SwiftF0(threads=1, spin=False)
envs = []
for z in sorted((ROOT / "data/noise/demand").glob("*_16k.zip")):
    with zipfile.ZipFile(z) as zf, zf.open([n for n in zf.namelist() if n.endswith("ch01.wav")][0]) as f:
        envs.append(sf.read(f, dtype="float32")[0])
pool = [sf.read(byid[u], dtype="float32")[0] for u in split["babble_pool"]]
tracks = {"demand": envs, "babble": [make_babble(pool, 90.0, 6, SEED + 1)], "alarm": [make_alarm(90.0, SEED + 2)]}
clips = []
for u in ids:
    x = sf.read(byid[u], dtype="float32")[0]; t0, t1 = vad.segment(x)
    clips.append((x, speech_mask(len(x), [(t0, t1)] if t1 > t0 else []), f0.detect(x, 16000)))
res = {"n_utts": len(ids), "rows": {}}


def cents(a_, b_):
    return np.abs(1200 * np.log2(a_ / b_))


for nt, trs in tracks.items():
    for snr in (10, 5, 0, -5):
        rng = np.random.default_rng(SEED + snr + 7)
        dev_rr, dev_rc, dev_ec, dv, vf_raw, vf_enh = [], [], [], [], [], []
        for k, (x, m, rc) in enumerate(clips):
            y = mix_at_snr_masked(x, trs[k % len(trs)], float(snr), m, rng)
            e = gt.process(y)
            rr, re = f0.detect(y, 16000), f0.detect(e, 16000)
            n = min(len(rr.confidence), len(re.confidence), len(rc.confidence))
            vr, ve, vc = rr.confidence[:n] >= 0.5, re.confidence[:n] >= 0.5, rc.confidence[:n] >= 0.5
            both = vr & ve
            dev_rr.extend(cents(re.pitch_hz[:n][both], rr.pitch_hz[:n][both]))
            b1 = vr & vc; dev_rc.extend(cents(rr.pitch_hz[:n][b1], rc.pitch_hz[:n][b1]))
            b2 = ve & vc; dev_ec.extend(cents(re.pitch_hz[:n][b2], rc.pitch_hz[:n][b2]))
            vf_raw.append(vr.mean()); vf_enh.append(ve.mean()); dv.append(ve.mean() - vr.mean())
        q = lambda v, p: round(float(np.percentile(v, p)), 1) if len(v) else None
        res["rows"][f"{nt}|{snr}"] = {"raw_vs_denoised_cents_median": q(dev_rr, 50), "raw_vs_denoised_cents_p90": q(dev_rr, 90),
                                       "raw_vs_clean_cents_median": q(dev_rc, 50), "denoised_vs_clean_cents_median": q(dev_ec, 50),
                                       "voiced_frac_raw": round(float(np.mean(vf_raw)), 3), "voiced_frac_denoised": round(float(np.mean(vf_enh)), 3),
                                       "voiced_frac_change_mean": round(float(np.mean(dv)), 3), "n_frames_both_voiced": len(dev_rr)}
        print(nt, snr, res["rows"][f"{nt}|{snr}"], flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))
