"""Writes clean/10/5/0/-5 dB samples (SNR over VAD mask) for each noise type from one dev utterance -> results/listen/. python tools/export_listen.py"""
import json, sys, zipfile
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from tonebridge.evalkit import datasets
from tonebridge.evalkit.noise import measured_snr_masked_db, mix_at_snr_masked, speech_mask
from tonebridge.evalkit.noisegen import make_alarm, make_babble
from tonebridge.stages.vad_silero import SileroVad
split = json.loads((ROOT / "configs/splits/fleurs_vi_dev_test.json").read_text()); byid = {u: p for u, p, _, _ in datasets.fleurs_vi()}
out = ROOT / "results/listen"; out.mkdir(parents=True, exist_ok=True); vad = SileroVad()
pool = [sf.read(byid[u], dtype="float32")[0] for u in split["babble_pool"][:60]]
with zipfile.ZipFile(ROOT / "data/noise/demand/PCAFETER_16k.zip") as zf, zf.open([n for n in zf.namelist() if n.endswith("ch01.wav")][0]) as f: dem, _ = sf.read(f, dtype="float32")
noises = {"demand_cafeteria": dem, "babble6": make_babble(pool, 60.0, 6, split["seed"] + 1), "alarm": make_alarm(60.0, split["seed"] + 2)}
uid = split["dev"][0]; x, _ = sf.read(byid[uid], dtype="float32"); t0, t1 = vad.segment(x); m = speech_mask(len(x), [(t0, t1)])
sf.write(out / f"{uid}_clean.wav", x, 16000); rows = [f"utt={uid} vad_span=({t0:.2f},{t1:.2f}) s"]
for nn, nz in noises.items():
    for snr in (10, 5, 0, -5):
        y = mix_at_snr_masked(x, nz, snr, m, np.random.default_rng(snr + 1000)); sf.write(out / f"{uid}_{nn}_{snr:+d}dB.wav", y, 16000)
        rows.append(f"{nn} target {snr:+d} dB measured(masked) {measured_snr_masked_db(x, y, m):+.2f} dB")
(out / "README.txt").write_text("SNR is computed over the VAD speech span of the clean clip (not whole-file RMS).\n" + "\n".join(rows))
print("\n".join(rows))
