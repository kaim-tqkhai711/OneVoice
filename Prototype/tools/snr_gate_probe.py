"""EXPLORATORY (dev only, no pipeline change): can a cheap blind SNR estimate flag utterances the ASR gets wrong, where the ASR confidence cannot (see
results/asr_conf_gate_check_dev.json)? Estimate = 10*log10(P_speech / P_noisefloor) from 20 ms frame energies of the noisy clip: speech = mean of the top
30 % frames, noise floor = mean of the bottom 20 % frames. Pure numpy (portable). Bad utterance = utterance WER > 30 %.
Reports AUC (SNR estimate vs bad) and, at the threshold that keeps false rejects of good utterances <= 10 %, the share of bad utterances caught.
python tools/snr_gate_probe.py"""
import json, sys, zipfile
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonebridge.evalkit import datasets
from tonebridge.evalkit.noise import mix_at_snr_masked, speech_mask
from tonebridge.evalkit.noisegen import make_babble
from tonebridge.evalkit.textnorm import normalize_vi
from tonebridge.stages.asr_sherpa import SherpaZipformerVi
from tonebridge.stages.vad_silero import SileroVad


def blind_snr_db(x: np.ndarray, sr: int = 16000) -> float:
    n = int(0.02 * sr)
    f = x[: len(x) // n * n].reshape(-1, n).astype(np.float64)
    e = np.sort((f ** 2).mean(1)) + 1e-12
    k = len(e)
    return float(10 * np.log10(e[int(0.7 * k):].mean() / e[: max(int(0.2 * k), 1)].mean()))


def auc(score_bad: np.ndarray, score_ok: np.ndarray) -> float:
    """P(score of a bad utterance is LOWER than that of a good one) (ties 0.5)."""
    a = (score_bad[:, None] < score_ok[None, :]).mean() + 0.5 * (score_bad[:, None] == score_ok[None, :]).mean()
    return float(a)


split = json.loads((ROOT / "configs/splits/fleurs_vi_dev_test.json").read_text()); byid = {u: (p, r) for u, p, r, _ in datasets.fleurs_vi()}
SEED = split["seed"]
vad, asr = SileroVad(), SherpaZipformerVi(decoder="decoder-epoch-12-avg-8.int8.onnx", threads=2)
pool = [sf.read(byid[u][0], dtype="float32")[0] for u in split["babble_pool"]] if False else [sf.read(byid[u][0], dtype="float32")[0] for u in split["babble_pool"]]
babble = make_babble(pool, 90.0, 6, SEED + 1)
envs = []
for z in sorted((ROOT / "data/noise/demand").glob("*_16k.zip")):
    with zipfile.ZipFile(z) as zf, zf.open([n for n in zf.namelist() if n.endswith("ch01.wav")][0]) as fh:
        envs.append(sf.read(fh, dtype="float32")[0])
rows = []  # (cell, wer, blind_snr)
for name, snr, tracks in (("clean", None, None), ("demand", 5, envs), ("demand", 0, envs), ("demand", -5, envs), ("babble", 5, [babble]), ("babble", 0, [babble]), ("babble", -5, [babble])):
    rng = np.random.default_rng(SEED + 11)
    for k, u in enumerate(split["dev"]):
        x = sf.read(byid[u][0], dtype="float32")[0]; ref = normalize_vi(byid[u][1])
        if snr is not None:
            t0, t1 = vad.segment(x); m = speech_mask(len(x), [(t0, t1)] if t1 > t0 else [])
            x = mix_at_snr_masked(x, tracks[k % len(tracks)], float(snr), m, rng)
        h = normalize_vi(asr.transcribe(x, "vi").text)
        w = jiwer.process_words(ref, h if h else "<empty>")
        rows.append((f"{name}|{snr}", 100 * (w.substitutions + w.deletions + w.insertions) / max(len(ref.split()), 1), blind_snr_db(x)))
    print(name, snr, "done", flush=True)
a = np.array([[r[1], r[2]] for r in rows]); bad = a[:, 0] > 30
out = {"n": len(a), "n_bad": int(bad.sum()), "auc_bad_has_lower_blind_snr": round(auc(a[bad, 1], a[~bad, 1]), 3)}
thr = float(np.percentile(a[~bad, 1], 10))  # keeps ~10 % of good utterances below the threshold
out.update({"threshold_db": round(thr, 2), "false_reject_good": round(float((a[~bad, 1] < thr).mean()), 3), "caught_bad": round(float((a[bad, 1] < thr).mean()), 3)})
out["by_cell_mean_blind_snr"] = {c: round(float(np.mean([r[2] for r in rows if r[0] == c])), 1) for c in dict.fromkeys(r[0] for r in rows)}
print(json.dumps(out, indent=1)); (ROOT / "results/snr_gate_probe_dev.json").write_text(json.dumps(out, indent=1))
