"""On hard dev cells (OFF arm, greedy): how often does the gate's ASR-confidence rule (REPEAT below asr_confidence_min) catch a bad transcript?
Per utterance: WER of that utterance, ASR confidence. Reports: share of utterances with utterance-WER > 30 % that have conf < threshold (caught) and
share of utterances with WER <= 30 % that are rejected (false reject), for thresholds 0.3/0.5/0.7. Dev only. python tools/asr_conf_gate_check.py"""
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

split = json.loads((ROOT / "configs/splits/fleurs_vi_dev_test.json").read_text()); byid = {u: (p, r) for u, p, r, _ in datasets.fleurs_vi()}
SEED = split["seed"]
vad, asr = SileroVad(), SherpaZipformerVi(decoder="decoder-epoch-12-avg-8.int8.onnx", threads=2)
pool = [sf.read(byid[u][0], dtype="float32")[0] for u in split["babble_pool"]]
babble = make_babble(pool, 90.0, 6, SEED + 1)
envs = []
for z in sorted((ROOT / "data/noise/demand").glob("*_16k.zip")):
    with zipfile.ZipFile(z) as zf, zf.open([n for n in zf.namelist() if n.endswith("ch01.wav")][0]) as f:
        envs.append(sf.read(f, dtype="float32")[0])
res = {}
for name, snr, tracks in (("babble", 0, [babble]), ("babble", -5, [babble]), ("demand", -5, envs), ("clean", None, None)):
    rng = np.random.default_rng(SEED + 5)
    rows = []
    for k, u in enumerate(split["dev"]):
        x = sf.read(byid[u][0], dtype="float32")[0]; ref = normalize_vi(byid[u][1])
        if snr is not None:
            t0, t1 = vad.segment(x); m = speech_mask(len(x), [(t0, t1)] if t1 > t0 else [])
            x = mix_at_snr_masked(x, tracks[k % len(tracks)], float(snr), m, rng)
        r = asr.transcribe(x, "vi"); h = normalize_vi(r.text)
        w = jiwer.process_words(ref, h if h else "<empty>"); n = len(ref.split())
        rows.append((100 * (w.substitutions + w.deletions + w.insertions) / max(n, 1), r.confidence))
    a = np.array(rows); bad = a[:, 0] > 30
    out = {"n": len(a), "n_bad_utts_wer_gt30": int(bad.sum()), "mean_conf_bad": round(float(a[bad, 1].mean()), 3) if bad.any() else None, "mean_conf_ok": round(float(a[~bad, 1].mean()), 3)}
    for th in (0.3, 0.5, 0.7):
        out[f"th{th}"] = {"caught_bad": round(float((a[bad, 1] < th).mean()), 3) if bad.any() else None, "false_reject_ok": round(float((a[~bad, 1] < th).mean()), 3) if (~bad).any() else None}
    res[f"{name}|{snr}"] = out
    print(name, snr, out, flush=True)
(ROOT / "results/asr_conf_gate_check_dev.json").write_text(json.dumps(res, indent=1))
