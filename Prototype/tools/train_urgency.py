"""Urgency pipeline entry point. Two subcommands:
  extract  <recordings_dir> -> results/urgency_features.json   (21 features per recording; noise-augmented copies clean/10/5/0 for TRAINING folds)
  loso     results/urgency_features.json --mode G|S             (leave-one-speaker-out, per-speaker table, MLP vs z-score baseline)
  selfcheck                                                    (seeded FAKE data, only proves the code runs; prints no metrics, writes nothing to results/)
No MLP is trained and no urgency F1 is reported until real team recordings exist (recording_kit/). Recording file name: <spk>_<NEU|URG>_<sid>.wav|flac.
"""
import argparse, json, sys, zlib
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def extract(rec_dir: Path, out: Path) -> None:
    from tonebridge.branch_b import FEATURE_NAMES, BranchB
    from tonebridge.evalkit import datasets
    from tonebridge.evalkit.noise import mix_at_snr_masked, speech_mask
    from tonebridge.evalkit.noisegen import make_babble
    from tonebridge.stages.vad_silero import SileroVad

    bb, vad = BranchB(), SileroVad()
    items = list(datasets.recording_kit(rec_dir))
    if not items:
        sys.exit("no recordings found: nothing to extract (this is expected before the team recordings exist)")
    split = json.loads((ROOT / "configs/splits/fleurs_vi_dev_test.json").read_text())
    byid = {u: p for u, p, _, _ in datasets.fleurs_vi()}
    pool = [sf.read(byid[u], dtype="float32")[0] for u in split["babble_pool"][:80]]
    noise = make_babble(pool, 60.0, 6, 5)
    rows = []
    for uid, path, ref, tag in items:
        spk = uid.split("_")[0]
        x, sr = sf.read(path, dtype="float32")
        x = x.mean(1) if x.ndim == 2 else x
        assert sr == 16000, f"{path}: resample to 16 kHz first (no silent resampling)"
        t0, t1 = vad.segment(x)
        seg = x[int(t0 * sr): int(t1 * sr)] if t1 > t0 else x
        m = speech_mask(len(seg), [(0.0, len(seg) / sr)])
        for snr in ["clean", "10", "5", "0"]:
            y = seg if snr == "clean" else mix_at_snr_masked(seg, noise, float(snr), m, np.random.default_rng(zlib.crc32(uid.encode())))
            f, nv = bb.features(y)
            rows.append({"speaker": spk, "label": "HIGH" if tag == "URG" else "LOW", "snr": snr, "utt": uid, "x": [f[k] for k in FEATURE_NAMES], "n_voiced": nv})
    out.write_text(json.dumps({"feature_names": FEATURE_NAMES, "rows": rows}))
    print("wrote", out, len(rows))


def do_loso(path: Path, mode: str) -> None:
    from tonebridge.urgency_train import loso
    d = json.loads(path.read_text())
    o = loso(d["rows"], mode=mode, seed=0)
    print(f"LOSO mode {mode}: speakers = {[f.speaker for f in o['mlp']]}")
    print("speaker | MLP macroF1 / HIGH recall | baseline macroF1 / HIGH recall | n_test")
    for m, b in zip(o["mlp"], o["baseline"]):
        print(f"{m.speaker} | {m.macro_f1:.3f} / {m.high_recall:.3f} | {b.macro_f1:.3f} / {b.high_recall:.3f} | {m.n_test}")
    print(json.dumps(o["mean_std"], indent=1))
    (ROOT / f"results/urgency_loso_{mode}.json").write_text(json.dumps({"mode": mode, "mean_std": o["mean_std"], "per_speaker": {
        arm: [f.__dict__ for f in o[arm]] for arm in ("mlp", "baseline")}}, indent=1))


def selfcheck() -> None:
    from tonebridge.urgency_train import loso, synthetic_rows
    for mode in ("G", "S"):
        o = loso(synthetic_rows(4, 8, 0), mode=mode, epochs=20)
        assert len(o["mlp"]) == 4
    print("selfcheck OK (fake data; metrics intentionally not printed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("rec_dir", type=Path); e.add_argument("--out", type=Path, default=ROOT / "results/urgency_features.json")
    l = sub.add_parser("loso"); l.add_argument("features", type=Path); l.add_argument("--mode", choices=["G", "S"], default="G")
    sub.add_parser("selfcheck")
    a = ap.parse_args()
    {"extract": lambda: extract(a.rec_dir, a.out), "loso": lambda: do_loso(a.features, a.mode), "selfcheck": selfcheck}[a.cmd]()
