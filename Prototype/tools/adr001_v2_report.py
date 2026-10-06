"""ADR-001 v2 table: WER by noise type x SNR x arm, paired bootstrap CI of (WER_off - WER_arm), and the verdict by the >= 2-point absolute rule.
python tools/adr001_v2_report.py dev   -> results/adr001_v2_dev_table.md"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
split = sys.argv[1] if len(sys.argv) > 1 else "dev"
cells, snr_check = {}, {}
for p in sorted((ROOT / "results").glob(f"adr001_v2_{split}_*.json")):
    d = json.loads(p.read_text())
    cells.update(d["cells"]); snr_check.update(d.get("snr_check", {}))
noises = ["demand", "babble", "alarm"]; snrs = ["clean", "10", "5", "0", "-5"]
arms = ["off", "on", "oa0.25", "oa0.5", "oa0.75", "cns"]
rng = np.random.default_rng(0)


def paired(a, b):  # per_utt cumulative -> per-utt (words, errs); returns mean/CI of WER(a)-WER(b) in points
    A, B = np.diff(np.array(cells[a]["per_utt"]), axis=0, prepend=0), np.diff(np.array(cells[b]["per_utt"]), axis=0, prepend=0)
    # per_utt rows were stored as increments already
    A, B = np.array(cells[a]["per_utt"]), np.array(cells[b]["per_utt"])
    n = len(A); idx = rng.integers(0, n, (1000, n))
    d = 100 * ((A[idx, 1].sum(1)) / A[idx, 0].sum(1) - (B[idx, 1].sum(1)) / B[idx, 0].sum(1))
    return 100 * (A[:, 1].sum() / A[:, 0].sum() - B[:, 1].sum() / B[:, 0].sum()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


L = [f"### ADR-001 v2 ({split}); WER %, SNR over the VAD speech mask. Delta = WER(off) - WER(arm), positive = arm better; [95% paired bootstrap]. WIN = Delta >= 2.0 points.", ""]
wins = {a: 0 for a in arms[1:]}; ncell = 0
for nz in noises:
    L += [f"**{nz}**", "", "| SNR | " + " | ".join(arms) + " | arm that beats OFF by >= 2 pts |", "|---|" + "---|" * (len(arms) + 1)]
    for s in snrs:
        if s == "clean" and nz != "demand":
            continue
        k = f"{nz if s != 'clean' else 'demand'}|{s}|"
        if k + "off" not in cells:
            L.append(f"| {s} | (not run) |"); continue
        row, best = [], []
        for a in arms:
            if k + a not in cells:
                row.append("-"); continue
            w = cells[k + a]["WER"] * 100
            if a == "off":
                row.append(f"{w:.2f}")
            else:
                dm, lo, hi = paired(k + "off", k + a)
                row.append(f"{w:.2f} ({dm:+.2f} [{lo:+.2f},{hi:+.2f}])")
                if dm >= 2.0:
                    best.append(a); wins[a] += 1
        ncell += 1
        L.append(f"| {s} | " + " | ".join(row) + f" | {', '.join(best) or 'none (OFF)'} |")
    L.append("")
L.append(f"Cells compared: {ncell}. Cells where an arm beat OFF by >= 2 points: {wins}")
L.append("Achieved SNR check (over mask): " + json.dumps(snr_check))
out = "\n".join(L)
(ROOT / f"results/adr001_v2_{split}_table.md").write_text(out, encoding="utf8"); print(out)
