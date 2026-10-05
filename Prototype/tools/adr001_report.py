"""Markdown table + paired-bootstrap CI for ADR-001 arms from run_adr001_grid.py output. Decision rule: an arm beats OFF only if
WER(OFF) - WER(arm) >= 2.0 points absolute (point estimate) in the cell; the CI is reported next to it."""
import json, sys
import numpy as np

r = json.load(open(sys.argv[1])); rng = np.random.default_rng(0)
cells = r["cells"]; snrs = [s for s in ["clean", "10", "5", "0"] if f"{s}|off" in cells]
arms = sorted({k.split("|")[1] for k in cells}, key=lambda a: (a != "off", a))
print(f"n_utts={r['n_utts']}  noise={r['noise_envs']}  measured SNR={r.get('measured_snr_db_mean')}\n")
print("| SNR | " + " | ".join(f"{a.upper()} WER / CER" for a in arms) + " | best arm by 2-pt rule |"); print("|---|" + "---|" * (len(arms) + 1))
for s in snrs:
    off = np.array(cells[f"{s}|off"]["per_utt"], float); row = []; best = "OFF"; gain = 0.0
    for a in arms:
        c = cells[f"{s}|{a}"]; txt = f"{100*c['WER']:.2f} / {100*c['CER']:.2f}"
        if a != "off":
            x = np.array(c["per_utt"], float); n = len(x); d = []
            for _ in range(2000):
                i = rng.integers(0, n, n); d.append(100 * (off[i, 1].sum() - x[i, 1].sum()) / off[i, 0].sum())
            diff = 100 * (off[:, 1].sum() - x[:, 1].sum()) / off[:, 0].sum()
            txt += f" (Δ {diff:+.2f} [{np.percentile(d,2.5):+.2f}, {np.percentile(d,97.5):+.2f}])"
            if diff >= 2.0 and diff > gain: best, gain = a.upper(), diff
        row.append(txt)
    print(f"| {s} | " + " | ".join(row) + f" | {best} |")
print("\nΔ = WER(OFF) − WER(arm) in points, positive = arm better; paired bootstrap over utterances, 95% CI.")
