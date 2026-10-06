"""Run a constrained-NMT configuration on the safety set (one split) and print per-group slot preservation (Clopper-Pearson) + latency.
python tools/eval_nmt_config.py --split dev --groups medication unit number intensity --bonus 4 --tag r1 [--variant int8]
Appends one line to results/optim_log.jsonl (also when the result is worse)."""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
from run_nmt_safety import build
from tonebridge.evalkit.slotgold import item_preserved
from tonebridge.evalkit.stats import fmt_rate
from tonebridge.nmt_constraints import GlossaryConstrainer

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev"); ap.add_argument("--groups", nargs="*", default=[]); ap.add_argument("--bonus", type=float, default=0.0)
ap.add_argument("--tag", required=True); ap.add_argument("--variant", default="int8"); ap.add_argument("--meds-pref", action="store_true")
ap.add_argument("--note", default=""); ap.add_argument("--hypothesis", default=""); ap.add_argument("--eos-penalty", type=float, default=None)
a = ap.parse_args()
nmt = build(a.variant)
gc = GlossaryConstrainer(groups=a.groups, preferred_only_meds=a.meds_pref) if a.groups and a.bonus else None
items = [j for j in map(json.loads, open(ROOT / "configs/safety/safety_set_v1.jsonl", encoding="utf8")) if a.split in ("all", j["split"])]
cnt = defaultdict(lambda: [0, 0]); lat = []; hyps = {}
for it in items:
    t = time.perf_counter(); ids = nmt.encode_ids(it["vi"])
    cons = gc(it["vi"], nmt.piece_ids) if gc else None
    o, _ = nmt.greedy(ids, cons, a.bonus, a.eos_penalty)
    h = nmt.decode_ids(o); lat.append((time.perf_counter() - t) * 1000); hyps[it["id"]] = h
    ok = item_preserved(it, h)
    for g in (it["group"], "ALL"):
        cnt[g][0] += ok; cnt[g][1] += 1
rates = {g: round(k / n, 4) for g, (k, n) in cnt.items()}
print(a.tag, {g: fmt_rate(*v) for g, v in sorted(cnt.items())}, f"lat p50 {np.percentile(lat, 50):.0f} p95 {np.percentile(lat, 95):.0f} ms (contended)")
(ROOT / f"results/nmt_cfg_{a.tag}_{a.split}.jsonl").write_text("\n".join(json.dumps({"id": i, "hyp": h}, ensure_ascii=False) for i, h in hyps.items()), encoding="utf8")
with open(ROOT / "results/optim_log.jsonl", "a", encoding="utf8") as f:
    f.write(json.dumps({"metric": "M2", "tag": a.tag, "split": a.split,
                        "config": {"groups": a.groups, "bonus": a.bonus, "variant": a.variant, "meds_pref": a.meds_pref, "eos_penalty": a.eos_penalty},
                        "hypothesis": a.hypothesis, "note": a.note, "rates": rates, "n": {g: v[1] for g, v in cnt.items()},
                        "lat_p50_ms_contended": round(float(np.percentile(lat, 50)), 1), "t": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False) + "\n")
