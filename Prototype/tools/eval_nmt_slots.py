"""Slot-preservation rate of raw NMT output by safety-set group, Clopper-Pearson 95%. python tools/eval_nmt_slots.py results/nmt_safety_int8_greedy.jsonl [--split dev|test|all]"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from tonebridge.evalkit.slotgold import item_preserved
from tonebridge.evalkit.stats import fmt_rate

ap = argparse.ArgumentParser(); ap.add_argument("hyp"); ap.add_argument("--split", default="all"); a = ap.parse_args()
items = {j["id"]: j for j in map(json.loads, open(ROOT / "configs/safety/safety_set_v1.jsonl", encoding="utf8"))}
hyp = {j["id"]: j["hyp"] for j in map(json.loads, open(a.hyp, encoding="utf8"))}
cnt = defaultdict(lambda: [0, 0])
for i, h in hyp.items():
    it = items[i]
    if a.split != "all" and it["split"] != a.split: continue
    ok = item_preserved(it, h)
    for g in (it["group"], "ALL"):
        cnt[g][0] += ok; cnt[g][1] += 1
for g, (k, n) in sorted(cnt.items()): print(f"{g:<11} {fmt_rate(k, n)}")
