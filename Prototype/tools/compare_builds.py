"""Compare NMT builds on the safety set: per group primary-slot preservation (all items), and INT8-only / fp32-only error counts.
python tools/compare_builds.py results/nmt_safety_fp32_greedy.jsonl results/nmt_safety_int8_greedy.jsonl [...]"""
import json, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from tonebridge.evalkit.slotgold import item_preserved
from tonebridge.evalkit.stats import fmt_rate
items = {j["id"]: j for j in map(json.loads, open(ROOT / "configs/safety/safety_set_v1.jsonl", encoding="utf8"))}
runs = {Path(p).stem.replace("nmt_safety_", ""): {j["id"]: j["hyp"] for j in map(json.loads, open(p, encoding="utf8"))} for p in sys.argv[1:]}
names = list(runs)
for n in names:
    cnt = defaultdict(lambda: [0, 0])
    for i, h in runs[n].items():
        ok = item_preserved(items[i], h)
        for g in (items[i]["group"], "ALL"): cnt[g][0] += ok; cnt[g][1] += 1
    print(n, {g: fmt_rate(*v) for g, v in sorted(cnt.items())})
if len(names) >= 2:
    a, b = names[0], names[1]
    only_b = [i for i in items if i in runs[a] and i in runs[b] and item_preserved(items[i], runs[a][i]) and not item_preserved(items[i], runs[b][i])]
    only_a = [i for i in items if i in runs[a] and i in runs[b] and not item_preserved(items[i], runs[a][i]) and item_preserved(items[i], runs[b][i])]
    print(f"errors only in {b}: {len(only_b)}; only in {a}: {len(only_a)}; n={len(items)}")
    for i in only_b[:12]: print("  ", b, "|", items[i]["vi"], "|", runs[a][i], "|", runs[b][i])
