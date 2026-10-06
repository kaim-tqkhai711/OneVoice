"""Append an M3 line to results/optim_log.jsonl from an eval_safety JSON. python tools/log_m3.py <eval.json> <tag> "<hypothesis>" "<note>" """
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
o = json.loads(Path(sys.argv[1]).read_text(encoding="utf8")); r = o["real"]
line = {"metric": "M3", "tag": sys.argv[2], "split": "dev", "hypothesis": sys.argv[3], "note": sys.argv[4],
        "recall_real": r["recall"][0] / max(r["recall"][1], 1), "n_errors": r["recall"][1], "false_block_any": r["false_block_any"][0] / max(r["false_block_any"][1], 1),
        "n_ok": r["false_block_any"][1], "seeded_recall": o["seeded"]["recall"][0] / o["seeded"]["recall"][1], "t": time.strftime("%Y-%m-%d %H:%M:%S")}
open(ROOT / "results/optim_log.jsonl", "a", encoding="utf8").write(json.dumps(line) + "\n"); print(line)
