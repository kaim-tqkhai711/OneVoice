"""Markdown table for the 30 clinical sentences: source | fp32 | INT8 arm64 | INT8enc+fp32dec | INT8 beam4, with safety-check flags and
which build the error belongs to. Flags are automatic (Semantic Safety Check v1 critical/confirm); 'SAFETY' marks critical flags. Writes results/nmt30_table.md"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from tonebridge.safety import SemanticSafetyChecker
d = json.loads((ROOT / "results/nmt30.json").read_text(encoding="utf8")); c = SemanticSafetyChecker()
V = ["fp32|greedy", "int8|greedy", "int8enc_fp32dec|greedy", "int8|beam4", "fp32|beam4"]
rows = ["| # | Source (vi) | fp32 | INT8 arm64 | INT8 enc + fp32 dec | INT8 beam4 | flags (check v1) | error belongs to |", "|---|---|---|---|---|---|---|---|"]
summ = {v: 0 for v in V}; attr = {"both": 0, "int8_only": 0, "fp32_only": 0, "none": 0}
for i, s in enumerate(d["sentences"]):
    hy = {v: d["variants"][v]["hyp"][i] for v in V if v in d["variants"]}
    rep = {v: c.check_texts(s, hy[v]) for v in hy}
    fl = {v: ("SAFETY" if not r.passed else "CONFIRM" if r.confirm else "") for v, r in rep.items()}
    for v in V:
        if v in fl and fl[v]: summ[v] += 1
    f32, i8 = bool(fl["fp32|greedy"]), bool(fl["int8|greedy"])
    a = "both" if f32 and i8 else "int8_only" if i8 else "fp32_only" if f32 else "none"; attr[a] += 1
    flags = "; ".join(f"{k.split('|')[0]}/{k.split('|')[1]}:{v}" for k, v in fl.items() if v) or "-"
    esc = lambda t: t.replace("|", "\|")
    rows.append(f"| {i+1} | {esc(s)} | {esc(hy['fp32|greedy'])} | {esc(hy['int8|greedy'])} | {esc(hy['int8enc_fp32dec|greedy'])} | {esc(hy['int8|beam4'])} | {flags} | {a} |")
out = "\n".join(rows) + f"\n\nFlagged sentences per build (critical or confirm): {summ}\nAttribution fp32-vs-INT8 greedy: {attr}\n"
(ROOT / "results/nmt30_table.md").write_text(out, encoding="utf8"); print(out)
