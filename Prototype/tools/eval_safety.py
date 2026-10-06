"""Semantic Safety Check v1 evaluation on two sets (VI->EN), each number with Clopper-Pearson 95% CI and n.
 Set 1  seeded errors : synthetic corruptions of the reference English (item["seeded"]); recall by error type; false-block on the correct references.
 Set 2  real NMT errors: the NMT's own outputs on the safety set; error = a checkable gold slot is not preserved; recall on those, false-block on the rest.
Checkable gold slot kinds: negation, medication, allergen, allergy, dose_number, dose_unit, intensity (symptom errors are out of scope of v1).
false-block = check says passed=False (critical) OR confirm=True (severity); both shown separately too.
python tools/eval_safety.py --hyp results/nmt_safety_int8_greedy.jsonl --split dev [--out results/safety_eval_dev.json]"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonebridge.evalkit.slotgold import slot_ok
from tonebridge.evalkit.stats import clopper_pearson, fmt_rate
from tonebridge.safety import SemanticSafetyChecker

CHECKABLE = {"negation", "medication", "allergen", "allergy", "dose_number", "dose_unit", "intensity"}


def flagged(rep) -> bool:
    return (not rep.passed) or rep.confirm


def real_error(item: dict, hyp: str) -> bool:
    return any(not slot_ok(s, hyp) for s in item["slots"] if s["kind"] in CHECKABLE)


def evaluate(items: list[dict], hyp: dict[str, str], checker: SemanticSafetyChecker) -> dict:
    out: dict = {"seeded": {}, "real": {}}
    # --- set 1
    rec, fb_crit, fb_any, per_type = [0, 0], [0, 0], [0, 0], defaultdict(lambda: [0, 0])
    for it in items:
        r = checker.check_texts(it["vi"], it["en_ref"])
        fb_crit[1] += 1; fb_any[1] += 1
        fb_crit[0] += (not r.passed); fb_any[0] += flagged(r)
        for typ, txt in it["seeded"]:
            f = flagged(checker.check_texts(it["vi"], txt))
            rec[0] += f; rec[1] += 1
            per_type[typ][0] += f; per_type[typ][1] += 1
    out["seeded"] = {"recall": rec, "false_block_critical_on_correct_refs": fb_crit, "false_block_any_on_correct_refs": fb_any, "recall_by_type": dict(per_type)}
    # --- set 2
    err = [0, 0]; fb = [0, 0]; fbc = [0, 0]; by_group = defaultdict(lambda: {"err": [0, 0], "ok": [0, 0]}); real_errors = []
    for it in items:
        h = hyp.get(it["id"])
        if h is None:
            continue
        r = checker.check_texts(it["vi"], h)
        e = real_error(it, h)
        tgt = by_group[it["group"]]["err" if e else "ok"]
        tgt[1] += 1
        if e:
            err[1] += 1; err[0] += flagged(r); tgt[0] += flagged(r)
            real_errors.append({"id": it["id"], "group": it["group"], "template": it["template"], "split": it["split"], "vi": it["vi"], "hyp": h, "ref": it["en_ref"],
                                "failed_slots": [s["kind"] for s in it["slots"] if not slot_ok(s, h)], "flagged": flagged(r), "reasons": r.reasons})
        else:
            fb[1] += 1; fb[0] += flagged(r); fbc[1] += 1; fbc[0] += (not r.passed); tgt[0] += flagged(r)
    out["real"] = {"recall": err, "false_block_any": fb, "false_block_critical": fbc, "by_group": {g: v for g, v in by_group.items()}, "n_items": len([1 for it in items if it["id"] in hyp])}
    out["real_errors"] = real_errors
    return out


def report(o: dict) -> str:
    L = ["### Set 1: seeded (synthetic) errors", f"- detection recall: {fmt_rate(*o['seeded']['recall'])}"]
    for t, (k, n) in sorted(o["seeded"]["recall_by_type"].items()):
        L.append(f"  - {t}: {fmt_rate(k, n)}")
    L += [f"- false-block (critical) on correct references: {fmt_rate(*o['seeded']['false_block_critical_on_correct_refs'])}",
          f"- false-block (critical or confirm) on correct references: {fmt_rate(*o['seeded']['false_block_any_on_correct_refs'])}",
          "### Set 2: real NMT errors", f"- NMT outputs with >= 1 checkable slot error: {o['real']['recall'][1]} of {o['real']['n_items']}",
          f"- detection recall: {fmt_rate(*o['real']['recall'])}",
          f"- false-block (critical or confirm) on error-free NMT outputs: {fmt_rate(*o['real']['false_block_any'])}",
          f"- false-block (critical only): {fmt_rate(*o['real']['false_block_critical'])}"]
    for g, v in sorted(o["real"]["by_group"].items()):
        L.append(f"  - {g}: recall {fmt_rate(*v['err'])}; false-block {fmt_rate(*v['ok'])}")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True); ap.add_argument("--split", default="dev"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    items = [j for j in map(json.loads, open(ROOT / "configs/safety/safety_set_v1.jsonl", encoding="utf8")) if a.split in ("all", j["split"])]
    hyp = {j["id"]: j["hyp"] for j in map(json.loads, open(a.hyp, encoding="utf8"))}
    o = evaluate(items, hyp, SemanticSafetyChecker())
    print(report(o))
    if a.out:
        Path(a.out).write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf8")
