"""Strict ID-complete text evaluation; references are drafts, never clinical gold.

--references-only evaluates checker coverage on supplied references, NOT NMT.
Human semantic labels are external to the checker and keyword dictionaries.
"""
from collections import Counter
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tonebridge.clinical_safety import ClinicalSafetyChecker
from tonebridge.nmt_evidence import DIRECTIONS, EvidenceMtResult


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_ids:" + str(path))
    return rows


def evaluate(items, hypotheses=None, labels=None):
    if hypotheses is not None:
        expected = {r["id"] for r in items}
        actual = set(hypotheses)
        if actual != expected:
            raise ValueError(f"hypothesis_ids_mismatch:missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    counts = Counter()
    decisions = []
    for row in items:
        checker = ClinicalSafetyChecker(row["src_lang"], row["tgt_lang"])
        if hypotheses is None:
            hypothesis = row["reference"]
            report = checker.check_texts(row["source"], hypothesis)
        else:
            mt = EvidenceMtResult.model_validate(hypotheses[row["id"]]["translation"])
            if (mt.src_text, mt.src_lang, mt.tgt_lang) != (row["source"], row["src_lang"], row["tgt_lang"]):
                raise ValueError("source_or_direction_mismatch:" + row["id"])
            hypothesis = mt.tgt_text
            report = checker.check(mt)
        counts[report.status] += 1
        label = (labels or {}).get(row["id"])
        if label is not None:
            if not label.get("reviewer") or not isinstance(label.get("clinical_correct"), bool):
                raise ValueError("invalid_human_label:" + row["id"])
            correct = label["clinical_correct"]
            counts["human_reviewed"] += 1
            counts["human_correct" if correct else "human_incorrect"] += 1
            counts["false_accept"] += int(not correct and report.status == "PASS")
            counts["false_block"] += int(correct and report.status != "PASS")
        decisions.append({"id": row["id"], "source": row["source"], "hypothesis": hypothesis,
                          "reference": row["reference"], "reference_status": row["reference_status"],
                          "safety": report.model_dump()})
    result = {"kind": "checker_on_draft_references" if hypotheses is None else "nmt_outputs",
            "n_items": len(items), "counts": dict(counts), "rows": decisions,
            "limitations": ["No keyword score is clinical gold", "Draft references require bilingual review",
                            "Korean automatic speech remains blocked pending review"]}
    if hypotheses is not None:
        from sacrebleu.metrics import CHRF
        result["draft_reference_chrf"] = CHRF().corpus_score(
            [r["hypothesis"] for r in decisions], [[r["reference"] for r in decisions]]).score
        result["chrf_is_not_clinical_correctness"] = True
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("configs/nmt/text_references_v1.jsonl"))
    ap.add_argument("--direction", choices=(*DIRECTIONS, "all"), default="all")
    ap.add_argument("--split", choices=("dev", "holdout"), default="dev")
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--hyp", type=Path)
    modes.add_argument("--references-only", action="store_true")
    ap.add_argument("--human-labels", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    items = [r for r in read_rows(args.dataset) if r["split"] == args.split
             and (args.direction == "all" or r["direction"] == args.direction)]
    if not items:
        raise ValueError("empty_evaluation")
    hypotheses = {r["id"]: r for r in read_rows(args.hyp)} if args.hyp else None
    labels = {r["id"]: r for r in read_rows(args.human_labels)} if args.human_labels else None
    if labels and not set(labels).issubset({r["id"] for r in items}):
        raise ValueError("human_labels_outside_evaluation")
    result = evaluate(items, hypotheses, labels)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
