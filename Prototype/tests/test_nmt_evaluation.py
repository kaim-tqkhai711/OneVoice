import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("text_eval", ROOT / "tools/evaluate_text_nmt.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_dataset_complete_partitioned_and_references_explicitly_draft():
    rows = module.read_rows(ROOT / "configs/nmt/text_references_v1.jsonl")
    assert len(rows) == 160
    dev, holdout = set(), set()
    for row in rows:
        assert row["source"].strip() and row["reference"].strip()
        assert row["direction"] == row["src_lang"] + "-" + row["tgt_lang"]
        assert row["reference_status"] == "draft_requires_bilingual_review"
        (dev if row["split"] == "dev" else holdout).add(row["intent_id"])
    assert not dev.intersection(holdout)
    for direction in ("vi-en", "en-vi", "en-ko", "ko-en"):
        assert len([r for r in rows if r["direction"] == direction and r["split"] == "dev"]) == 20
        assert len([r for r in rows if r["direction"] == direction and r["split"] == "holdout"]) == 20


def test_missing_or_extra_hypothesis_not_silently_skipped():
    items = module.read_rows(ROOT / "configs/nmt/text_references_v1.jsonl")[:1]
    with pytest.raises(ValueError, match="hypothesis_ids_mismatch"):
        module.evaluate(items, {})


def test_references_only_does_not_report_nmt_accuracy():
    items = [r for r in module.read_rows(ROOT / "configs/nmt/text_references_v1.jsonl")
             if r["split"] == "dev" and r["direction"] == "en-ko"]
    result = module.evaluate(items)
    assert result["kind"] == "checker_on_draft_references"
    assert result["counts"].get("PASS", 0) == 0
    assert "draft_reference_chrf" not in result


def test_unverified_human_labels_rejected():
    items = module.read_rows(ROOT / "configs/nmt/text_references_v1.jsonl")[:1]
    with pytest.raises(ValueError, match="invalid_human_label"):
        module.evaluate(items, labels={items[0]["id"]: {"clinical_correct": True}})
