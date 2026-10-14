"""Atomic, fingerprinted research checkpoints; no silent mismatched resume."""
import hashlib
import json
import math
from pathlib import Path


def fingerprint(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def resume_checkpoint(path: Path, spec: dict, expected_cells: list[str], ids: list[str]) -> dict:
    """Only matching, structurally valid checkpoints may contribute old cells."""
    digest = fingerprint(spec)
    if not path.exists():
        return {"schema_version": "adr001-v3", "fingerprint": digest, "run_spec": spec,
                "ids": ids, "cells": {}, "snr_check": {}, "complete": False}
    saved = json.loads(path.read_text(encoding="utf8"))
    if (saved.get("schema_version") != "adr001-v3" or saved.get("fingerprint") != digest
            or saved.get("run_spec") != spec or saved.get("ids") != ids):
        raise ValueError("checkpoint fingerprint/config/input mismatch; use a fresh output path")
    cells = saved.get("cells")
    if not isinstance(cells, dict) or set(cells) - set(expected_cells):
        raise ValueError("unexpected checkpoint cells")
    for cell in cells.values():
        validate_cell(cell, len(ids))
    saved["complete"] = set(cells) == set(expected_cells)
    return saved


def validate_cell(cell: dict, expected_count: int) -> None:
    """Validate sample coverage, nonnegative counts and rates against the counts."""
    if (not isinstance(cell, dict) or len(cell.get("per_utt", [])) != expected_count
            or len(cell.get("hyp", [])) != expected_count or not all(isinstance(h, str) for h in cell["hyp"])):
        raise ValueError("incomplete checkpoint cell")
    for counts in cell["per_utt"]:
        if not isinstance(counts, list) or len(counts) != 4 or any(type(n) is not int or n < 0 for n in counts):
            raise ValueError("invalid checkpoint counts")
    totals = [sum(counts[i] for counts in cell["per_utt"]) for i in range(4)]
    if totals[0] <= 0 or totals[2] <= 0 or cell.get("words") != totals[0]:
        raise ValueError("invalid checkpoint reference totals")
    for key, value in (("WER", totals[1] / totals[0]), ("CER", totals[3] / totals[2])):
        if not isinstance(cell.get(key), (int, float)) or not math.isfinite(cell[key]) or not math.isclose(cell[key], value, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("checkpoint rate disagrees with counts")
