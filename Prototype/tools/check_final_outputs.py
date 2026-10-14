"""Validate every frozen-test output and atomically create the completion marker."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from benchmark_integrity import atomic_json, fingerprint, validate_cell


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(root: Path) -> dict:
    """Coverage and provenance checks; never accepts an unbound legacy result."""
    ids = json.loads((root / "configs/splits/fleurs_vi_dev_test.json").read_text(encoding="utf8"))["test"]
    outputs = []
    for noise in ("demand", "babble", "alarm"):
        path = root / f"results/final_test_m1_{noise}.json"
        result = json.loads(path.read_text(encoding="utf8"))
        spec = result.get("run_spec", {})
        cells = [f"{noise}|{snr}|off" for snr in (["clean"] if noise == "demand" else []) + ["10", "5", "0", "-5"]]
        if (result.get("schema_version") != "adr001-v3" or result.get("complete") is not True
                or result.get("fingerprint") != fingerprint(spec) or result.get("ids") != ids
                or spec.get("split") != "test" or spec.get("noise") != noise or spec.get("arms") != ["off"]
                or spec.get("config_file_sha256") != sha(root / "configs/pipeline.json")
                or set(result.get("cells", {})) != set(cells)):
            raise ValueError("M1 provenance/coverage mismatch")
        for cell in result["cells"].values():
            validate_cell(cell, len(ids))
        outputs.append(path)
    refs = [json.loads(line) for line in (root / "configs/safety/safety_set_v1.jsonl").read_text(encoding="utf8").splitlines()]
    safety_ids = {row["id"] for row in refs if row["split"] == "test"}
    path = root / "results/nmt_cfg_final_test_test.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    if len(rows) != len(safety_ids) or {row.get("id") for row in rows} != safety_ids or not all(isinstance(row.get("hyp"), str) for row in rows):
        raise ValueError("M2 incomplete hypotheses")
    outputs.append(path)
    path = root / "results/safety_eval_test_final.json"
    result = json.loads(path.read_text(encoding="utf8"))
    if result.get("real", {}).get("n_items") != len(safety_ids) or not isinstance(result.get("seeded"), dict):
        raise ValueError("M3 incomplete evaluation")
    outputs.append(path)
    process_rows = [json.loads(line) for line in (root / "results/final_test_processes.jsonl").read_text().splitlines()]
    if len(process_rows) != 5 or {row.get("task") for row in process_rows} != {"m1_demand", "m1_babble", "m1_alarm", "m2", "m3"} or any(row.get("exit_code") != 0 for row in process_rows):
        raise ValueError("not all subprocesses succeeded")
    return {"status": "COMPLETE", "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "outputs_sha256": {path.name: sha(path) for path in outputs}, "subprocesses": process_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    if args.marker.exists(): parser.error("completion marker already exists")
    try:
        result = validate(args.root)
    except Exception:
        parser.error("final output validation failed; completion marker not created")
    atomic_json(args.marker, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
