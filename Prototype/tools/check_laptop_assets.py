"""Verify current restoration manifest without downloading or editing assets."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(manifest):
    errors = []
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT / "models"):
            errors.append({"path": item["path"], "reason": "outside_models"})
            continue
        if not path.is_file():
            errors.append({"path": item["path"], "reason": "missing"})
        elif path.stat().st_size != item["bytes"]:
            errors.append({"path": item["path"], "reason": "size_mismatch"})
        else:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                    errors.append({"path": item["path"], "reason": "hash_mismatch"})
    return errors


if __name__ == "__main__":
    path = ROOT / "configs/laptop_assets_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors = verify(manifest)
    print(json.dumps({"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n_files": len(manifest["files"]), "errors": errors}, indent=2))
    raise SystemExit(bool(errors))
