"""Inventory already-approved local assets. No network/download and no conversion."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", required=True, help="40-character upstream commit, obtained from approved acquisition")
    args = ap.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.revision):
        raise ValueError("revision_must_be_pinned_commit")
    d = args.model_dir.resolve()
    if not d.is_dir():
        raise FileNotFoundError(d)
    files = {}
    for path in sorted(d.iterdir()):
        if path.is_file() and path.suffix in (".json", ".spm", ".safetensors") and path.name != "asset_manifest.json":
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    required = {"config.json", "source.spm", "target.spm", "vocab.json"}
    if not required.issubset(files) or not any(name.endswith(".safetensors") for name in files):
        raise ValueError("incomplete_local_assets")
    manifest = {"model_id": args.model_id, "revision": args.revision, "files": files,
                "provenance": "declared upstream revision; local hashes verify consistency, not origin"}
    (d / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"model_id": args.model_id, "revision": args.revision,
                      "bytes": sum(f["bytes"] for f in files.values())}))


if __name__ == "__main__":
    main()
