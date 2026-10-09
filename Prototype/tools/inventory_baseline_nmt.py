"""Inventory the pinned public ONNX VI-EN export acquired by provision_laptop."""
import json
from provision_laptop import ROOT, MODELS, digest


def inventory():
    directory = MODELS / "nmt/vi-en-int8-arm64"
    names = ["config.json", "source.spm", "target.spm", "vocab.json", "encoder_model_quantized.onnx", "decoder_model_merged_quantized.onnx"]
    # Existing recorded assets must match before recording an export identity.
    previous = json.loads((ROOT / "configs/laptop_assets_manifest.json").read_text(encoding="utf-8"))
    known = {row["path"]: row for row in previous["files"]}
    rows = {}
    for name in names:
        path = directory / name
        item = {"bytes": path.stat().st_size, "sha256": digest(path)}
        if known.get(path.relative_to(ROOT).as_posix(), {}).get("sha256") != item["sha256"]:
            raise ValueError("baseline_differs_from_recorded_assets:" + name)
        rows[name] = item
    metadata = {"model_id": "Xenova/opus-mt-vi-en", "revision": "541dfd72ec07e8f1563795e0224408ad8d729e80", "files": rows}
    (directory / "asset_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    registry_path = ROOT / "configs/nmt/models_laptop.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["models"]["vi-en"] = {"backend": "onnx", "adapter": "OrtMarianNmt", "model_id": metadata["model_id"],
        "revision": metadata["revision"], "assets": "models/nmt/vi-en-int8-arm64", "status": "local_assets_ready"}
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in metadata.items() if k != "files"}))


if __name__ == "__main__":
    inventory()
