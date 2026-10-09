"""Explicit setup for the experimental EN-KO replacement. No clinical release claim."""
import json
from pathlib import Path
import shutil
import zipfile

from provision_laptop import ROOT, MODELS, download, digest

URL = "https://data.argosopentech.com/argospm/v1/translate-en_ko-1_1.argosmodel"
ARCHIVE_SHA256 = "e03d8e65e6d44525ec5808c3409fcf8728c76c2c76925372b6d3dc3278de17fc"
INDEX_REVISION = "ff90de60728f7c1338ff6b75974e4c89b2442d22"


def provision():
    packed = download(URL, MODELS / "downloads/translate-en_ko-1_1.argosmodel")
    if digest(packed) != ARCHIVE_SHA256:
        raise ValueError("archive_integrity_mismatch")
    destination = MODELS / "nmt-laptop/en-ko-argos"
    destination.mkdir(parents=True, exist_ok=True)
    names = ["sentencepiece.model", "model/model.bin", "model/shared_vocabulary.txt", "metadata.json", "README.md"]
    with zipfile.ZipFile(packed) as archive:
        for name in names:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open("en_ko/" + name) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)
    metadata = {"model_id": "argosopentech/translate-en_ko-1_1", "revision": INDEX_REVISION,
                "source_url": URL, "archive_sha256": ARCHIVE_SHA256,
                "license_status": "Package does not supply a license; release requires separate review",
                "files": {name: {"bytes": (destination / name).stat().st_size, "sha256": digest(destination / name)} for name in names}}
    (destination / "asset_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Select only after real local constructor, tokenizer probes and EOS succeed.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from tonebridge.stages.nmt_argos import ArgosEnKoNmt
    nmt = ArgosEnKoNmt(destination)
    result = nmt.translate("Hello.", "en", "ko")
    if not result.evidence.eos_reached or result.evidence.tokenizer_issues or result.evidence.source_unknown_tokens or result.evidence.output_unknown_tokens:
        raise ValueError("candidate_smoke_failed")
    registry_path = ROOT / "configs/nmt/models_laptop.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry["models"]["en-ko"].get("backend") != "argos":
        registry["quarantined_en_ko"] = registry["models"]["en-ko"]
    registry["models"]["en-ko"] = {"backend": "argos", "adapter": "ArgosEnKoNmt", "model_id": metadata["model_id"],
        "revision": INDEX_REVISION, "assets": "models/nmt-laptop/en-ko-argos", "status": "experimental_local_smoke_passed",
        "license_status": metadata["license_status"], "korean_safety": "confirmation_required_pending_human_review"}
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    provision()
