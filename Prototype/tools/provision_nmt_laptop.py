"""Explicit pinned checkpoint setup. Audit tokenizers before downloading weights.

Upstream .bin checkpoints are loaded with torch weights_only=True and converted
locally to safetensors. Inference never fetches assets or repairs vocabulary IDs.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from provision_laptop import download, digest
from tonebridge.stages.nmt_marian import tokenizer_audit

SOURCES = {
    "en-vi": ("Helsinki-NLP/opus-mt-en-vi", "989c9fb9ec63987901022baf0182dcec3e149be6", "pytorch_model.bin"),
    "ko-en": ("Helsinki-NLP/opus-mt-ko-en", "e42d1f41b66194e6d10512f8a27bebc1f4f5097e", "pytorch_model.bin"),
    "en-ko": ("Helsinki-NLP/opus-mt-tc-big-en-ko", "ae8606b7b29a495f31ce679cee2007f536a3a5ce", "model.safetensors"),
}


def provision(direction, metadata_only=False):
    from transformers import MarianConfig, MarianTokenizer, MarianMTModel
    repo, revision, weights = SOURCES[direction]
    destination = ROOT / "models/nmt-laptop" / direction
    upstream = destination / "_upstream"
    names = ["config.json", "generation_config.json", "tokenizer_config.json", "source.spm", "target.spm", "vocab.json", "README.md"]
    for name in names:
        download(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", upstream / name)
    config = MarianConfig.from_pretrained(str(upstream), local_files_only=True)
    tokenizer = MarianTokenizer.from_pretrained(str(upstream), local_files_only=True,
                                              separate_vocabs=getattr(config, "separate_vocabs", False))
    issues = tokenizer_audit(tokenizer, config,
                            ["안녕하세요."] if direction.startswith("ko") else ["Hello.", "I need help."])
    warnings = []
    if direction in ("en-vi", "ko-en"):
        # Only these frozen published vocabularies use this policy. EN-KO remains
        # quarantined; even its basic source probe contains UNK.
        warnings = [issue for issue in issues if issue.startswith("source_sentencepiece_vocab_mismatch:")]
        issues = [issue for issue in issues if issue not in warnings]
    report = {"direction": direction, "model_id": repo, "revision": revision, "issues": issues,
              "vocabulary_warnings": warnings,
              "status": "quarantined" if issues else "tokenizer_audit_passed", "weights_downloaded": False}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "tokenizer_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if issues or metadata_only:
        print(json.dumps(report), flush=True)
        return report
    source_weight = download(f"https://huggingface.co/{repo}/resolve/{revision}/{weights}", upstream / weights)
    if weights.endswith(".bin"):
        import torch
        state = torch.load(source_weight, map_location="cpu", weights_only=True)
        model = MarianMTModel(config)
        # Older published checkpoints omit lm_head because it is tied to shared.
        # Accept that omission only when the model actually shares the storage.
        if "lm_head.weight" not in state and config.tie_word_embeddings and model.lm_head.weight.data_ptr() == model.model.shared.weight.data_ptr():
            state["lm_head.weight"] = state["model.shared.weight"]
        model.load_state_dict(state, strict=True)
        del state
        model.save_pretrained(destination, safe_serialization=True)
        del model
    else:
        shutil.copyfile(source_weight, destination / weights)
        shutil.copyfile(upstream / "config.json", destination / "config.json")
        shutil.copyfile(upstream / "generation_config.json", destination / "generation_config.json")
    tokenizer.save_pretrained(destination)
    report["weights_downloaded"] = True
    report["status"] = "local_assets_ready"
    (destination / "tokenizer_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
             for p in destination.iterdir() if p.is_file() and p.name != "asset_manifest.json"}
    provenance = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in upstream.iterdir() if p.is_file()}
    manifest = {"model_id": repo, "revision": revision, "files": files,
                "upstream_files": provenance,
                "conversion": "torch.load(weights_only=True), strict state_dict, save_pretrained safetensors" if weights.endswith(".bin") else "none"}
    (destination / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions", nargs="+", choices=list(SOURCES), default=["en-vi", "ko-en"])
    ap.add_argument("--metadata-only", action="store_true")
    args = ap.parse_args()
    reports = [provision(d, args.metadata_only) for d in args.directions]
    registry_path = ROOT / "configs/nmt/models_laptop.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for report in reports:
        spec = registry["models"][report["direction"]]
        spec["revision"] = report["revision"]
        spec["status"] = "quarantined_pending_tokenizer_audit" if report["issues"] else report["status"]
        spec["allow_partial_vocabulary"] = bool(report["vocabulary_warnings"]) and not report["issues"]
    registry["status"] = "local_setup_results_recorded; see per-direction status"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
