"""Freeze upstream laptop registries and authenticate active assets before load.

The laptop restoration manifest is distinct from the historical A-C pins. Its
explicit config references are snapshotted once; factories use those snapshots,
and their actual hashes participate in the effective config fingerprint.
"""
import json
from pathlib import Path

from tonebridge.assets import AssetEntry, verify_files
from tonebridge.config import PipelineConfig, sha256_file


def prepare_laptop(cfg: PipelineConfig, threads: int = 2) -> PipelineConfig:
    """Reject unpinned, missing, escaping or modified active pack files."""
    if cfg.audio_config is None:
        raise ValueError("laptop runtime requires an explicit audio_config")
    out = cfg.prepare("stub")
    out._mode = "full"
    out.decode.threads = threads
    for name, value, attr in (("audio_registry", cfg.audio_config, "_laptop_audio_specs"),
                              ("nmt_registry", cfg.nmt_config, "_nmt_registry"),
                              ("clinical_lexicon", "configs/clinical_lexicon.json", "_clinical_lexicon_data")):
        path = out.resolve_path(value)
        data = json.loads(path.read_text(encoding="utf8"))
        if not isinstance(data, dict):
            raise ValueError("invalid laptop registry")
        setattr(out, attr, data)
        out._resources[name] = {"path": str(path), "sha256": sha256_file(path)}
    source, target = out.direction.split("-")
    audio = out._laptop_audio_specs
    asr, tts = audio["asr"][source], audio["tts"][target]
    spec = out._nmt_registry["models"].get(out.direction)
    if spec is None or str(spec.get("status", "")).startswith("quarantined"):
        raise ValueError("nmt_quarantined_or_unregistered:" + out.direction)
    manifest_path = out.resolve_path("configs/laptop_assets_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    pins = {}
    for row in manifest["files"]:
        path = out.resolve_path(row["path"])
        if not path.is_relative_to(out.resolve_path("models")) or path in pins:
            raise ValueError("invalid laptop manifest path")
        entry = AssetEntry.model_validate({**row, "reader": "laptop registry", "required": True})
        pins[path] = {**entry.model_dump(), "path": str(path)}
    paths = [out.resolve_path(p) for p in asr.values()]
    for key, value in tts.items():
        if key in ("kind", "speaker_id"):
            continue
        path = out.resolve_path(value)
        if key == "data_dir":
            children = [p for p in pins if p.is_relative_to(path)]
            if not children:
                raise ValueError("phonemizer data is not pinned")
            paths.extend(children)
        else:
            paths.append(path)
    paths.append(out.resolve_path("models/silero_vad.onnx"))
    nmt_dir = out.resolve_path(spec["assets"])
    nmt_files = [p for p in pins if p.is_relative_to(nmt_dir)]
    if not nmt_files:
        raise ValueError("NMT pack is not pinned in laptop manifest")
    paths.extend(nmt_files)
    if out.denoise_mode != "off":
        paths.append(out.model_path("denoise"))
    if out.urgency_enabled:
        paths.extend((out.model_path("swift_f0"), out.model_path("band_features")))
        if out.branch_b.mlp_path and out.branch_b.standardizer:
            paths.extend((out.resolve_path(out.branch_b.mlp_path), out.resolve_path(out.branch_b.standardizer)))
    missing_pins = [p for p in paths if p not in pins]
    if missing_pins:
        raise ValueError("active laptop asset is not pinned: " + str(missing_pins[0]))
    entries = [pins[p] for p in sorted(set(paths))]
    issues = verify_files(entries)
    if issues:
        raise ValueError("laptop asset preflight failed: " + issues[0])
    out._resources["assets"] = entries
    out._resources["laptop_manifest_sha256"] = sha256_file(manifest_path)
    return out
