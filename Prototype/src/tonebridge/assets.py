"""Mode-aware, fail-closed asset inventory and streamed integrity checks."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from tonebridge.config import Mode, PipelineConfig, Settings, sha256_file


class AssetEntry(Settings):
    path: str
    reader: str
    required: bool
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def read_manifest(cfg: PipelineConfig) -> dict[Path, dict]:
    """Reject malformed, duplicate and escaping manifest entries."""
    rows = json.loads(cfg.resolve_path(cfg.asset_manifest).read_text(encoding="utf8"))
    if not isinstance(rows, list):
        raise ValueError("runtime_files.json must contain an array")
    result = {}
    for row in rows:
        entry = AssetEntry.model_validate(row)
        path = cfg.resolve_path(entry.path)
        if not path.is_relative_to(cfg.root.resolve()):
            raise ValueError(f"manifest path escapes project root: {entry.path}")
        if path in result:
            raise ValueError(f"duplicate manifest path: {entry.path}")
        result[path] = {**entry.model_dump(), "path": str(path)}
    return result


def selected_assets(cfg: PipelineConfig, mode: Mode) -> list[dict]:
    """Select exactly the assets consumed by configured packs and stages."""
    if mode == "stub":
        return []
    manifest = read_manifest(cfg)
    d = cfg.decode
    paths = [cfg.model_path("vad")]
    paths += [cfg.model_path("asr_vi") / n for n in (d.asr_encoder, d.asr_decoder, d.asr_joiner, "tokens.txt")]
    paths += [cfg.model_path("nmt_vi_en") / n for n in (d.nmt_encoder, d.nmt_decoder, "source.spm", "target.spm", "vocab.json", "config.json")]
    paths += [cfg.model_path("tts_en") / n for n in (d.tts_model, "tokens.txt")]
    phonemes = cfg.model_path("tts_en") / "espeak-ng-data"
    phoneme_paths = [p for p in manifest if p.is_relative_to(phonemes)]
    if not phoneme_paths:
        raise ValueError(f"phonemizer files are not pinned in manifest: {phonemes}")
    paths += phoneme_paths
    if mode == "full":
        paths += [cfg.model_path("swift_f0"), cfg.model_path("band_features")]
        if cfg.branch_b.mlp_path is not None and cfg.branch_b.standardizer is not None:
            paths += [cfg.resolve_path(cfg.branch_b.mlp_path), cfg.resolve_path(cfg.branch_b.standardizer)]
    if cfg.denoise_mode != "off":
        paths.append(cfg.model_path("denoise"))
    missing = [str(p) for p in paths if p not in manifest]
    if missing:
        raise ValueError("configured assets not pinned in manifest:\n" + "\n".join(missing))
    return [manifest[p] for p in sorted(set(paths))]


def verify_files(entries: list[dict]) -> list[str]:
    """Report every missing, size, hash or read error without loading any model."""
    issues = []
    for row in entries:
        path = Path(row["path"])
        try:
            if not path.is_file():
                issues.append(f"missing asset: {path}")
            elif path.stat().st_size != row["bytes"]:
                issues.append(f"size mismatch: {path}")
            elif sha256_file(path) != row["sha256"]:
                issues.append(f"sha256 mismatch: {path}")
        except OSError as exc:
            issues.append(f"cannot read asset ({type(exc).__name__}): {path}")
    return issues
