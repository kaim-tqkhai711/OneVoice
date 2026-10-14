"""Validated settings and effective snapshots used by factories and fingerprints.

pipeline.json is the single source of runtime settings. Resource contents are
resolved before model loading; factories do not consult legacy child configs.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

ROOT = Path(__file__).resolve().parents[2]
Mode = Literal["stub", "branch-a-bench", "full"]


def sha256_file(path: Path) -> str:
    """Hash model files without loading them into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class ModelSpec(Settings):
    """Pack location/provenance; exact file hashes are pinned in the manifest."""
    name: str
    version: str
    license: str
    path: str | None = None
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class Thresholds(Settings):
    asr_confidence_min: float = Field(0.5, ge=0, le=1)
    asr_abstain_floor: float = Field(0.2, ge=0, le=1)
    voicing_quality_min: float = Field(0.7, ge=0, le=1)
    urgency_high: float = Field(0.7, ge=0, le=1)
    min_text_chars: int = Field(1, ge=1)
    max_len_ratio: float = Field(6.0, ge=1)

    @model_validator(mode="after")
    def ordered_confidence(self):
        if self.asr_abstain_floor > self.asr_confidence_min:
            raise ValueError("asr_abstain_floor must not exceed asr_confidence_min")
        return self

    @classmethod
    def load(cls, path: Path = ROOT / "configs/pipeline.json") -> "Thresholds":
        """Compatibility entry point reading the central config only."""
        return PipelineConfig.load(path).thresholds


class BranchBSettings(Settings):
    mlp_path: str | None = Field(None, min_length=1)
    standardizer: str | None = Field(None, min_length=1)
    min_duration_s: float = Field(0.3, gt=0)
    min_voiced_frames: int = Field(20, ge=1)
    min_voiced_frac: float = Field(0.15, ge=0, le=1)
    min_voicing_quality: float = Field(0.7, ge=0, le=1)
    voiced_conf: float = Field(0.5, ge=0, le=1)
    silence_peak: float = Field(0.001, ge=0, le=1)
    high_threshold: float = Field(0.7, ge=0, le=1)
    tuned: bool = False

    @classmethod
    def load(cls, path: Path = ROOT / "configs/pipeline.json") -> "BranchBSettings":
        return PipelineConfig.load(path).branch_b


class QualitySettings(Settings):
    """Conservative rejection thresholds; proxy SNR is not measured physical SNR.

    Accepting audio requires an explicitly identified, frozen calibration.
    Until that evaluation exists, even strong acoustic evidence asks CONFIRM.
    """
    min_vad_ratio: float = Field(0.15, ge=0, le=1)
    min_rms_dbfs: float = Field(-45.0, ge=-120, le=0)
    min_snr_proxy_db: float = Field(6.0, ge=0, le=100)
    max_clipping_fraction: float = Field(0.01, ge=0, le=1)
    min_voicing_quality: float = Field(0.35, ge=0, le=1)
    max_words_per_second: float = Field(8.0, gt=0)
    max_repeated_bigram_fraction: float = Field(0.6, ge=0, le=1)
    calibrated: bool = False
    calibration_id: str | None = None

    @model_validator(mode="after")
    def require_calibration(self):
        if self.calibrated and not (self.calibration_id and self.calibration_id.strip()):
            raise ValueError("calibrated quality policy requires calibration_id")
        return self


class DecodeSettings(Settings):
    threads: int = Field(2, ge=1, le=32)
    branch_b_threads: int = Field(1, ge=1, le=32)
    asr_method: Literal["greedy_search", "modified_beam_search"] = "greedy_search"
    asr_active_paths: int = Field(4, ge=1)
    asr_encoder: str = "encoder-epoch-12-avg-8.int8.onnx"
    asr_decoder: str = "decoder-epoch-12-avg-8.int8.onnx"
    asr_joiner: str = "joiner-epoch-12-avg-8.int8.onnx"
    nmt_encoder: str = "encoder_model_quantized.onnx"
    nmt_decoder: str = "decoder_model_merged_quantized.onnx"
    max_new_tokens: int = Field(48, ge=1, le=512)
    num_beams: int = Field(1, ge=1, le=16)
    length_penalty: float = Field(1.0, ge=0)
    glossary_enabled: bool = True
    glossary_groups: list[Literal["medication", "unit", "number", "intensity"]] = Field(
        default_factory=lambda: ["medication", "unit", "number", "intensity"])
    glossary_bonus: float = Field(5.0, ge=0)
    eos_penalty: float | None = Field(None, ge=0)
    preferred_only_meds: bool = False
    tts_model: str = "en_US-ljspeech-medium.onnx"
    tts_chunk_s: float = Field(0.25, gt=0)
    tts_first_clause: bool = True

    @model_validator(mode="after")
    def supported_decode(self):
        if self.glossary_enabled and self.num_beams != 1:
            raise ValueError("glossary constraints require greedy decoding (num_beams=1)")
        for name in ("asr_encoder", "asr_decoder", "asr_joiner", "nmt_encoder", "nmt_decoder", "tts_model"):
            value = getattr(self, name)
            if not value or Path(value).name != value or "\\" in value or "/" in value:
                raise ValueError(f"{name} must be a file name, not a path")
        return self


class VadSettings(Settings):
    min_silence_s: float = Field(0.5, gt=0)
    min_speech_s: float = Field(0.15, gt=0)
    threshold: float = Field(0.5, ge=0, le=1)
    pad_s: float = Field(0.1, ge=0)


def default_models() -> dict[str, ModelSpec]:
    """Active pack locations shared by config, preflight and factory."""
    rows = {
        "asr_vi": ("zipformer-vi-int8", "2025-04-20", "Apache-2.0", "models/asr/zipformer-vi-int8"),
        "nmt_vi_en": ("opus-mt-vi-en INT8 arm64", "runtime_files.json", "Apache-2.0", "models/nmt/vi-en-int8-arm64"),
        "tts_en": ("piper en_US-ljspeech-medium", "runtime_files.json", "LJ Speech PD; phonemizer license unresolved", "models/tts/vits-piper-en_US-ljspeech-medium"),
        "vad": ("silero-vad", "runtime_files.json", "MIT", "models/silero_vad.onnx"),
        "swift_f0": ("swift-f0", "0.3.0", "MIT", "models/branch_b/swiftf0_model.onnx"),
        "band_features": ("tonebridge band features", "v1", "project", "models/branch_b/band_feats.onnx"),
        "denoise": ("GTCRN", "runtime_files.json", "MIT; weights terms unresolved", "models/denoise/gtcrn_simple.onnx"),
    }
    return {k: ModelSpec(name=n, version=v, license=lic, path=p) for k, (n, v, lic, p) in rows.items()}


class PipelineConfig(Settings):
    seed: int = 1234
    sample_rate: Literal[16000] = 16000
    tts_sample_rate: int = Field(22050, gt=0)
    direction: Literal["vi-ko", "vi-en", "en-vi", "en-ko", "ko-en"] = "vi-en"
    denoise_mode: Literal["off", "on", "oa"] = "off"
    oa_beta: float = Field(0.5, ge=0, le=1)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    branch_b: BranchBSettings = Field(default_factory=BranchBSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    decode: DecodeSettings = Field(default_factory=DecodeSettings)
    vad: VadSettings = Field(default_factory=VadSettings)
    max_utterance_s: float = Field(15.0, gt=0, le=15)
    silence_peak: float = Field(1e-6, ge=0, le=1)
    branch_timeout_s: float = Field(10.0, gt=0, le=600)
    turn_timeout_s: float = Field(30.0, gt=0, le=600)
    model_load_timeout_s: float = Field(60.0, gt=0, le=600)
    asset_manifest: str = "configs/runtime_files.json"
    safety_lexicon: str = "configs/safety_lexicon_vi_en.json"
    audio_config: str | None = None
    text_factory: str | None = None
    nmt_config: str = "configs/nmt/models_laptop.json"
    format_asr_source: bool = True
    urgency_enabled: bool = True
    log_content: Literal[False] = False  # content requires an explicit CLI --eval-mode
    models: dict[str, ModelSpec] = Field(default_factory=default_models)
    _root: Path = PrivateAttr(default=ROOT)
    _mode: str | None = PrivateAttr(default=None)
    _lexicon_data: dict | None = PrivateAttr(default=None)
    _resources: dict = PrivateAttr(default_factory=dict)
    _laptop_audio_specs: dict | None = PrivateAttr(default=None)
    _nmt_registry: dict | None = PrivateAttr(default=None)
    _clinical_lexicon_data: dict | None = PrivateAttr(default=None)

    @property
    def root(self) -> Path:
        return self._root

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve assets relative to the project root rather than process cwd."""
        path = Path(value)
        return (path if path.is_absolute() else self.root / path).resolve()

    def model_path(self, key: str) -> Path:
        spec = self.models.get(key)
        if spec is None or not spec.path:
            raise ValueError(f"models.{key}.path is required")
        return self.resolve_path(spec.path)

    def config_hash(self) -> str:
        """Fingerprint settings, resolved locations, pinned resource bytes and mode."""
        data = self.model_dump(mode="json")
        data["model_paths"] = {k: str(self.resolve_path(v.path)) for k, v in sorted(self.models.items()) if v.path}
        data["effective_resources"] = self._resources
        data["mode"] = self._mode
        blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf8")).hexdigest()[:16]

    def prepare(self, mode: Mode, *, verify_assets: bool = True) -> "PipelineConfig":
        """Resolve a fresh snapshot; stages must not independently reload settings."""
        from tonebridge.assets import selected_assets, verify_files

        if mode not in ("stub", "branch-a-bench", "full"):
            raise ValueError(f"unsupported mode: {mode}")
        if mode != "stub" and self.direction != "vi-en":
            raise ValueError("real modes support vi-en only (ADR-002)")
        out = PipelineConfig.model_validate(self.model_dump())
        out._root, out._mode = self.root, mode
        lex_path = out.resolve_path(out.safety_lexicon)
        lex_bytes = lex_path.read_bytes()
        out._lexicon_data = json.loads(lex_bytes.decode("utf8"))
        if not isinstance(out._lexicon_data, dict) or not isinstance(out._lexicon_data.get("version"), str):
            raise ValueError("safety lexicon must be an object with a version string")
        from tonebridge.safety import SemanticSafetyChecker
        from tonebridge.clinical_relations import PARSER_VERSION
        from tonebridge.quality_gate import QUALITY_VERSION
        from tonebridge.urgency_model import SCHEMA_VERSION
        from tonebridge.branch_b import FEATURE_NAMES
        try:
            SemanticSafetyChecker(out._lexicon_data).check_texts("", "")
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid safety lexicon schema") from exc
        out._resources = {"clinical_parser": PARSER_VERSION, "acoustic_policy": QUALITY_VERSION,
                          "urgency_schema": SCHEMA_VERSION,
                          "urgency_feature_order": hashlib.sha256(json.dumps(FEATURE_NAMES).encode()).hexdigest(),
                          "lexicon": {"path": str(lex_path), "version": out._lexicon_data.get("version"),
                                      "sha256": hashlib.sha256(lex_bytes).hexdigest()}}
        if mode != "stub":
            entries = selected_assets(out, mode)
            if verify_assets:
                issues = verify_files(entries)
                if issues:
                    raise ValueError("asset preflight failed:\n" + "\n".join(issues))
            out._resources["assets"] = entries
            for key, spec in out.models.items():
                if spec.sha256:
                    path = out.model_path(key)
                    if not any(Path(e["path"]).is_relative_to(path) for e in entries):
                        continue  # An inactive pack must not become a mode prerequisite.
                    matches = [e for e in entries if Path(e["path"]) == path]
                    if not matches or matches[0]["sha256"] != spec.sha256:
                        raise ValueError(f"models.{key}.sha256 does not match the manifest")
        return out

    def seed_everything(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        """Missing, malformed or unknown settings fail immediately."""
        path = Path(path).resolve(strict=True)
        out = cls.model_validate_json(path.read_text(encoding="utf8"))
        out._root = path.parent.parent if path.parent.name == "configs" else path.parent
        return out
