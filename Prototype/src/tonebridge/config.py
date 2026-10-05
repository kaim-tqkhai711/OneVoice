"""Single deterministic config file. Everything that changes a result lives here and is hashed."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """A pinned model asset. License is mandatory (Proposal risk table: license conflict)."""

    name: str
    version: str
    license: str
    path: str | None = None
    sha256: str | None = None


class Thresholds(BaseModel):
    asr_confidence_min: float = 0.5  # below -> REPEAT
    voicing_quality_min: float = 0.5  # below -> urgency UNKNOWN
    urgency_high: float = 0.7


class PipelineConfig(BaseModel):
    seed: int = 1234
    sample_rate: int = 16000
    tts_sample_rate: int = 22050
    direction: Literal["vi-ko", "vi-en"] = "vi-en"
    denoise_mode: Literal["off", "on", "oa"] = "off"
    oa_beta: float = Field(0.5, ge=0.0, le=1.0)  # x_oa = beta*x_enh + (1-beta)*x_raw; locked on dev before test
    thresholds: Thresholds = Thresholds()
    max_utterance_s: float = 15.0
    models: dict[str, ModelSpec] = Field(default_factory=dict)

    def config_hash(self) -> str:
        """SHA-256 (first 16 hex) of the canonical JSON. Logged on every turn."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def seed_everything(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
