"""Message contracts between pipeline stages (Pydantic). Audio itself travels as numpy arrays.

Audio convention: ``wav: [T] float32 in [-1, 1] @ 16 kHz mono`` unless stated otherwise.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

Lang = Literal["vi", "en", "ko"]
UrgencyLabel = Literal["LOW", "HIGH", "UNKNOWN", "DISABLED", "UNCONFIGURED"]


class GateAction(str, Enum):
    """The five Fusion & Safety Gate outputs (Proposal section 1.2)."""

    SPEAK = "SPEAK"
    SPEAK_CUE = "SPEAK_CUE"  # speak + urgency cue
    REPEAT = "REPEAT"
    CONFIRM = "CONFIRM"
    ABSTAIN = "ABSTAIN"


class AudioBuffer(BaseModel):
    """Mono audio plus its sample rate. ``wav: [T]`` float32."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    wav: np.ndarray
    sample_rate: int = 16000

    @property
    def duration_s(self) -> float:
        return float(self.wav.shape[0]) / self.sample_rate  # [T] -> scalar


class Utterance(BaseModel):
    """One PTT utterance after endpointing. Timestamps are shared by both branches."""

    session_id: str
    utterance_id: str
    sample_rate: int = 16000
    n_samples: int
    speech_start_s: float = 0.0
    speech_end_s: float
    src_lang: Lang


class AcousticEvidence(BaseModel):
    """Independent acoustic measurements, never a probability of correct ASR."""
    model_config = ConfigDict(allow_inf_nan=False, validate_assignment=True)
    duration_s: float = Field(gt=0)
    rms_dbfs: float = Field(ge=-120, le=0)
    snr_proxy_db: float = Field(ge=0)
    clipping_fraction: float = Field(ge=0, le=1)
    vad_ratio: float | None = Field(None, ge=0, le=1)
    voicing_quality: float | None = Field(None, ge=0, le=1)
    words_per_second: float = Field(ge=0)
    repeated_bigram_fraction: float = Field(ge=0, le=1)


class AsrResult(BaseModel):
    text: str
    lang: Lang
    confidence: float = Field(ge=0.0, le=1.0)  # mean token prob proxy, 0..1
    quality: AcousticEvidence | None = None


class MtResult(BaseModel):
    src_text: str
    tgt_text: str
    src_lang: Lang
    tgt_lang: Lang
    hops: list[str] = Field(default_factory=list)  # e.g. ["vi>en", "en>ko"]
    truncated: bool = False  # decoder hit its token cap without EOS
    terminated_by_eos: bool | None = None
    constraints_satisfied: bool | None = None


class SlotCheck(BaseModel):
    slot: Literal["negation", "medication", "dose_unit", "dose_number", "allergy", "symptom", "intensity", "clinical_relation"]
    src_value: str | None
    tgt_value: str | None
    preserved: bool


class SafetyReport(BaseModel):
    passed: bool  # False = a critical slot (negation / medication / dose / allergy) is missing, contradictory or uncertain
    confirm: bool = False  # True = only the severity ("mức độ") group differs: gate asks for CONFIRM, nothing critical is missing
    checks: list[SlotCheck] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    status: Literal["PASS", "CONFIRM", "FAIL"] | None = None

    @model_validator(mode="after")
    def consistent_status(self):
        """Keep the upstream status API consistent with the gate's boolean contract."""
        self.status = "FAIL" if not self.passed else "CONFIRM" if self.confirm else "PASS"
        return self


class UrgencyResult(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    label: UrgencyLabel
    score: float | None = Field(None, ge=0, le=1)
    voicing_quality: float | None = Field(None, ge=0, le=1)  # SwiftF0 pitch-quality evidence, 0..1
    classifier_status: Literal["READY", "UNCONFIGURED", "DISABLED", "ERROR", "LEGACY"] = "LEGACY"
    cue_enabled: bool = True  # Legacy callers retained; real Branch B always sets this explicitly.
    reasons: list[str] = Field(default_factory=list)  # why UNKNOWN (silence, too_short, voicing_quality_low, mlp_not_trained, ...)


class GateDecision(BaseModel):
    action: GateAction
    reasons: list[str] = Field(default_factory=list)
    speak_text: str | None = None  # text approved for TTS, None when blocked
    urgency: UrgencyResult | None = None


class TurnRecord(BaseModel):
    """Internal turn state. Serialization is metadata-only unless explicitly opted in.

    Text remains in memory for safety/TTS. Evaluation serializers must pass
    ``context={"eval_mode": True}``; ordinary dumps cannot leak that state.
    """

    session_id: str
    utterance_id: str
    config_hash: str
    direction: str
    audio_s: float
    stage_ms: dict[str, float]  # keys: denoise, vad, asr, nmt, safety, branch_b, gate, tts
    endpoint_to_first_audio_ms: float | None  # PTT release -> first TTS sample
    endpoint_to_first_delivery_ms: float | None = None  # consumer callback returned; not measured audible sound
    endpoint_to_text_ms: float | None = None  # PTT release -> EN text available (after safety check + gate)
    total_ms: float
    rtf: float  # total processing time / audio duration
    peak_rss_mb: float

    @property
    def peak_rss_mib(self) -> float:
        """Unambiguous binary unit; peak_rss_mb is the legacy API alias (MiB)."""
        return self.peak_rss_mb
    gate: GateDecision
    asr_text: str
    nmt_text: str
    mode: str = "custom"
    capabilities: dict[str, str] = Field(default_factory=dict)
    status: Literal["ok", "no_speech", "invalid_audio", "error", "timeout"] = "ok"
    error_stage: str | None = None
    error_type: str | None = None
    runtime_manifest_sha256: str | None = None
    nmt_evidence: dict | None = None
    safety_status: str | None = None
    safety_reasons: list[str] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def serialize_with_privacy(self, handler, info):
        row = handler(self)
        row["peak_rss_mib"] = self.peak_rss_mib
        row["rss_unit"] = "MiB"
        if info.context and info.context.get("eval_mode") is True:
            return row
        from tonebridge.telemetry import metadata_only
        return metadata_only(row)
