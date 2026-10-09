"""Message contracts between pipeline stages (Pydantic). Audio itself travels as numpy arrays.

Audio convention: ``wav: [T] float32 in [-1, 1] @ 16 kHz mono`` unless stated otherwise.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

Lang = Literal["vi", "en", "ko"]
UrgencyLabel = Literal["LOW", "HIGH", "UNKNOWN"]


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


class AsrResult(BaseModel):
    text: str
    lang: Lang
    confidence: float = Field(ge=0.0, le=1.0)  # mean token prob proxy, 0..1


class MtResult(BaseModel):
    src_text: str
    tgt_text: str
    src_lang: Lang
    tgt_lang: Lang
    hops: list[str] = Field(default_factory=list)  # e.g. ["vi>en", "en>ko"]
    terminated_by_eos: bool | None = None  # None = legacy adapter did not report evidence
    truncated: bool = False
    constraints_satisfied: bool | None = None


class SlotCheck(BaseModel):
    slot: Literal["negation", "medication", "dose_unit", "dose_number", "allergy", "symptom", "intensity"]
    src_value: str | None
    tgt_value: str | None
    preserved: bool


class SafetyReport(BaseModel):
    passed: bool  # False = a critical slot (negation / medication / dose / allergy) is missing, contradictory or uncertain
    confirm: bool = False  # True = only the severity ("mức độ") group differs: gate asks for CONFIRM, nothing critical is missing
    checks: list[SlotCheck] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class UrgencyResult(BaseModel):
    label: UrgencyLabel
    score: float | None = None
    voicing_quality: float | None = None  # SwiftF0 pitch-quality evidence, 0..1
    reasons: list[str] = Field(default_factory=list)  # why UNKNOWN (silence, too_short, voicing_quality_low, mlp_not_trained, ...)


class GateDecision(BaseModel):
    action: GateAction
    reasons: list[str] = Field(default_factory=list)
    speak_text: str | None = None  # text approved for TTS, None when blocked
    urgency: UrgencyResult | None = None


class TurnRecord(BaseModel):
    """One JSONL row per turn. Numbers are measurements; ``None`` means not measured."""

    session_id: str
    utterance_id: str
    config_hash: str
    direction: str
    audio_s: float
    stage_ms: dict[str, float]  # keys: denoise, vad, asr, nmt, safety, branch_b, gate, tts
    endpoint_to_first_audio_ms: float | None  # PTT release -> first TTS sample
    endpoint_to_text_ms: float | None = None  # PTT release -> EN text available (after safety check + gate)
    total_ms: float
    rtf: float  # total processing time / audio duration
    peak_rss_mb: float
    gate: GateDecision
    asr_text: str
    nmt_text: str
    status: Literal["ok", "rejected", "error"] = "ok"
    error_stage: str | None = None
    error_type: str | None = None
    runtime_manifest_sha256: str | None = None
    nmt_evidence: dict | None = None
    safety_status: str | None = None
    safety_reasons: list[str] = Field(default_factory=list)
