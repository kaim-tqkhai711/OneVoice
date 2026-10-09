"""Additive NMT/safety contract. Shared contracts.py remains runtime-owner property."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field
from tonebridge.contracts import MtResult, SafetyReport

DIRECTIONS = ("vi-en", "en-vi", "en-ko", "ko-en")


class TranslationEvidence(BaseModel):
    eos_reached: bool
    truncated: bool
    input_truncated: bool = False
    source_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    source_unknown_tokens: int = Field(default=0, ge=0)
    output_unknown_tokens: int = Field(default=0, ge=0)
    constraints_requested: int = Field(default=0, ge=0)
    constraints_satisfied: int = Field(default=0, ge=0)
    unsatisfied_constraints: list[str] = Field(default_factory=list)
    tokenizer_issues: list[str] = Field(default_factory=list)
    model_id: str = ""
    revision: str = ""
    backend: str = ""


class EvidenceMtResult(MtResult):
    evidence: TranslationEvidence


class DetailedSafetyReport(SafetyReport):
    status: Literal["PASS", "CONFIRM", "FAIL"]
    direction: str
    coverage: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @classmethod
    def decision(cls, status: Literal["PASS", "CONFIRM", "FAIL"], direction: str,
                 reasons: list[str], coverage: list[str] | None = None):
        # Existing Gate checks passed first, then confirm: no shared-interface edit needed.
        return cls(status=status, direction=direction, passed=status != "FAIL",
                   confirm=status == "CONFIRM", reasons=reasons, coverage=coverage or [],
                   limitations=["finite clinical grammar; not a diagnosis or ASR correctness check"])
