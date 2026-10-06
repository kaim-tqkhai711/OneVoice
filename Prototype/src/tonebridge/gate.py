"""Fusion & Safety Gate (Proposal 1.2 / 4.2): five actions SPEAK / SPEAK_CUE / REPEAT / CONFIRM / ABSTAIN. Pure logic, thresholds from configs/gate.json.

FSM, evaluated in this fixed order (first match wins):
  S0  no speech / empty ASR text                                   -> REPEAT   (VAD or ASR produced nothing usable)
  S1  ASR confidence < abstain_floor                               -> ABSTAIN  (evidence too weak even to ask for a confirmation)
  S2  ASR confidence < asr_confidence_min:
        safety check also failed                                   -> ABSTAIN  (two independent weak signals)
        else                                                       -> REPEAT
  S3  empty translation, or length ratio out of range              -> ABSTAIN
  S4  safety check: critical slot missing/contradictory/uncertain  -> CONFIRM   (TTS blocked)
  S5  safety check: severity group differs                         -> CONFIRM   (TTS blocked)
  S6  urgency HIGH with voicing quality >= voicing_quality_min     -> SPEAK_CUE
  S7  otherwise (LOW, UNKNOWN, weak voicing)                       -> SPEAK     (a safe translation is spoken even when urgency is UNKNOWN)
Only SPEAK and SPEAK_CUE carry ``speak_text``: every other action blocks TTS.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tonebridge.contracts import AsrResult, GateAction, GateDecision, SafetyReport, UrgencyResult

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class GateConfig:
    asr_confidence_min: float = 0.5
    asr_abstain_floor: float = 0.2
    urgency_high: float = 0.7
    voicing_quality_min: float = 0.7
    min_text_chars: int = 1
    max_len_ratio: float = 6.0

    @classmethod
    def load(cls, path: Path = ROOT / "configs/gate.json") -> "GateConfig":
        d = json.loads(Path(path).read_text(encoding="utf8"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def decide(asr: AsrResult, safety: SafetyReport, urgency: UrgencyResult, text: str, cfg: GateConfig) -> GateDecision:
    def out(a: GateAction, why: list[str], speak: str | None = None) -> GateDecision:
        return GateDecision(action=a, reasons=why, speak_text=speak, urgency=urgency)

    if len(asr.text.strip()) < cfg.min_text_chars:
        return out(GateAction.REPEAT, ["no_speech_or_empty_asr"])
    if asr.confidence < cfg.asr_abstain_floor:
        return out(GateAction.ABSTAIN, ["asr_confidence_below_abstain_floor"])
    if asr.confidence < cfg.asr_confidence_min:
        if not safety.passed:
            return out(GateAction.ABSTAIN, ["asr_confidence_low", "safety_failed"] + safety.reasons)
        return out(GateAction.REPEAT, ["asr_confidence_low"])
    n_src, n_tgt = len(asr.text.split()), len(text.split())
    if n_tgt == 0 or (n_src > 0 and (n_tgt / n_src > cfg.max_len_ratio or n_src / n_tgt > cfg.max_len_ratio)):
        return out(GateAction.ABSTAIN, ["translation_empty_or_length_ratio"])
    if not safety.passed:
        return out(GateAction.CONFIRM, safety.reasons or ["safety_failed"])
    if safety.confirm:
        return out(GateAction.CONFIRM, safety.reasons or ["severity_confirm"])
    if urgency.label == "HIGH" and (urgency.voicing_quality is None or urgency.voicing_quality >= cfg.voicing_quality_min):
        return out(GateAction.SPEAK_CUE, ["urgency_high"], text)
    return out(GateAction.SPEAK, ["urgency_" + urgency.label.lower()], text)


class Gate:
    """Callable with the pipeline's gate signature ``gate(asr, safety, urgency, text, pipeline_cfg)``."""

    def __init__(self, cfg: GateConfig | None = None) -> None:
        self.cfg = cfg or GateConfig.load()

    def __call__(self, asr: AsrResult, safety: SafetyReport, urgency: UrgencyResult, text: str, _pipeline_cfg=None) -> GateDecision:
        return decide(asr, safety, urgency, text, self.cfg)
