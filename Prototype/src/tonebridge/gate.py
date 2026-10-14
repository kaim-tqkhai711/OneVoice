"""Fusion & Safety Gate: five actions, thresholds from the effective pipeline config.

FSM, evaluated in this fixed order (first match wins):
  S0  no speech / empty ASR text                                   -> REPEAT   (VAD or ASR produced nothing usable)
  S1  ASR confidence < abstain_floor                               -> ABSTAIN  (evidence too weak even to ask for a confirmation)
  S2  ASR confidence < asr_confidence_min:
        safety check also failed                                   -> ABSTAIN  (two independent weak signals)
        else                                                       -> REPEAT
  S3  empty translation, or length ratio out of range              -> ABSTAIN
  S4  safety check: critical slot missing/contradictory/uncertain  -> CONFIRM   (TTS blocked)
  S5  safety check: severity group differs                         -> CONFIRM   (TTS blocked)
  S5a independent acoustic evidence weak/missing/uncalibrated      -> REPEAT/CONFIRM (full runtime requires evidence)
  S6  urgency HIGH with voicing quality >= voicing_quality_min     -> SPEAK_CUE
  S7  otherwise (LOW, UNKNOWN, weak voicing)                       -> SPEAK     (a safe translation is spoken even when urgency is UNKNOWN)
Only SPEAK and SPEAK_CUE carry ``speak_text``: every other action blocks TTS.
"""
from __future__ import annotations

from tonebridge.config import Thresholds, QualitySettings
from tonebridge.quality_gate import assess_quality
from tonebridge.contracts import AsrResult, GateAction, GateDecision, SafetyReport, UrgencyResult

GateConfig = Thresholds


def decide(asr: AsrResult, safety: SafetyReport, urgency: UrgencyResult, text: str, cfg: GateConfig,
           quality_cfg: QualitySettings | None = None, *, require_quality: bool = False) -> GateDecision:
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
    if asr.lang == "ko" or any("\uac00" <= c <= "\ud7a3" for c in text):
        # Korean transducers can return unspaced Hangul; word counts are not comparable.
        n_src = sum(c.isalnum() for c in asr.text)
        n_tgt = sum(c.isalnum() for c in text)
    if n_tgt == 0 or (n_src > 0 and (n_tgt / n_src > cfg.max_len_ratio or n_src / n_tgt > cfg.max_len_ratio)):
        return out(GateAction.ABSTAIN, ["translation_empty_or_length_ratio"])
    if not safety.passed:
        return out(GateAction.CONFIRM, safety.reasons or ["safety_failed"])
    if safety.confirm:
        return out(GateAction.CONFIRM, safety.reasons or ["severity_confirm"])
    quality_action, quality_reasons = assess_quality(asr.quality, quality_cfg or QualitySettings(), required=require_quality)
    if quality_action is not None:
        return out(quality_action, quality_reasons)
    if (urgency.label == "HIGH" and urgency.cue_enabled and urgency.classifier_status in ("READY", "LEGACY") and urgency.voicing_quality is not None
            and urgency.voicing_quality >= cfg.voicing_quality_min
            and urgency.score is not None and urgency.score >= cfg.urgency_high):
        return out(GateAction.SPEAK_CUE, ["urgency_high"], text)
    return out(GateAction.SPEAK, ["urgency_" + urgency.label.lower()], text)


class Gate:
    """Callable with the pipeline's gate signature ``gate(asr, safety, urgency, text, pipeline_cfg)``."""

    def __init__(self, cfg: GateConfig | None = None) -> None:
        self.cfg = cfg.model_copy(deep=True) if cfg is not None else GateConfig()

    def __call__(self, asr: AsrResult, safety: SafetyReport, urgency: UrgencyResult, text: str, pipeline_cfg=None) -> GateDecision:
        cfg = pipeline_cfg.thresholds if pipeline_cfg is not None else self.cfg
        quality = pipeline_cfg.quality if pipeline_cfg is not None else QualitySettings()
        required = pipeline_cfg is not None and pipeline_cfg._mode == "full"
        return decide(asr, safety, urgency, text, cfg, quality, require_quality=required)
