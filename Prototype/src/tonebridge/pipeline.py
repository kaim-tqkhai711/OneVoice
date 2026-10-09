"""Dual-path orchestrator (laptop reference pipeline).

Flow (shapes in comments):
    raw wav [T] -> FrontEnd (shared) -> [T] -> VAD -> (start_s, end_s) -> segment [T_seg]
        Branch A: Denoiser [T_seg] -> ASR -> NMT -> Safety          (meaning)
        Branch B: BranchB  [T_seg] (NOT denoised)                    (urgency)
    both -> Gate -> TTS (only if approved) -> out wav [T_out] @ tts_sample_rate
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np

from tonebridge.config import PipelineConfig
from tonebridge.contracts import (AsrResult, GateAction, GateDecision, Lang, MtResult, SafetyReport, TurnRecord,
                                  UrgencyResult, Utterance)
from tonebridge.stages.base import Asr, BranchB, Denoiser, FrontEnd, Nmt, SafetyChecker, Tts, Vad
from tonebridge.stages.stubs import v0_gate
from tonebridge.telemetry import TurnClock, peak_rss_mb, rtf


class StageFailure(RuntimeError):
    def __init__(self, stage: str, error: Exception):
        self.stage, self.error_type = stage, type(error).__name__
        super().__init__(stage)


def _call(clk, stage, fn, *args):
    try:
        with clk.stage(stage):
            return fn(*args)
    except Exception as error:
        raise StageFailure(stage, error) from error


@dataclass
class Stages:
    frontend: FrontEnd
    vad: Vad
    denoiser: Denoiser
    asr: Asr
    nmt: Nmt
    safety: SafetyChecker
    branch_b: BranchB
    tts: Tts
    gate: Callable[..., GateDecision] = v0_gate


@dataclass
class TurnResult:
    record: TurnRecord
    out_wav: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))  # [T_out] @ tts_sample_rate


class Pipeline:
    def __init__(self, cfg: PipelineConfig, stages: Stages) -> None:
        self.cfg, self.s = cfg, stages
        cfg.seed_everything()
        manifest = Path(__file__).resolve().parents[2] / "configs/laptop_assets_manifest.json"
        self.manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
        self._workers = None
        self._closed = False

    def close(self):
        """Release the persistent branch workers after the final sequential turn."""
        self._closed = True
        if self._workers is not None:
            self._workers.shutdown(wait=True)
            self._workers = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _branch_a(self, seg: np.ndarray, src: Lang, tgt: Lang, clk: TurnClock) -> tuple[AsrResult, MtResult, SafetyReport]:
        x = _call(clk, "denoise", self.s.denoiser.process, seg)
        if x.shape != seg.shape or not np.isfinite(x).all():
            raise StageFailure("denoise", ValueError("invalid_denoised_audio"))
        asr = _call(clk, "asr", self.s.asr.transcribe, x, src)
        if asr.lang != src:
            raise StageFailure("asr", ValueError("language_mismatch"))
        if not asr.text.strip():
            return asr, MtResult(src_text=asr.text, tgt_text="", src_lang=src, tgt_lang=tgt), SafetyReport(passed=True)
        mt = _call(clk, "nmt", self.s.nmt.translate, asr.text, src, tgt)
        if (mt.src_lang, mt.tgt_lang, mt.src_text) != (src, tgt, asr.text):
            raise StageFailure("nmt", ValueError("translation_contract_mismatch"))
        if mt.truncated or mt.terminated_by_eos is False or mt.constraints_satisfied is False:
            return asr, mt, SafetyReport(passed=False, reasons=["translation_incomplete"])
        evidence = getattr(mt, "evidence", None)
        if evidence is not None and (evidence.truncated or evidence.input_truncated or not evidence.eos_reached
                                    or evidence.source_unknown_tokens or evidence.output_unknown_tokens
                                    or evidence.tokenizer_issues or evidence.unsatisfied_constraints
                                    or evidence.constraints_satisfied != evidence.constraints_requested
                                    or evidence.source_unknown_tokens > evidence.source_tokens
                                    or evidence.output_unknown_tokens > evidence.output_tokens):
            return asr, mt, SafetyReport(passed=False, reasons=["translation_evidence_invalid_or_incomplete"])
        if evidence is None and mt.terminated_by_eos is None:
            return asr, mt, SafetyReport(passed=False, reasons=["translation_evidence_missing"])
        safety = _call(clk, "safety", self.s.safety.check, mt)
        return asr, mt, safety

    def _branch_b(self, seg: np.ndarray, clk: TurnClock) -> UrgencyResult:
        try:
            return _call(clk, "branch_b", self.s.branch_b.analyze, seg)
        except StageFailure:
            return UrgencyResult(label="UNKNOWN", reasons=["branch_b_failed"])

    def run(self, wav: np.ndarray, utt_id: str, session_id: str = "s0", src: Lang | None = None) -> TurnResult:
        """One PTT turn. ``wav``: [T] float32 @ cfg.sample_rate, already endpointed by PTT release."""
        cfg, clk = self.cfg, TurnClock()
        source, tgt = cfg.direction.split("-")
        audio_s = len(wav) / cfg.sample_rate if isinstance(wav, np.ndarray) and wav.ndim else 0.0
        clk.mark("endpoint")  # PTT release: latency clock starts here
        t_proc = clk.stage_ms  # alias for readability
        asr = AsrResult(text="", lang=source, confidence=0.0)
        mt = MtResult(src_text="", tgt_text="", src_lang=source, tgt_lang=tgt)
        safety = None
        status, error_stage, error_type = "ok", None, None
        chunks: list[np.ndarray] = []
        decision = GateDecision(action=GateAction.REPEAT, reasons=["no_speech_or_empty_asr"])
        with clk.stage("total_proc"):
            try:
                if self._closed:
                    raise StageFailure("input", RuntimeError("pipeline_closed"))
                if src is not None and src != source:
                    raise StageFailure("input", ValueError("source_direction_mismatch"))
                if not isinstance(wav, np.ndarray) or wav.ndim != 1 or not np.isfinite(wav).all() or (wav.size and np.max(np.abs(wav)) > 1.001):
                    raise StageFailure("input", ValueError("invalid_audio"))
                if audio_s > cfg.max_utterance_s:
                    decision = GateDecision(action=GateAction.REPEAT, reasons=["utterance_too_long"])
                elif not wav.size or np.max(np.abs(wav)) < 1e-4:
                    pass
                else:
                    x = _call(clk, "frontend", self.s.frontend.process, wav)
                    if x.shape != wav.shape or not np.isfinite(x).all():
                        raise StageFailure("frontend", ValueError("invalid_frontend_audio"))
                    t0, t1 = _call(clk, "vad", self.s.vad.segment, x)
                    if not (0 <= t0 <= t1 <= audio_s + 1e-6):
                        raise StageFailure("vad", ValueError("invalid_timestamps"))
                    seg = x[int(t0 * cfg.sample_rate): int(t1 * cfg.sample_rate)]
                    if seg.size:
                        # Native CPU libraries keep thread-local workspaces.
                        # Creating fresh branch threads per turn grows those caches.
                        if self._workers is None:
                            self._workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tonebridge")
                        fa = self._workers.submit(self._branch_a, seg, source, tgt, clk)
                        fb = self._workers.submit(self._branch_b, seg, clk)
                        wait((fa, fb))  # settle both branches even if A failed
                        asr, mt, safety = fa.result()
                        urgency = fb.result()
                        decision = _call(clk, "gate", self.s.gate, asr, safety, urgency, mt.tgt_text, cfg)
                        # Safety and completion checks cannot be bypassed by a custom gate.
                        if not safety.passed or safety.confirm:
                            if decision.action in (GateAction.SPEAK, GateAction.SPEAK_CUE):
                                decision = GateDecision(action=GateAction.CONFIRM, reasons=safety.reasons, urgency=urgency)
                        clk.mark("text_ready")
                        if decision.action in (GateAction.SPEAK, GateAction.SPEAK_CUE) and decision.speak_text:
                            if decision.speak_text != mt.tgt_text:
                                raise StageFailure("gate", ValueError("unvalidated_speak_text"))
                            try:
                                with clk.stage("tts"):
                                    for c in self.s.tts.stream(decision.speak_text, tgt):
                                        c = np.asarray(c, dtype=np.float32)
                                        if c.ndim != 1 or not np.isfinite(c).all():
                                            raise ValueError("invalid_tts_audio")
                                        if c.size:
                                            if not chunks:
                                                clk.mark("first_audio")
                                            chunks.append(c)
                                    if not chunks:
                                        raise ValueError("empty_tts_audio")
                            except Exception as error:
                                raise StageFailure("tts", error) from error
            except Exception as error:
                chunks.clear()
                status = "error"
                error_stage = error.stage if isinstance(error, StageFailure) else "pipeline"
                error_type = error.error_type if isinstance(error, StageFailure) else type(error).__name__
                decision = GateDecision(action=GateAction.ABSTAIN, reasons=["runtime_failure:" + error_stage])
        if status == "ok" and decision.action not in (GateAction.SPEAK, GateAction.SPEAK_CUE):
            status = "rejected"
        out = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)  # [T_out]
        total_ms = t_proc.pop("total_proc")
        rec = TurnRecord(
            session_id=session_id, utterance_id=utt_id, config_hash=cfg.config_hash(), direction=cfg.direction,
            audio_s=audio_s, stage_ms=dict(t_proc), endpoint_to_first_audio_ms=clk.between_ms("endpoint", "first_audio") if chunks else None,
            endpoint_to_text_ms=clk.between_ms("endpoint", "text_ready"),
            total_ms=total_ms, rtf=rtf(total_ms, audio_s), peak_rss_mb=peak_rss_mb(), gate=decision,
            asr_text=asr.text, nmt_text=mt.tgt_text, status=status, error_stage=error_stage, error_type=error_type,
            runtime_manifest_sha256=self.manifest_hash,
            nmt_evidence=getattr(mt, "evidence", None).model_dump(mode="json") if getattr(mt, "evidence", None) is not None else None,
            safety_status=(getattr(safety, "status", None) or ("FAIL" if not safety.passed else "CONFIRM" if safety.confirm else "PASS")) if safety is not None else None,
            safety_reasons=safety.reasons if safety is not None else [])
        return TurnResult(record=rec, out_wav=out)


def utterance_meta(session_id: str, utt_id: str, n: int, sr: int, end_s: float, src: Lang) -> Utterance:
    """Helper for callers that need the shared metadata message."""
    return Utterance(session_id=session_id, utterance_id=utt_id, sample_rate=sr, n_samples=n,
                     speech_end_s=end_s, src_lang=src)
