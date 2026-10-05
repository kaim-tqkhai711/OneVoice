"""Dual-path orchestrator (laptop reference pipeline).

Flow (shapes in comments):
    raw wav [T] -> FrontEnd (shared) -> [T] -> VAD -> (start_s, end_s) -> segment [T_seg]
        Branch A: Denoiser [T_seg] -> ASR -> NMT -> Safety          (meaning)
        Branch B: BranchB  [T_seg] (NOT denoised)                    (urgency)
    both -> Gate -> TTS (only if approved) -> out wav [T_out] @ tts_sample_rate
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from tonebridge.config import PipelineConfig
from tonebridge.contracts import (AsrResult, GateAction, GateDecision, Lang, MtResult, SafetyReport, TurnRecord,
                                  UrgencyResult, Utterance)
from tonebridge.stages.base import Asr, BranchB, Denoiser, FrontEnd, Nmt, SafetyChecker, Tts, Vad
from tonebridge.stages.stubs import v0_gate
from tonebridge.telemetry import TurnClock, peak_rss_mb, rtf


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

    def _branch_a(self, seg: np.ndarray, src: Lang, tgt: Lang, clk: TurnClock) -> tuple[AsrResult, MtResult, SafetyReport]:
        with clk.stage("denoise"):
            x = self.s.denoiser.process(seg)  # [T_seg] -> [T_seg]
        with clk.stage("asr"):
            asr = self.s.asr.transcribe(x, src)
        with clk.stage("nmt"):
            mt = self.s.nmt.translate(asr.text, src, tgt)
        with clk.stage("safety"):
            safety = self.s.safety.check(mt)
        return asr, mt, safety

    def _branch_b(self, seg: np.ndarray, clk: TurnClock) -> UrgencyResult:
        with clk.stage("branch_b"):
            return self.s.branch_b.analyze(seg)  # [T_seg] raw-ish, pitch-preserving

    def run(self, wav: np.ndarray, utt_id: str, session_id: str = "s0", src: Lang = "vi") -> TurnResult:
        """One PTT turn. ``wav``: [T] float32 @ cfg.sample_rate, already endpointed by PTT release."""
        cfg, clk = self.cfg, TurnClock()
        tgt: Lang = cfg.direction.split("-")[1]  # type: ignore[assignment]
        audio_s = wav.shape[0] / cfg.sample_rate
        clk.mark("endpoint")  # PTT release: latency clock starts here
        t_proc = clk.stage_ms  # alias for readability
        with clk.stage("total_proc"):
            x = self.s.frontend.process(wav)  # [T] -> [T]
            with clk.stage("vad"):
                t0, t1 = self.s.vad.segment(x)
            seg = x[int(t0 * cfg.sample_rate): int(t1 * cfg.sample_rate)]  # [T] -> [T_seg]
            with ThreadPoolExecutor(max_workers=2) as pool:  # fan-out: same segment to both branches
                fa = pool.submit(self._branch_a, seg, src, tgt, clk)
                fb = pool.submit(self._branch_b, seg, clk)
                asr, mt, safety = fa.result()
                urgency = fb.result()
            with clk.stage("gate"):
                decision = self.s.gate(asr, safety, urgency, mt.tgt_text, cfg)
            chunks: list[np.ndarray] = []
            if decision.action in (GateAction.SPEAK, GateAction.SPEAK_CUE) and decision.speak_text:
                with clk.stage("tts"):
                    for c in self.s.tts.stream(decision.speak_text, tgt):  # each [T_chunk]
                        if not chunks:
                            clk.mark("first_audio")
                        chunks.append(c)
        out = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)  # [T_out]
        total_ms = t_proc.pop("total_proc")
        rec = TurnRecord(
            session_id=session_id, utterance_id=utt_id, config_hash=cfg.config_hash(), direction=cfg.direction,
            audio_s=audio_s, stage_ms=dict(t_proc), endpoint_to_first_audio_ms=clk.between_ms("endpoint", "first_audio"),
            total_ms=total_ms, rtf=rtf(total_ms, audio_s), peak_rss_mb=peak_rss_mb(), gate=decision,
            asr_text=asr.text, nmt_text=mt.tgt_text)
        return TurnResult(record=rec, out_wav=out)


def utterance_meta(session_id: str, utt_id: str, n: int, sr: int, end_s: float, src: Lang) -> Utterance:
    """Helper for callers that need the shared metadata message."""
    return Utterance(session_id=session_id, utterance_id=utt_id, sample_rate=sr, n_samples=n,
                     speech_end_s=end_s, src_lang=src)
