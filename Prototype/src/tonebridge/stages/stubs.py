"""Stub implementations for the walking skeleton. Each is replaced by a real module on its planned day."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from tonebridge.config import PipelineConfig
from tonebridge.contracts import (AsrResult, GateAction, GateDecision, Lang, MtResult, SafetyReport,
                                  UrgencyResult)


class PassthroughFrontEnd:
    """Stub for WebRTC APM (real: D2)."""

    def process(self, wav: np.ndarray) -> np.ndarray:  # [T] -> [T]
        return wav


class WholeClipVad:
    """Stub for Silero VAD (real: D2): treats the whole PTT clip as speech."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sr = sample_rate

    def segment(self, wav: np.ndarray) -> tuple[float, float]:  # [T] -> (0, T/sr)
        return 0.0, wav.shape[0] / self.sr


class PassthroughDenoiser:
    """ADR-001 mode OFF. ON/OA (GTCRN + aligned mix) arrive on D2."""

    def process(self, wav: np.ndarray) -> np.ndarray:  # [T] -> [T]
        return wav


class SidecarAsr:
    """Stub ASR: reads ``<wav stem>.txt`` next to the input file, else a fixed phrase."""

    def __init__(self, transcript_path: Path | None = None) -> None:
        self.transcript_path = transcript_path

    def transcribe(self, wav: np.ndarray, lang: Lang) -> AsrResult:  # [T] -> text
        text = "xin chào"
        if self.transcript_path and self.transcript_path.exists():
            text = self.transcript_path.read_text(encoding="utf-8").strip()
        return AsrResult(text=text, lang=lang, confidence=0.9)


class TagNmt:
    """Stub NMT: tags the text with the hop chain. Real Opus-MT / bake-off winner: D2."""

    def __init__(self, direction: str) -> None:
        self.hops = {"vi-ko": ["vi>en", "en>ko"], "vi-en": ["vi>en"]}[direction]
        self.tgt: Lang = direction.split("-")[1]  # type: ignore[assignment]

    def translate(self, text: str, src: Lang, tgt: Lang) -> MtResult:
        return MtResult(src_text=text, tgt_text=f"[STUB {'+'.join(self.hops)}] {text}", src_lang=src, tgt_lang=tgt,
                        hops=self.hops)


class AlwaysPassSafety:
    """Stub safety check (real rules: D3)."""

    def check(self, mt: MtResult) -> SafetyReport:
        return SafetyReport(passed=True)


class UnknownBranchB:
    """Stub Branch B (real SwiftF0 + eGeMAPS + MLP: D3)."""

    def analyze(self, wav: np.ndarray) -> UrgencyResult:  # [T_seg] -> urgency
        return UrgencyResult(label="UNKNOWN")


class ToneTts:
    """Stub TTS: a sine tone whose length scales with the text. Yields two chunks to exercise first-audio timing."""

    def __init__(self, sample_rate: int) -> None:
        self.sr = sample_rate

    def stream(self, text: str, lang: Lang) -> Iterator[np.ndarray]:
        n = max(int(0.05 * len(text) * self.sr), self.sr // 10)  # duration ~ 50 ms per char
        t = np.arange(n, dtype=np.float32) / self.sr  # [n]
        tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)  # [n]
        half = n // 2
        yield tone[:half]  # [n/2]
        yield tone[half:]  # [n - n/2]


def v0_gate(asr: AsrResult, safety: SafetyReport, urgency: UrgencyResult, text: str,
            cfg: PipelineConfig) -> GateDecision:
    """Skeleton gate (real selective gate: D3). Order: weak ASR -> REPEAT, safety fail -> CONFIRM, else SPEAK."""
    if asr.confidence < cfg.thresholds.asr_confidence_min:
        return GateDecision(action=GateAction.REPEAT, reasons=["asr_confidence_low"], urgency=urgency)
    if not safety.passed:
        return GateDecision(action=GateAction.CONFIRM, reasons=safety.reasons or ["safety_failed"], urgency=urgency)
    action = GateAction.SPEAK_CUE if urgency.label == "HIGH" else GateAction.SPEAK
    return GateDecision(action=action, speak_text=text, urgency=urgency)
