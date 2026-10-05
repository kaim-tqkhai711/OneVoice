"""Stage interfaces (typing.Protocol). Real modules replace stubs one by one without touching the pipeline.

Shapes: ``wav: [T]`` float32 @ 16 kHz mono unless stated otherwise.
"""
from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np

from tonebridge.contracts import AsrResult, Lang, MtResult, SafetyReport, UrgencyResult


class FrontEnd(Protocol):
    """Shared DSP front-end (WebRTC APM: NS/HPF). Both branches consume its output."""

    def process(self, wav: np.ndarray) -> np.ndarray:  # [T] -> [T]
        ...


class Vad(Protocol):
    def segment(self, wav: np.ndarray) -> tuple[float, float]:  # [T] -> (start_s, end_s)
        ...


class Denoiser(Protocol):
    """Branch A only (ADR-001 modes off/on/oa). Must return the same length and be sample-aligned with input."""

    def process(self, wav: np.ndarray) -> np.ndarray:  # [T] -> [T]
        ...


class Asr(Protocol):
    def transcribe(self, wav: np.ndarray, lang: Lang) -> AsrResult:  # [T] -> text
        ...


class Nmt(Protocol):
    def translate(self, text: str, src: Lang, tgt: Lang) -> MtResult:
        ...


class SafetyChecker(Protocol):
    def check(self, mt: MtResult) -> SafetyReport:
        ...


class BranchB(Protocol):
    """Prosody / urgency. Input is the pitch-preserving segment, never denoised Branch-A audio."""

    def analyze(self, wav: np.ndarray) -> UrgencyResult:  # [T_seg] -> urgency
        ...


class Tts(Protocol):
    def stream(self, text: str, lang: Lang) -> Iterator[np.ndarray]:  # yields [T_chunk] float32 @ tts_sample_rate
        ...
