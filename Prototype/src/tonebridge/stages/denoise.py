"""ADR-001 front-end arms for Branch A: OFF (passthrough), GTCRN ON, OA(beta) mix. Branch B never calls these."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import sherpa_onnx

ROOT = Path(__file__).resolve().parents[3]


def oa_mix(noisy: np.ndarray, enhanced: np.ndarray, beta: float) -> np.ndarray:  # [T],[T] -> [T]
    """Observation Adding: y = beta*enhanced + (1-beta)*noisy. Signals must be sample-aligned and equal length."""
    if noisy.shape != enhanced.shape:
        raise ValueError(f"OA mix needs equal length, got {noisy.shape} vs {enhanced.shape}")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    return (beta * enhanced + (1.0 - beta) * noisy).astype(np.float32)


class GtcrnDenoiser:
    """GTCRN (0.5 MB ONNX) through sherpa-onnx; output is trimmed/padded to the input length so downstream mixing stays aligned."""

    def __init__(self, model: Path = ROOT / "models/denoise/gtcrn_simple.onnx", threads: int = 1) -> None:
        cfg = sherpa_onnx.OfflineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=str(model)), num_threads=threads, provider="cpu"))
        self._d = sherpa_onnx.OfflineSpeechDenoiser(cfg)

    def process(self, wav: np.ndarray) -> np.ndarray:  # [T] float32 @16k -> [T]
        out = self._d.run(wav.astype(np.float32, copy=False), 16000)
        y = np.asarray(out.samples, np.float32)
        if len(y) >= len(wav):
            return y[: len(wav)]
        return np.pad(y, (0, len(wav) - len(y)))


class OaDenoiser:
    def __init__(self, inner: GtcrnDenoiser, beta: float) -> None:
        self.inner, self.beta = inner, beta

    def process(self, wav: np.ndarray) -> np.ndarray:  # [T] -> [T]
        return oa_mix(wav, self.inner.process(wav), self.beta)
