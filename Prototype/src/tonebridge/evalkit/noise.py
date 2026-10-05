"""Reproducible SNR mixing for the ADR-001 grid. Pure numpy."""
from __future__ import annotations

import numpy as np


def rms_power(x: np.ndarray) -> float:  # [T] -> scalar mean square
    return float(np.mean(np.square(x, dtype=np.float64)))


def noise_segment(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:  # [Tn] -> [n]
    """Random n-sample excerpt; looped if the noise is shorter than the clip."""
    if len(noise) < n:
        noise = np.tile(noise, int(np.ceil(n / len(noise))))
    start = int(rng.integers(0, len(noise) - n + 1))
    return noise[start:start + n]


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """noisy = clean + g*noise with 10*log10(P_clean / P_(g*noise)) = snr_db, P over the whole clip.
    If the sum peaks above 0.99 both are scaled together (SNR unchanged)."""
    n = noise_segment(noise, len(clean), rng)
    g = np.sqrt(rms_power(clean) / (rms_power(n) * 10 ** (snr_db / 10) + 1e-20))
    y = clean.astype(np.float64) + g * n
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y *= 0.99 / peak
    return y.astype(np.float32)


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """SNR of a mix given the clean signal it was built from (noisy may be rescaled: fit the gain)."""
    a = float(np.dot(noisy, clean) / (np.dot(clean, clean) + 1e-20))
    resid = noisy - a * clean
    return 10 * np.log10(rms_power(a * clean) / (rms_power(resid) + 1e-20))
