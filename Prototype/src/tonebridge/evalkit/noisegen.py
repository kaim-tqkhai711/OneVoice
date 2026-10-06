"""Seeded synthetic noise for ADR-001: babble (sum of other speakers) and non-stationary alarm/beep. Pure numpy; no external data except speech
clips handed in by the caller (FLEURS utterances disjoint from dev/test)."""
from __future__ import annotations

import numpy as np

SR = 16000


def make_babble(speech_pool: list[np.ndarray], seconds: float, n_talkers: int, seed: int) -> np.ndarray:  # -> [seconds*SR]
    """Each talker = random utterances concatenated to the target length; talkers summed after RMS-normalising each to equal power."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    out = np.zeros(n)
    for _ in range(n_talkers):
        parts, tot = [], 0
        while tot < n:
            x = speech_pool[int(rng.integers(len(speech_pool)))]
            parts.append(x); tot += len(x)
        t = np.concatenate(parts)[:n].astype(np.float64)
        out += t / (np.sqrt(np.mean(t ** 2)) + 1e-9)
    return (out / (np.max(np.abs(out)) + 1e-9) * 0.5).astype(np.float32)


def make_alarm(seconds: float, seed: int) -> np.ndarray:  # -> [seconds*SR]
    """Monitor-alarm-like: bursts of 2-3 beeps (tone 800-2500 Hz + 2nd harmonic, 100-200 ms, raised-cosine edges) repeated at random 1-4 s
    intervals, random per-burst level (-6..0 dB), plus low-level pink-ish hiss so the floor is not digital silence."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    out = np.zeros(n)
    t = int(rng.uniform(0.2, 1.0) * SR)
    while t < n:
        f = rng.uniform(800, 2500)
        lvl = 10 ** (rng.uniform(-6, 0) / 20)
        for _ in range(int(rng.integers(2, 4))):
            d = int(rng.uniform(0.10, 0.20) * SR)
            k = np.arange(d) / SR
            env = np.hanning(d) ** 0.3  # soft edges, flat top
            b = lvl * env * (np.sin(2 * np.pi * f * k) + 0.3 * np.sin(2 * np.pi * 2 * f * k))
            e = min(t + d, n)
            out[t:e] += b[: e - t]
            t += d + int(rng.uniform(0.05, 0.12) * SR)
        t += int(rng.uniform(1.0, 4.0) * SR)
    hiss = np.cumsum(rng.standard_normal(n)); hiss -= np.convolve(hiss, np.ones(800) / 800, "same")
    out += 0.01 * hiss / (np.std(hiss) + 1e-9)
    return (out / (np.max(np.abs(out)) + 1e-9) * 0.5).astype(np.float32)
