"""Classical noise suppression arm (ADR-001, laptop only): numpy-only STFT, noise PSD by minimum-statistics tracking, over-subtracted
Wiener-style gain with a floor and temporal smoothing. No external DSP library (portable to Kotlin). Output is the same length as input."""
from __future__ import annotations

import numpy as np


class ClassicalNs:
    def __init__(self, n_fft: int = 512, hop: int = 128, over: float = 1.5, floor_db: float = -15.0, win_s: float = 1.5, smooth: float = 0.8,
                 sr: int = 16000) -> None:
        self.n, self.h, self.over, self.floor = n_fft, hop, over, 10 ** (floor_db / 20)
        self.win_frames, self.smooth = max(int(win_s * sr / hop), 4), smooth
        self.w = np.hanning(n_fft + 1)[:-1].astype(np.float64)  # periodic Hann, WOLA with hop = n/4 -> sum(w^2) const

    def process(self, wav: np.ndarray) -> np.ndarray:  # [T] -> [T]
        x = wav.astype(np.float64)
        T = len(x)
        pad = self.n
        xp = np.concatenate([np.zeros(pad), x, np.zeros(pad + self.n)])
        nfr = 1 + (len(xp) - self.n) // self.h
        idx = np.arange(self.n)[None, :] + self.h * np.arange(nfr)[:, None]
        X = np.fft.rfft(xp[idx] * self.w, axis=1)  # [F, 257]
        P = np.abs(X) ** 2
        Ps = np.empty_like(P)
        a = 0.85
        Ps[0] = P[0]
        for i in range(1, nfr):  # first-order smoothing of the periodogram
            Ps[i] = a * Ps[i - 1] + (1 - a) * P[i]
        N = np.empty_like(P)
        for i in range(nfr):  # sliding minimum of smoothed power x bias compensation
            lo = max(0, i - self.win_frames + 1)
            N[i] = Ps[lo:i + 1].min(axis=0) * 2.0
        snr_post = P / (N + 1e-12)
        g = np.maximum(1.0 - self.over / np.maximum(snr_post, 1e-6), self.floor)
        for i in range(1, nfr):  # temporal gain smoothing against musical noise
            g[i] = self.smooth * g[i - 1] + (1 - self.smooth) * g[i]
        Y = X * g
        y_frames = np.fft.irfft(Y, n=self.n, axis=1) * self.w
        y = np.zeros(len(xp)); wsum = np.zeros(len(xp))
        for i in range(nfr):
            y[idx[i, 0]: idx[i, 0] + self.n] += y_frames[i]
            wsum[idx[i, 0]: idx[i, 0] + self.n] += self.w ** 2
        y = y / np.maximum(wsum, 1e-8)
        return y[pad: pad + T].astype(np.float32)
