"""Silero VAD (MIT) through sherpa-onnx. Returns one (start_s, end_s) span covering first speech start to last speech end.
No speech found -> (0.0, 0.0): the empty segment makes ASR confidence 0 and the gate asks for a repeat (Proposal 4.4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import sherpa_onnx

ROOT = Path(__file__).resolve().parents[3]


class SileroVad:
    def __init__(self, model: Path = ROOT / "models/silero_vad.onnx", sample_rate: int = 16000, min_silence_s: float = 0.5,
                 min_speech_s: float = 0.15, threshold: float = 0.5, pad_s: float = 0.1) -> None:
        self.sr, self.pad = sample_rate, pad_s
        self._cfg = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(model=str(model), threshold=threshold, min_silence_duration=min_silence_s,
                                                        min_speech_duration=min_speech_s, window_size=512),
            sample_rate=sample_rate, num_threads=1, provider="cpu")
        self._vad = sherpa_onnx.VoiceActivityDetector(self._cfg, buffer_size_in_seconds=60.0)  # max_utterance_s is 15

    def segment(self, wav: np.ndarray) -> tuple[float, float]:  # [T] -> (start_s, end_s)
        vad = self._vad
        vad.reset()
        w = 512
        for i in range(0, len(wav) - w + 1, w):
            vad.accept_waveform(wav[i:i + w].astype(np.float32, copy=False))
        vad.flush()
        starts, ends = [], []
        while not vad.empty():
            seg = vad.front
            starts.append(seg.start / self.sr)
            ends.append((seg.start + len(seg.samples)) / self.sr)
            vad.pop()
        if not starts:
            return 0.0, 0.0
        dur = len(wav) / self.sr
        return max(0.0, starts[0] - self.pad), min(dur, ends[-1] + self.pad)
