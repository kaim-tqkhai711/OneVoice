"""Real VI ASR: sherpa-onnx offline Zipformer transducer (INT8 encoder/joiner). Implements the `Asr` protocol."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import sherpa_onnx

from ..contracts import AsrResult, Lang

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / "models/asr/zipformer-vi-int8"


class SherpaZipformerVi:
    def __init__(self, model_dir: Path = DEFAULT_DIR, decoder: str = "decoder-epoch-12-avg-8.onnx", threads: int = 2) -> None:
        d = Path(model_dir)
        self.decoder_file = decoder
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder-epoch-12-avg-8.int8.onnx"), decoder=str(d / decoder),
            joiner=str(d / "joiner-epoch-12-avg-8.int8.onnx"), tokens=str(d / "tokens.txt"),
            num_threads=threads, sample_rate=16000, feature_dim=80, decoding_method="greedy_search", provider="cpu")

    def transcribe(self, wav: np.ndarray, lang: Lang) -> AsrResult:  # [T] float32 @16k -> text
        assert lang == "vi", lang
        s = self._rec.create_stream()
        s.accept_waveform(16000, wav.astype(np.float32, copy=False))
        self._rec.decode_stream(s)
        r = s.result
        lp = list(r.ys_log_probs)
        conf = math.exp(sum(lp) / len(lp)) if lp else 0.0  # geometric-mean token prob, 0..1
        return AsrResult(text=r.text.strip().lower(), lang="vi", confidence=min(1.0, max(0.0, conf)))
