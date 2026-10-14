"""EN TTS: Piper en_US-ljspeech-medium (public domain data) through sherpa-onnx. Implements the `Tts` protocol.
License note: phonemization goes through espeak-ng (GPL-3.0), see docs/TTS_GPL_OPTIONS.md.
Each synthesis call is buffered by sherpa-onnx. First-clause mode yields the
first clause before synthesizing the rest; this is clause streaming, not token
streaming. The pipeline must approve the entire translation before calling TTS."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..contracts import Lang

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / "models/tts/vits-piper-en_US-ljspeech-medium"


def split_clauses(text: str, min_words: int = 3) -> list[str]:
    """Split at punctuation (, ; : . ! ?) and before ' and ' / ' but ' / ' then '; clauses shorter than min_words are merged into the next one."""
    import re

    raw = [p.strip() for p in re.split(r"(?<=[,;:.!?])\s+|\s+(?=(?:and|but|then)\s)", text) if p.strip()]
    out: list[str] = []
    buf = ""
    for p in raw:
        buf = (buf + " " + p).strip()
        if len(buf.split()) >= min_words:
            out.append(buf)
            buf = ""
    if buf:
        if out:
            out[-1] = out[-1] + " " + buf
        else:
            out.append(buf)
    return out or [text]


class PiperEn:
    sample_rate = 22050

    def __init__(self, model_dir: Path = DEFAULT_DIR, threads: int = 2, chunk_s: float = 0.25,
                 model_file: str = "en_US-ljspeech-medium.onnx", first_clause: bool = True) -> None:
        import sherpa_onnx
        d = Path(model_dir)
        cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=str(d / model_file), tokens=str(d / "tokens.txt"),
                                                       data_dir=str(d / "espeak-ng-data")),
            num_threads=threads, provider="cpu"))
        self._tts = sherpa_onnx.OfflineTts(cfg)
        self.sample_rate = self._tts.sample_rate
        self._chunk = max(1, int(chunk_s * self.sample_rate))
        self.first_clause = first_clause

    def synth_first_clause(self, text: str) -> tuple[np.ndarray, str]:
        """Synthesise only the first clause (see split_clauses); returns (audio, remaining text). Lowers first-audio latency; the rest is synthesised afterwards."""
        parts = split_clauses(text)
        return self.synth(parts[0]), " ".join(parts[1:])

    def synth(self, text: str) -> np.ndarray:  # -> [T] float32 @ sample_rate
        return np.asarray(self._tts.generate(text, sid=0, speed=1.0).samples, np.float32)

    def stream(self, text: str, lang: Lang) -> Iterator[np.ndarray]:  # yields [T_chunk]
        """Deliver first-clause samples before starting the remaining synthesis."""
        if lang != "en" or not text.strip():
            raise ValueError("Piper needs nonempty English text")
        if self.first_clause:
            first, remaining = self.synth_first_clause(text)
        else:
            first, remaining = self.synth(text), ""
        if not first.size:
            raise ValueError("TTS returned no first-clause audio")
        for i in range(0, len(first), self._chunk):
            yield first[i:i + self._chunk]
        if remaining:
            rest = self.synth(remaining)
            if not rest.size:
                raise ValueError("TTS returned no remaining audio")
            for i in range(0, len(rest), self._chunk):
                yield rest[i:i + self._chunk]
