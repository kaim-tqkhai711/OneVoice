"""WER / CER with explicit counts so results can be pooled (corpus-level, not mean of per-utterance rates)."""
from __future__ import annotations

from dataclasses import dataclass

import jiwer

from .textnorm import normalize_vi


@dataclass
class ErrorCounts:
    words: int = 0
    word_errors: int = 0
    chars: int = 0
    char_errors: int = 0
    n_utts: int = 0

    @property
    def wer(self) -> float:
        return self.word_errors / self.words if self.words else float("nan")

    @property
    def cer(self) -> float:
        return self.char_errors / self.chars if self.chars else float("nan")

    def add(self, ref: str, hyp: str) -> None:
        r, h = normalize_vi(ref), normalize_vi(hyp)
        if not r:
            return
        w = jiwer.process_words(r, h if h else "<empty>")
        c = jiwer.process_characters(r, h if h else "<empty>")
        self.words += len(r.split())
        self.word_errors += w.substitutions + w.deletions + w.insertions
        self.chars += len(r)
        self.char_errors += c.substitutions + c.deletions + c.insertions
        self.n_utts += 1
