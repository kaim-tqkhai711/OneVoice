"""Glossary v0 (vi>en): pure logic over `configs/glossary_vi_en.json`. Finds source terms and checks the output contains an accepted English form.

Matching is longest-term-first on word boundaries, and a matched span is consumed so "miligam" is not also read as "gam".
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _norm(t: str) -> str:
    return unicodedata.normalize("NFC", t).lower()


@dataclass
class TermCheck:
    vi: str
    slot: str
    strict: str
    en_forms: list[str]
    hit: bool


class Glossary:
    def __init__(self, path: Path = ROOT / "configs/glossary_vi_en.json") -> None:
        data = json.loads(Path(path).read_text(encoding="utf8"))
        self.version = data["version"]
        self.terms = sorted(data["terms"], key=lambda t: -len(t["vi"]))
        self._res = [re.compile(r"(?<!\w)" + re.escape(_norm(t["vi"])) + r"(?!\w)") for t in self.terms]

    def find(self, src: str) -> list[dict]:
        """Terms present in the source, longest first, each source span used once."""
        s, found = _norm(src), []
        for t, rx in zip(self.terms, self._res):
            m = rx.search(s)
            if m:
                found.append(t)
                s = s[: m.start()] + " " * (m.end() - m.start()) + s[m.end():]  # consume span
        return found

    def check(self, src: str, tgt: str) -> list[TermCheck]:
        out, tl = [], _norm(tgt) + " "
        for t in self.find(src):
            def accepted(form):
                left = "" if form.startswith("n't") else r"(?<!\w)"
                suffix = r"(?:y|ies|ic)" if form == "allerg" else r"(?:s|es)?" if t["slot"] in ("medication", "dose_unit") else ""
                return re.search(left + re.escape(form) + suffix + r"(?!\w)", tl) is not None
            hit = any(accepted(f) for f in t["en"])
            out.append(TermCheck(t["vi"], t["slot"], t["strict"], t["en"], hit))
        return out
