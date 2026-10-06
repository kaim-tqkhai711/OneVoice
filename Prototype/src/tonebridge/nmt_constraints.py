"""Glossary -> soft lexical constraints for the NMT decoder (M2 lever). Pure logic over the safety lexicon JSON; the decoder applies them
(nmt_ort.OrtMarianNmt.greedy). Source terms found in the Vietnamese text become constraints: medication names, dose units, dose numbers,
intensity words. Each constraint is a list of accepted target surface forms; the preferred form comes first."""
from __future__ import annotations

from pathlib import Path

from tonebridge.safety import SemanticSafetyChecker, _norm, _word_re

ROOT = Path(__file__).resolve().parents[2]
UNIT_PREF = {"miligam": ["milligrams", "mg"], "microgam": ["micrograms", "mcg"], "mililít": ["milliliters", "ml"], "gam": ["grams"],
             "viên": ["tablets", "tablet", "pills"], "giọt": ["drops"], "ống": ["vials", "ampoules"]}
INT_PREF = {"rất": ["very", "extremely"], "cực kỳ": ["extremely", "very"], "vô cùng": ["extremely", "very"], "quá": ["too", "very"],
            "hơi": ["slightly", "a little"], "khá": ["fairly", "quite"], "dữ dội": ["severe", "intense"]}


class GlossaryConstrainer:
    def __init__(self, groups=("medication", "unit", "number", "intensity"), preferred_only_meds: bool = False,
                 lexicon: Path = ROOT / "configs/safety_lexicon_vi_en.json") -> None:
        self.chk = SemanticSafetyChecker(lexicon)
        self.groups = set(groups)
        self.meds_pref = preferred_only_meds

    def forms(self, text: str) -> list[list[str]]:
        s = _norm(text).rstrip(" ?.!")
        toks = s.split()
        out: list[list[str]] = []
        if "medication" in self.groups:
            used = s
            for m in self.chk.meds:
                mo = _word_re(m["vi"]).search(used)
                if mo:
                    used = used[:mo.start()] + " " * (mo.end() - mo.start()) + used[mo.end():]
                    out.append(m["en"][:1] if self.meds_pref else m["en"])
        if "unit" in self.groups or "number" in self.groups:
            for u in self.chk.units:
                for i, w in enumerate(toks):
                    if w != u["vi"]:
                        continue
                    j = i
                    while j > 0 and (toks[j - 1] in self.chk.vi_num or toks[j - 1].isdigit()):
                        j -= 1
                    num = self.chk._parse_vi_number(toks[j:i]) if j < i else None
                    if "unit" in self.groups:
                        out.append(UNIT_PREF.get(u["vi"], u["en"]))
                    if "number" in self.groups and num is not None:
                        out.append([str(num)])
        if "intensity" in self.groups:
            rest = s
            for m in self.chk.inten:
                mo = _word_re(m["vi"]).search(rest)
                if mo:
                    rest = rest[:mo.start()] + " " * (mo.end() - mo.start()) + rest[mo.end():]
                    out.append(INT_PREF.get(m["vi"], m["en"][:2]))
        return out

    def __call__(self, text: str, piece_ids) -> list[list[list[int]]]:
        res = []
        for alts in self.forms(text):
            seqs = []
            for a in alts:
                ids = piece_ids(a.strip())
                if ids and ids not in seqs:
                    seqs.append(ids)
            if seqs:
                res.append(seqs)
        return res
