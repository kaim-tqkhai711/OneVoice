"""Semantic Safety Check v1 (VI->EN). Pure logic over ``configs/safety_lexicon_vi_en.json``: no model, no framework (portable by design).

For every critical slot found in the Vietnamese source it demands an accepted English rendering in the translation:
  negation, medication, allergy, dose (number AND unit)  -> missing / contradictory / uncertain = CRITICAL (``passed=False``, TTS blocked)
  intensity ("rất/quá/cực kỳ/..." <-> very/extremely/severe/...)                      -> ``confirm=True`` (gate action CONFIRM)
Sentence-final "không"/"chưa" (and "có ... không") is a question particle, not a negation. ASR text has no punctuation, so this is positional.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from tonebridge.contracts import MtResult, SafetyReport, SlotCheck

ROOT = Path(__file__).resolve().parents[2]


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t).lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", t).strip()


def _has_form(text: str, form: str) -> bool:
    """Left word boundary always (except n't); right boundary only for short forms/digits so stems like 'allerg' match 'allergic'."""
    f = re.escape(form)
    left = "" if form.startswith("n't") else r"(?<![a-z0-9])"
    right = r"(?![a-z0-9])" if (len(form.strip()) <= 3 or form[0].isdigit()) else ""
    return re.search(left + f + right, text) is not None


def _any(text: str, forms: list[str]) -> str | None:
    for f in forms:
        if _has_form(text, f):
            return f
    return None


def _any_word(text: str, forms: list[str]) -> str | None:
    """Whole-word match (optional plural s/es) for drug and allergen names: 'advilized' or 'paracetamino' must NOT count as 'advil' / 'paracetamol'.
    Multi-word forms and the stems used for classes ('antibiotic') still work because only the right edge is anchored after an optional plural."""
    for f in forms:
        if re.search(r"(?<![a-z0-9])" + re.escape(f) + r"(?:s|es)?(?![a-z0-9])", text):
            return f
    return None


def _word_re(term: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")


class SemanticSafetyChecker:
    def __init__(self, lexicon: Path | dict = ROOT / "configs/safety_lexicon_vi_en.json", enable: dict[str, bool] | None = None) -> None:
        lex = lexicon if isinstance(lexicon, dict) else json.loads(Path(lexicon).read_text(encoding="utf8"))
        self.lex = lex
        self.enable = {"negation": True, "extra_negation": True, "medication": True, "allergy": True, "dose": True, "intensity": True, **(enable or {})}
        self.meds = sorted(lex["medications"], key=lambda m: -len(m["vi"]))
        self.units = lex["dose_units"]
        self.inten = sorted(lex["intensity"], key=lambda m: -len(m["vi"]))
        self.vi_num = lex["vi_numbers"]
        self.en_num = lex["en_numbers"]
        from tonebridge.clinical_safety import ClinicalSafetyChecker
        self.clinical = ClinicalSafetyChecker()

    # ---- number parsing -------------------------------------------------------------------------------------------------------------
    def _parse_vi_number(self, toks: list[str]) -> int | None:
        """Vietnamese number words (0..9999) or a digit string -> int; None if the sequence is not a number."""
        if len(toks) == 1 and toks[0].isdigit():
            return int(toks[0])
        total = hund = rest = pend = 0
        seen = False
        for t in toks:
            v = self.vi_num.get(t)
            if v is None:
                return None
            seen = True
            if v == "x10":
                rest += (pend or 1) * 10
                pend = 0
            elif v == "x100":
                hund = (pend or 1) * 100
                pend = 0
            elif v == "x1000":
                total += ((hund + rest + pend) or 1) * 1000
                hund = rest = pend = 0
            elif v == 10:
                rest += 10
            elif t in ("lăm", "mốt", "tư") and rest:
                rest += v
            else:
                pend = v
        return total + hund + rest + pend if seen else None

    def _en_numbers(self, text: str) -> set[int]:
        vals: set[int] = set(int(m) for m in re.findall(r"(?<![\d.])(\d+)(?![\d])", text))
        toks = re.findall(r"[a-z]+", text.replace("-", " "))
        cur, seen = 0, False
        for ti, t in enumerate(toks + ["#"]):
            v = self.en_num.get(t)
            if t == "and" and seen and ti + 1 < len(toks) and toks[ti + 1] in self.en_num:
                continue  # "two hundred and fifty": 'and' joins number words
            if v is None:
                if seen:
                    vals.add(cur)
                cur, seen = 0, False
                continue
            seen = True
            if isinstance(v, int):
                cur += v
            elif v == "x100":
                cur = (cur or 1) * 100
            elif v == "x1000":
                cur = (cur or 1) * 1000
        if re.search(r"(?<![a-z])(a|an)(?![a-z])", text):
            vals.add(1)
        if "half" in toks:
            vals.add(0)
        return vals

    # ---- main -----------------------------------------------------------------------------------------------------------------------
    def check_texts(self, src: str, tgt: str) -> SafetyReport:
        s, t = _norm(src).rstrip(" ?.!"), _norm(tgt)
        toks = s.split()
        checks: list[SlotCheck] = []
        reasons: list[str] = []
        critical_fail, confirm = False, False
        lex = self.lex

        def rec(slot: str, sv: str | None, tv: str | None, ok: bool, why: str, critical: bool = True) -> None:
            nonlocal critical_fail, confirm
            checks.append(SlotCheck(slot=slot, src_value=sv, tgt_value=tv, preserved=ok))  # type: ignore[arg-type]
            if not ok:
                reasons.append(why)
                if critical:
                    critical_fail = True
                else:
                    confirm = True

        # negation (question particle = final "không"/"chưa" or "có ... không" => not a negation)
        if self.enable["negation"]:
            cues = lex["negation"]["src_cues"]
            neg_idx = [i for i, w in enumerate(toks) if w in cues]
            qfinal = lex["negation"]["question_final"]
            real = []
            for i in neg_idx:
                if i == len(toks) - 1 and toks[i] in qfinal:
                    continue  # sentence-final particle
                if toks[i] == "không" and "có" in toks[:i] and i >= len(toks) - 2 and i != 0:
                    continue  # "có ... không" frame
                real.append(i)
            hit = _any(t, lex["negation"]["tgt_forms"]) if real else None
            if real:
                rec("negation", toks[real[0]], hit, hit is not None, "negation_missing")
            elif self.enable["extra_negation"]:
                extra = _any(t, lex["negation"]["tgt_forms_extra_negation_check"])
                if extra:
                    rec("negation", None, extra, False, "negation_added_contradictory")

        # allergy
        if self.enable["allergy"]:
            for term in lex["allergy"]["src"]:
                if _word_re(term).search(s):
                    hit = _any(t, lex["allergy"]["tgt_forms"])
                    rec("allergy", term, hit, hit is not None, "allergy_missing")

        # medication + other allergens (longest first, span consumed)
        if self.enable["medication"]:
            rest = s
            for m in self.meds:
                rx = _word_re(m["vi"])
                mo = rx.search(rest)
                if mo:
                    rest = rest[:mo.start()] + " " * (mo.end() - mo.start()) + rest[mo.end():]
                    hit = _any_word(t, m["en"])
                    rec("medication", m["vi"], hit, hit is not None, f"medication_missing:{m['vi']}")
            if "dị ứng" in s:
                for m in lex["allergens_other"]:
                    if _word_re(m["vi"]).search(s):
                        hit = _any_word(t, m["en"])
                        rec("medication", m["vi"], hit, hit is not None, f"allergen_missing:{m['vi']}")

        # dose: number immediately before a dose unit; unit must appear, number must match
        if self.enable["dose"]:
            en_vals = self._en_numbers(t)
            for u in self.units:
                uw = u["vi"]
                for i, w in enumerate(toks):
                    if w != uw:
                        continue
                    j = i
                    while j > 0 and (toks[j - 1] in self.vi_num or toks[j - 1].isdigit()):
                        j -= 1
                    num = self._parse_vi_number(toks[j:i]) if j < i else None
                    hit = _any(t, u["en"])
                    if num is None:
                        # unit without a number (e.g. "uống thuốc theo viên"): only the unit must survive if it is a strict unit
                        if uw not in ("viên",):
                            rec("dose_unit", uw, hit, hit is not None, f"dose_unit_missing:{uw}")
                        continue
                    rec("dose_unit", uw, hit, hit is not None, f"dose_unit_missing:{uw}")
                    ok = num in en_vals
                    rec("dose_number", str(num), ",".join(map(str, sorted(en_vals))) or None, ok, f"dose_number_mismatch:{num}")

        # intensity -> CONFIRM
        if self.enable["intensity"]:
            # "quá" directly before a number = "more than N" (a limit, e.g. "không uống quá hai viên"), not the intensifier "quá". Found on the TEST split
            # after the single run: this fix is v1.2 and is NOT part of the reported test numbers.
            nums = "|".join(re.escape(k) for k, v in self.vi_num.items() if len(k) > 1 and k not in ("mươi", "trăm", "nghìn", "ngàn"))
            rest = re.sub(r"quá (?=(?:\d|" + nums + r")(?!\w))", "    ", s)
            for m in self.inten:
                rx = _word_re(m["vi"])
                mo = rx.search(rest)
                if mo:
                    rest = rest[:mo.start()] + " " * (mo.end() - mo.start()) + rest[mo.end():]
                    hit = _any(t, m["en"])
                    rec("intensity", m["vi"], hit, hit is not None, f"intensity_missing:{m['vi']}", critical=False)

        # Preserve the public text-only helper and number parser for existing tools.
        # Add relational checks; ordinary/intensity-only legacy calls remain lexical.
        # Explicit legacy ablation flags are text-evaluation-only. Runtime check()
        # always uses the strict checker; disabling a lexical group cannot bypass it.
        if all(self.enable.values()):
            relation = self.clinical.check_texts(src, tgt)
            if relation.status == "FAIL":
                critical_fail = True
                reasons.extend(relation.reasons)
            elif relation.status == "CONFIRM" and any(c.slot in ("medication", "dose_unit", "dose_number", "allergy", "negation")
                                                      for c in checks):
                confirm = True
                reasons.extend(relation.reasons)
        return SafetyReport(passed=not critical_fail, confirm=confirm and not critical_fail,
                            checks=checks, reasons=list(dict.fromkeys(reasons)))

    def check(self, mt: MtResult) -> SafetyReport:
        # Runtime path is deliberately strict, including EOS/tokenizer evidence.
        return self.clinical.check(mt)
