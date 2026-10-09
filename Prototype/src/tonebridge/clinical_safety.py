"""Conservative bilingual clinical relation checker with explicit parse coverage.

This is a finite grammar, not general semantic equivalence or medical advice.
Unsupported text requires confirmation. Korean lexicon is draft pending review.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import re
import unicodedata

from tonebridge.clinical_numbers import number
from tonebridge.nmt_evidence import DIRECTIONS, DetailedSafetyReport

ROOT = Path(__file__).resolve().parents[2]


def norm(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    for short, expanded in {"can't": "cannot", "don't": "do not", "doesn't": "does not",
                            "isn't": "is not", "aren't": "are not", "won't": "will not"}.items():
        text = text.replace(short, expanded)
    return re.sub(r"\s+", " ", text).strip()


def term_pattern(term, lang):
    if lang == "ko":
        # Limited particles only. Arbitrary Hangul suffixes do NOT match drug identities.
        return re.compile(r"(?<![a-z0-9가-힣])" + re.escape(term)
                          + r"(?:에게|에서|에는|으로|은|는|에|을|를|과|와|이|가|의|로|도)?(?![a-z0-9가-힣])")
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")


class ClinicalSafetyChecker:
    def __init__(self, src_lang="vi", tgt_lang="en", lexicon=ROOT / "configs/clinical_lexicon.json",
                 korean_reviewed=False):
        self.direction = f"{src_lang}-{tgt_lang}"
        self.src_lang, self.tgt_lang = src_lang, tgt_lang
        self.lex = json.loads(Path(lexicon).read_text(encoding="utf-8")) if not isinstance(lexicon, dict) else lexicon
        self.korean_reviewed = korean_reviewed and self.lex.get("korean_reviewed", False)

    def _parse(self, text, lang):
        s = norm(text)
        question = "?" in s or (lang == "vi" and s.startswith(("bạn có ", "có "))
                               and bool(re.search(r"(?:không|chưa)[?.!]*$", s))) or (
            lang == "en" and s.startswith(("do you ", "are you ", "can you ", "does "))) or (
            lang == "ko" and bool(re.search(r"(?:습니까|나요|까요)", s)))
        if lang == "vi":
            s = re.sub(r"(?:\s+không|\s+chưa)[?.!]*$", "", s) if s.startswith(("bạn có ", "có ")) else s
        s = re.sub(r"(?<=hundred) and (?=\w)", " ", s)
        clauses = re.split(r"\s+(?:và|nhưng|and|but|그리고|하지만|및)\s+|[;!?。]|\.(?![0-9])", s)
        events, entities, reasons = [], [], []
        patterns = dict(self.lex["patterns"][lang])
        if lang in ("vi", "en"):
            patterns = {key: r"(?<!\w)(?:" + value + r")(?!\w)" for key, value in patterns.items()}
        inherited_action = None
        inherited_neg = False
        inherited_actor = None
        for clause in filter(str.strip, clauses):
            c = clause.strip()
            residue = list(c)
            actor = next((name for name, forms in {
                "speaker": {"vi": ["tôi"], "en": ["i"], "ko": ["저는", "제가", "나는"]},
                "patient": {"vi": ["bệnh nhân"], "en": ["patient"], "ko": ["환자는", "환자가"]},
                "listener": {"vi": ["bạn"], "en": ["you"], "ko": ["당신은"]},
                "third_female": {"vi": ["cô ấy"], "en": ["she"], "ko": ["그녀는"]},
                "third_male": {"vi": ["anh ấy"], "en": ["he"], "ko": ["그는"]}
            }.items() if any(term_pattern(form, lang).search(c) for form in forms[lang])), None)

            def erase(start, end):
                residue[start:end] = " " * (end - start)

            found = []
            forms = [(e, f) for e in self.lex["entities"] for f in e["forms"][lang]]
            for entity, form in sorted(forms, key=lambda row: -len(row[1])):
                for match in term_pattern(form, lang).finditer(c):
                    if all(v == " " for v in residue[match.start():match.end()]):
                        continue
                    erase(*match.span())
                    found.append(entity["id"])
            entities.extend(found)
            if len(found) > 1:
                reasons.append("ambiguous_entity_binding")
            action = None
            action_hits = set()
            for key in ("stop", "continue", "allergy", "inject", "take"):
                matches = list(re.finditer(patterns[key], c))
                if matches:
                    action_hits.add(key)
                if matches and action is None:
                    action = key
                for match in matches:
                    erase(*match.span())
            neg = bool(re.search(patterns["negation"], c))
            if len(list(re.finditer(patterns["negation"], c))) > 1:
                reasons.append("ambiguous_negation_scope")
            uncertain = bool(re.search(patterns["certainty"], c))
            method = "inject" if "inject" in action_hits else "take" if "take" in action_hits else None
            if ({"stop", "continue"}.issubset(action_hits)
                    or {"inject", "take"}.issubset(action_hits)
                    or ("allergy" in action_hits and len(action_hits) > 1)):
                reasons.append("ambiguous_action_scope")
            if action is None and found:
                if inherited_neg and not neg:
                    reasons.append("ambiguous_negation_inheritance")
                action = inherited_action or ("symptom" if any(e["id"] in found and e["kind"] == "symptom"
                                                             for e in self.lex["entities"]) else "mention")
                neg = neg or inherited_neg
                method = "take" if action in ("take", "stop", "continue") else "inject" if action == "inject" else None
            elif action:
                inherited_action, inherited_neg = action, neg
            actor = actor or inherited_actor or ("listener" if action in ("take", "inject", "stop", "continue") else "unspecified")
            inherited_actor = actor
            if action in ("stop", "continue") and method is None:
                reasons.append("administration_method_unspecified")
            comparator = None
            for key in ("comparator_more", "comparator_less"):
                if re.search(patterns[key], c):
                    comparator = key
            frequency = None
            for key in ("frequency_every", "frequency_daily"):
                match = re.search(patterns[key], c)
                if match:
                    n = number(match["n"], lang)
                    if n is None:
                        reasons.append("unsupported_frequency")
                    else:
                        u = match.groupdict().get("u", "day")
                        u = "h" if u in ("giờ", "hour", "hours", "시간") else "day"
                        frequency = (key, str(n.normalize()), u)
                        erase(*match.span())
            # Units are longest-first and cannot match inside another unit.
            doses, used = [], set()
            forms = [(u, f) for u in self.lex["units"] for f in u["forms"][lang]]
            for unit, form in sorted(forms, key=lambda row: -len(row[1])):
                # Digits may be attached (5mg / 2정), alphabetic prefixes may not.
                tail = r"(?:을|를|씩|으로|은|는)?(?![a-z가-힣])" if lang == "ko" else r"(?!\w)"
                rx = re.compile(r"(?<![a-z가-힣])" + re.escape(form) + tail)
                for match in rx.finditer(c):
                    if used.intersection(range(*match.span())):
                        continue
                    used.update(range(*match.span()))
                    prefix = c[:match.start()].rstrip()
                    # Try bounded adjacent number phrases, shortest valid suffix first.
                    tokens = list(re.finditer(r"[^\s]+", prefix))
                    value, start = None, match.start()
                    for token in reversed(tokens[-8:]):
                        candidate = prefix[token.start():]
                        value = number(candidate, lang)
                        if value is not None:
                            start = token.start()
                            # Grow through adjacent number words for "twenty five", "hai mươi lăm".
                            continue
                        if start < match.start():
                            break
                    value = number(prefix[start:], lang) if start < match.start() else None
                    if value is None:
                        reasons.append("unsupported_dose_number")
                        erase(*match.span())
                        continue
                    # Exact dimension conversion only; never infer a medication dose.
                    canonical = unit["id"]
                    if canonical in ("mg", "mcg", "g"):
                        value *= {"mcg": Decimal(1), "mg": Decimal(1000), "g": Decimal(1000000)}[canonical]
                        canonical = "mass_mcg"
                    doses.append((str(value.normalize()), canonical))
                    erase(start, match.end())
            for key in ("negation", "certainty", "comparator_more", "comparator_less"):
                for match in re.finditer(patterns[key], c):
                    erase(*match.span())
            # Function words do not convey additional clinical facts.
            for word in self.lex["filler"][lang].split():
                for match in term_pattern(word, lang).finditer(c):
                    erase(*match.span())
            remaining = re.sub(r"[^\w가-힣]+", " ", "".join(residue)).strip()
            if remaining:
                reasons.append("unparsed_content:" + remaining)
            if found or doses or action:
                if action == "mention" and doses:
                    reasons.append("unsupported_dose_action")
                events.append((tuple(sorted(found)), action, method, actor, question, neg, uncertain, comparator,
                               tuple(sorted(doses)), frequency))
        return Counter(events), Counter(entities), sorted(set(reasons))

    def check_texts(self, src, tgt):
        if self.direction not in DIRECTIONS:
            return DetailedSafetyReport.decision("CONFIRM", self.direction, ["checker_unavailable"])
        if not isinstance(src, str) or not isinstance(tgt, str) or not src.strip() or not tgt.strip():
            return DetailedSafetyReport.decision("FAIL", self.direction, ["empty_or_invalid_text"])
        if max(len(src), len(tgt)) > 4096:
            return DetailedSafetyReport.decision("CONFIRM", self.direction, ["text_exceeds_checker_limit"])
        a, b = norm(src).strip(".!?"), norm(tgt).strip(".!?")
        ordinary = any(a in entry[self.src_lang] and b in entry[self.tgt_lang] for entry in self.lex["ordinary"])
        se, sm, sr = self._parse(src, self.src_lang) if not ordinary else (Counter(), Counter(), [])
        te, tm, tr = self._parse(tgt, self.tgt_lang) if not ordinary else (Counter(), Counter(), [])
        if sm != tm:
            return DetailedSafetyReport.decision("FAIL", self.direction, ["entity_identity_missing_added_or_changed"], ["entities"])
        if se != te and (sr or tr):
            return DetailedSafetyReport.decision("CONFIRM", self.direction,
                                                ["relation_not_comparable"] + sorted(set(sr + tr)), ["entities"])
        if se != te:
            return DetailedSafetyReport.decision("FAIL", self.direction, ["clinical_relation_mismatch"], ["entities", "event_relations"])
        pending = sr + tr
        if not ordinary and not se:
            pending.append("semantic_coverage_unverified")
        if "ko" in (self.src_lang, self.tgt_lang) and not self.korean_reviewed:
            pending.append("korean_lexicon_pending_human_review")
        return DetailedSafetyReport.decision("CONFIRM" if pending else "PASS", self.direction,
                                            sorted(set(pending)), ["entities", "event_relations"] if se else ["ordinary_phrase"])

    def check(self, mt):
        if (mt.src_lang, mt.tgt_lang) != (self.src_lang, self.tgt_lang):
            return DetailedSafetyReport.decision("FAIL", self.direction, ["direction_mismatch"])
        semantic = self.check_texts(mt.src_text, mt.tgt_text)
        ev = getattr(mt, "evidence", None)
        if ev is None:
            return DetailedSafetyReport.decision(
                "FAIL" if semantic.status == "FAIL" else "CONFIRM", self.direction,
                ["translation_evidence_missing"] + semantic.reasons, semantic.coverage)
        issues = []
        if ev.truncated or ev.input_truncated or not ev.eos_reached:
            issues.append("translation_incomplete")
        if ev.source_unknown_tokens or ev.output_unknown_tokens or ev.tokenizer_issues:
            issues.append("tokenizer_or_unknown_token_issue")
        if ev.constraints_satisfied < ev.constraints_requested or ev.unsatisfied_constraints:
            issues.append("constraints_unsatisfied")
        if (ev.constraints_satisfied > ev.constraints_requested
                or ev.source_unknown_tokens > ev.source_tokens
                or ev.output_unknown_tokens > ev.output_tokens):
            issues.append("invalid_evidence_counts")
        if issues:
            return DetailedSafetyReport.decision(
                "FAIL" if semantic.status == "FAIL" else "CONFIRM", self.direction,
                issues + semantic.reasons, semantic.coverage)
        return semantic
