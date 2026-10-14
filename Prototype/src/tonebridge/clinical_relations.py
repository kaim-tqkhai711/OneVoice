"""Scoped, deterministic VI/EN clinical relations for short communication turns.

This is a bounded grammar, not a general medical language model. Recognized
objects, dosage, frequency, subjects and predicates are compared as relations;
unsupported syntax in a critical clause produces an issue, never implicit PASS.
Opaque drug names can be copied exactly when anchored by a medical predicate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import re
import unicodedata


PARSER_VERSION = "clinical-relations-v1"
NUMBER = r"[+-]?\d+(?:[.,]\d+)?"


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFC", text).lower().replace("’", "'")
    for contraction, expanded in {"can't": "cannot", "don't": "do not", "doesn't": "does not",
        "isn't": "is not", "aren't": "are not", "haven't": "have not", "hasn't": "has not",
        "hadn't": "had not", "won't": "will not", "i'm": "i am", "i've": "i have",
        "he's": "he is", "she's": "she is"}.items():
        value = re.sub(r"(?<!\w)" + re.escape(contraction) + r"(?!\w)", expanded, value)
    return re.sub(r"\s+", " ", value).strip()


def decimal_number(value: str, lang: str, numbers: dict) -> Decimal | None:
    """Parse decimal digits or number words exactly; commas are decimal separators.

    Grouped thousands, ranges and fractions other than half are not a supported
    notation. Never recover a trailing digit from a malformed decimal string.
    """
    value = value.strip().replace("-", " ") if not re.fullmatch(NUMBER, value.strip()) else value.strip()
    if re.fullmatch(NUMBER, value):
        try:
            return Decimal(value.replace(",", "."))
        except InvalidOperation:
            return None
    if value in ("half", "a half", "nửa", "một nửa"):
        return Decimal("0.5")
    decimal_marker = "point" if lang == "en" else r"phẩy|chấm"
    parts = re.split(r"\s+(?:" + decimal_marker + r")\s+", value)
    if len(parts) == 2:
        integer = decimal_number(parts[0], lang, numbers)
        digits = [numbers.get(token) for token in parts[1].split()]
        if integer is not None and integer == integer.to_integral_value() and digits and all(isinstance(v, int) and 0 <= v <= 9 for v in digits):
            return Decimal(str(integer) + "." + "".join(str(v) for v in digits))
        return None
    tokens = value.split()
    for left, right in zip(tokens, tokens[1:]):
        lv, rv = numbers.get(left), numbers.get(right)
        if isinstance(lv, int) and isinstance(rv, int):
            # Adjacent unit digits ("one two", "hai năm") are not a valid
            # cardinal; summing/keeping the last digit would invent a dosage.
            if lv < 10 or rv >= 10:
                return None
    if lang == "en":
        if tokens in (["a"], ["an"]):
            return Decimal(1)
        total = current = 0
        seen = False
        for token in tokens:
            if token == "and" and seen:
                continue
            if token in ("a", "an") and not seen:
                current, seen = 1, True
                continue
            number = numbers.get(token)
            if number is None:
                return None
            seen = True
            if number == "x100":
                current = (current or 1) * 100
            elif number == "x1000":
                total += (current or 1) * 1000
                current = 0
            elif isinstance(number, int):
                current += number
            else:
                return None
        return Decimal(total + current) if seen else None
    total = hundreds = tens = pending = 0
    seen = False
    for token in tokens:
        number = numbers.get(token)
        if number is None:
            if token in ("linh", "lẻ") and seen:
                continue
            return None
        seen = True
        if number == "x10":
            tens += (pending or 1) * 10
            pending = 0
        elif number == "x100":
            hundreds = (pending or 1) * 100
            pending = 0
        elif number == "x1000":
            total += (hundreds + tens + pending or 1) * 1000
            hundreds = tens = pending = 0
        elif number == 10:
            tens += 10
        elif token in ("mốt", "lăm", "tư") and tens:
            tens += number
        elif isinstance(number, int):
            pending = number
        else:
            return None
    return Decimal(total + hundreds + tens + pending) if seen else None


@dataclass(frozen=True)
class Frequency:
    kind: str  # interval or rate; these are deliberately not interchanged
    amount: Decimal
    unit: str


@dataclass(frozen=True)
class ClinicalRelation:
    """Subject denotes the experiencer/recipient of the clinical predicate."""
    subject: str
    drug: str | None
    amount: Decimal | None
    unit: str | None
    frequency: Frequency | None
    action: str
    negation: str
    modality: str
    symptom: str | None = None
    limit: str = "exact"
    context: tuple[str, ...] = ()  # conditions, route/site, history and timing

    def as_tuple(self) -> tuple:
        """The eight requested fields; symptom/limit remain additional evidence."""
        return (self.subject, self.drug, self.amount, self.unit, self.frequency,
                self.action, self.negation, self.modality)


@dataclass
class ParseResult:
    relations: list[ClinicalRelation] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    critical: bool = False


def _phrase(form: str) -> str:
    return r"(?<!\w)" + re.escape(form) + r"(?!\w)"


def _matches(text: str, aliases: dict[str, str]) -> list[tuple[int, int, str]]:
    found, occupied = [], set()
    for form, canonical in sorted(aliases.items(), key=lambda it: -len(it[0])):
        for match in re.finditer(_phrase(form), text):
            span = set(range(match.start(), match.end()))
            if not span & occupied:
                found.append((match.start(), match.end(), canonical))
                occupied |= span
    return sorted(found)


def _erase(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


class RelationParser:
    """Pure parser over the same lexicon snapshot used by NMT and safety."""

    def __init__(self, lexicon: dict):
        self.lex = lexicon
        self.drugs = {"vi": {}, "en": {}}
        for entry in lexicon["medications"] + lexicon["allergens_other"]:
            canonical = entry["en"][0].rstrip("s") if entry["en"][0] in ("antibiotics", "painkillers") else entry["en"][0]
            self.drugs["vi"][entry["vi"]] = canonical
            for form in entry["en"]:
                self.drugs["en"][form] = canonical
                self.drugs["en"][form + "s"] = canonical
        self.drugs["vi"]["thuốc"] = "generic_medication"
        for form in ("medicine", "medicines", "medication", "medications", "drug", "drugs"):
            self.drugs["en"][form] = "generic_medication"
        self.symptoms = {"vi": {}, "en": {}}
        for entry in lexicon.get("symptoms", []):
            for lang in ("vi", "en"):
                for form in entry[lang]:
                    self.symptoms[lang][form] = entry["id"]
        self.units = {"vi": {}, "en": {}}
        canonical_units = {"miligam": "mg", "microgam": "mcg", "mililít": "ml", "gam": "g",
                           "viên": "tablet", "giọt": "drop", "ống": "vial"}
        for entry in lexicon["dose_units"]:
            canonical = canonical_units.get(entry["vi"], entry["vi"])
            self.units["vi"][entry["vi"]] = canonical
            for form in entry["en"]:
                self.units["en"][form] = canonical
                self.units["en"][form + "s"] = canonical
        # Short forms are commonly produced by ASR in either language.
        for lang in ("vi", "en"):
            for form in ("mg", "mcg", "ml", "g"):
                self.units[lang][form] = form

    def _numbers(self, text: str, lang: str) -> str:
        """Canonicalize complete number spans before clause splitting."""
        numbers = self.lex[lang + "_numbers"]
        compact_units = "|".join(re.escape(u) for u in sorted(self.units[lang], key=len, reverse=True))
        text = re.sub(r"(?<=\d)(?=(?:" + compact_units + r")(?!\w))", " ", text)
        text = re.sub(r"(?<!\w)" + NUMBER + r"(?!\w)", lambda m: str(Decimal(m[0].replace(",", "."))), text)
        words = sorted(set(numbers) | ({"half", "a", "an", "and", "point"} if lang == "en" else {"nửa", "phẩy", "chấm", "linh", "lẻ"}), key=len, reverse=True)
        starts = [w for w in words if w != "and"]
        pattern = r"(?<!\w)(?:" + "|".join(re.escape(w) for w in starts) + r")(?:[ -]+(?:" + "|".join(re.escape(w) for w in words) + r"))*(?!\w)"

        def convert(match):
            raw = match[0].strip()
            tail = text[match.end():].lstrip()
            # 'không' is zero only before a unit or decimal marker, not a predicate.
            if raw == "không" and not any(re.match(_phrase(u), tail) for u in self.units[lang]):
                return raw
            if raw in ("a", "an", "and") and not any(re.match(_phrase(u), tail) for u in self.units[lang]):
                return raw
            # Do not consume a trailing coordinating 'and' into a number.
            trailing = " and" if raw.endswith(" and") else ""
            value = decimal_number(raw[:-4] if trailing else raw, lang, numbers)
            return str(value) + trailing if value is not None else raw

        return re.sub(pattern, convert, text)

    def _frequency(self, text: str, lang: str) -> tuple[Frequency | None, list[tuple[int, int]], list[str]]:
        time_unit = r"giờ|phút|ngày|hours?|minutes?|days?"
        intervals = list(re.finditer(r"\b(?:mỗi|every|per)\s+(" + NUMBER + r")?\s*(" + time_unit + r")\b", text))
        intervals += list(re.finditer(r"\bq(" + NUMBER + r")(h)\b", text))
        rates = list(re.finditer(r"\b(" + NUMBER + r"|once|twice)\s*(?:lần|times?)?\s*(?:một|a|per|1)\s+(?:ngày|day)\b", text))
        daily = list(re.finditer(r"\b(?:daily|hằng ngày|hàng ngày)\b", text))
        daily += list(re.finditer(r"\ba day\b", text)) if not rates else []
        candidates = []
        for match in intervals:
            value = Decimal((match[1] or "1").replace(",", "."))
            unit = match[2]
            multiplier = Decimal(86400) if unit in ("ngày", "day", "days") else Decimal(60) if unit in ("phút", "minute", "minutes") else Decimal(3600)
            candidates.append((match.span(), Frequency("interval", value * multiplier, "second")))
        for match in rates:
            value = {"once": Decimal(1), "twice": Decimal(2)}.get(match[1])
            if value is None:
                value = Decimal(match[1].replace(",", "."))
            candidates.append((match.span(), Frequency("rate", value, "day")))
        candidates += [(m.span(), Frequency("interval", Decimal(86400), "second")) for m in daily]
        if len(candidates) > 1:
            return None, [c[0] for c in candidates], ["ambiguous_frequency"]
        return (candidates[0][1], [candidates[0][0]], []) if candidates else (None, [], [])

    @staticmethod
    def _subject(text: str, lang: str, inherited: str | None, action: str) -> str:
        aliases = {"vi": {"con tôi": "my_child", "bệnh nhân": "patient", "cô ấy": "she", "anh ấy": "he",
                    "chúng tôi": "we", "tôi": "self", "bạn": "you"},
                   "en": {"my child": "my_child", "the patient": "patient", "patient": "patient", "she": "she",
                    "he": "he", "i": "self", "we": "we", "you": "you", "your": "you", "my": "self"}}
        found = _matches(text, aliases[lang])
        # We give YOU an injection: the relation's subject is the recipient.
        if action == "inject" and any(item[2] == "you" for item in found):
            return "you"
        return found[0][2] if found else inherited or ("you" if action in ("take", "stop", "continue", "start", "inject", "limit", "bring", "own", "apply", "need") else "unspecified")

    def parse(self, text: str, lang: str) -> ParseResult:
        if lang not in ("vi", "en"):
            return ParseResult(issues=["unsupported_language"], critical=True)
        text = self._numbers(normalize(text), lang)
        # Decimal punctuation is not a clause boundary. Numeric 'and' has already
        # been consumed as part of a number by _numbers.
        parts = re.split(r"\s+(nhưng|but|và|and|rồi|then)\s+|(?<!\d)[,.;!?]|[,.;!?](?!\d)", text)
        result, prior, connector = ParseResult(), None, None
        if re.search(r"\d[.,]\d+[.,]\d", text):
            result.issues.append("malformed_decimal")
            result.critical = True
        for part in parts:
            if part is None or not part.strip():
                continue
            if part in ("nhưng", "but", "và", "and", "rồi", "then"):
                connector = part
                continue
            clause = part.strip()
            parsed = self._clause(clause, lang, prior, connector, text)
            result.relations.extend(parsed.relations)
            result.issues.extend(parsed.issues)
            result.critical |= parsed.critical
            if parsed.relations:
                prior = parsed.relations[-1]
            connector = None
        result.issues = list(dict.fromkeys(result.issues))
        return result

    def _clause(self, text: str, lang: str, prior: ClinicalRelation | None, connector: str | None, whole: str) -> ParseResult:
        drugs = _matches(text, self.drugs[lang])
        symptoms = _matches(text, self.symptoms[lang])
        unit_hits = _matches(text, self.units[lang])
        action_patterns = {
            "allergy": r"dị ứng|allergic|allergy|allergies",
            "stop": r"ngừng|dừng|stop|discontinue|ceased?",
            "continue": r"tiếp tục|continue|keep taking",
            "start": r"bắt đầu|start|begin",
            "inject": r"tiêm|truyền|inject|injection|infuse|infusion|give(?: you)?\s+(?:an?\s+)?(?:\d|injection)",
            "take": r"uống|dùng|take|taken|taking|takes|took|use|using|drink|drinking",
            "need": r"cần|need|needs",
            "prescribe": r"kê|prescribe|prescribed",
            "bring": r"mang theo|bring",
            "own": r"đây là|this is your",
            "dose": r"liều là|the dose is",
            "apply": r"nhỏ|put|apply",
        }
        action_matches = [(key, list(re.finditer(r"(?<!\w)(?:" + pattern + r")(?!\w)", text))) for key, pattern in action_patterns.items()]
        actions = [key for key, matches in action_matches if matches]
        action = actions[0] if actions else None
        limit = "upper" if re.search(r"không (?:uống |dùng )?quá|no more than|not (?:take )?more than|maximum|tối đa", text) else "exact"
        if limit == "upper":
            action = "limit"
        # Opaque names are recognized only immediately after a clinical object
        # anchor. They are never guessed from arbitrary narrative words.
        anchors = (r"(?:dị ứng(?: với)?|(?:ngừng |tiếp tục |bắt đầu )?(?:uống|dùng)|thuốc)\s+" if lang == "vi"
                   else r"(?:allergic to|allergy to|(?:stop |continue |start )?(?:taking|take)|medication(?: called)?)\s+")
        for match in re.finditer(anchors + r"([a-z][a-z0-9-]*)\b", text):
            span = match.span(1)
            token = match[1]
            ignored = {"với", "nước", "nhiều", "một", "hai", "the", "a", "an", "two", "this", "any", "water", "more"}
            if token not in ignored and not any(s <= span[0] < e for s, e, _ in drugs + symptoms + unit_hits):
                drugs.append((*span, "opaque:" + token))
        drugs.sort()
        # Opaque coordinated objects inherit the predicate, not its drug identity.
        # Otherwise 'take aspirin and clopidogrel' silently loses its second drug.
        if prior and connector in ("and", "và") and not actions and not drugs and not symptoms:
            object_text = re.sub(r"^(?:the|a|an)\s+", "", text)
            if re.fullmatch(r"[a-z][a-z0-9-]*", object_text):
                span = (text.index(object_text), text.index(object_text) + len(object_text))
                drugs.append((*span, "opaque:" + object_text))
        entity_residue = _erase(text, [(s, e) for s, e, _ in drugs + symptoms])
        unsupported_clinical = bool(re.search(r"\b(?:đau|sưng|ung thư|chảy máu|pain|cancer|bleeding|swelling|diagnosed|diagnosis)\b", entity_residue))
        critical = bool(drugs or symptoms or unit_hits or action == "allergy" or limit == "upper" or unsupported_clinical)
        if not critical:
            return ParseResult()
        issues = []
        if unsupported_clinical:
            issues.append("unparsed_clinical_entity")
        if len(actions) > 1 and not (len(actions) == 2 and actions[1] == "take" and actions[0] in ("stop", "continue", "start")):
            issues.append("ambiguous_multiple_predicates")
        if len(drugs) > 1 or len(symptoms) > 1:
            issues.append("ambiguous_multiple_entities")
        fragment = action is None and not symptoms
        if action is None:
            action = "experience" if symptoms else prior.action if prior and connector in ("and", "và") else None
        if action is None:
            issues.append("unparsed_clinical_predicate")
            action = "unknown"
        if action == "take" and not drugs and symptoms:
            issues.append("ambiguous_predicate_object")
        inherited_subject = prior.subject if prior else None
        subject = self._subject(text, lang, inherited_subject, action)
        question = (bool(re.search(r"\b(?:có|nào|gì)\b.*\b(?:không|chưa)\s*$|phải không\s*$|bao lâu", text)) if lang == "vi"
                    else bool(re.match(r"(?:do|does|did|are|is|have|has|can|how)\s", text)) and not bool(re.match(r"do not\b", text)))
        question |= lang == "en" and bool(re.search(re.escape(text) + r"\s*,\s*right\s*\?", whole))
        # Question punctuation on another clause must never re-scope this one.
        modality = "question" if question else "uncertain" if re.search(r"\b(?:có thể|có lẽ|may|might|could|possibly|perhaps|maybe)\b", text) else "future" if re.search(r"\b(?:sẽ|will)\b", text) else "advised" if re.search(r"\b(?:nên|should)\b", text) else "asserted"
        negative_text = re.sub(r"\s+(?:không|chưa)\s*$", "", text) if question and lang == "vi" else text
        negation = "not_yet" if re.search(r"\bchưa\b|\bnot\b.*\byet\b", negative_text) else "negated" if re.search(r"\b(?:không|đừng|chẳng|chớ|not|no|never|cannot|without)\b", negative_text) else "affirmed"
        if re.search(r"chưa bao giờ|\bnever\b", text):
            negation = "never"
        elif lang == "en" and re.search(r"\b(?:have|has) not taken\b", text):
            negation = "not_yet"
        if limit == "upper":
            negation = "affirmed"  # the bound, not the predicate, carries 'not more than'
        if fragment and prior and connector in ("and", "và"):
            negation, modality = prior.negation, prior.modality
        frequency, frequency_spans, frequency_issues = self._frequency(text, lang)
        issues += frequency_issues
        dose_text = _erase(text, frequency_spans)
        quantities = []
        amount_spans = []
        number_words = self.lex[lang + "_numbers"]
        for start, end, unit in unit_hits:
            if any(a <= start < b for a, b in frequency_spans):
                continue
            before = dose_text[:start].rstrip()
            match = re.search(r"(?<![\w.,])(" + NUMBER + r"|half|nửa|a|an)\s*$", before)
            amount = decimal_number(match[1], lang, number_words) if match else None
            if match:
                amount_spans.append(match.span())
                if amount is None or amount < 0:
                    issues.append("invalid_dose_number")
            quantities.append((amount, unit))
        if len(quantities) > 1:
            issues.append("ambiguous_multiple_doses")
        amount, unit = quantities[0] if quantities else (None, None)
        if fragment and prior and prior.amount is not None and amount is None:
            issues.append("ambiguous_shared_dose")
        if fragment and prior and frequency is None:
            frequency = prior.frequency
        entity = drugs[0][2] if drugs else None
        symptom = symptoms[0][2] if symptoms else None
        if symptom == "blood_pressure":
            symptom += ":" + ("high" if re.search(r"\b(?:cao|high)\b", text) else "low" if re.search(r"\b(?:thấp|low)\b", text) else "unspecified")
        if action == "allergy" and not entity and symptom:
            issues.append("ambiguous_allergen")
        # Modifiers carry clinical meaning and must match; never merely whitelist
        # words such as 'pregnant', 'alcohol', 'right' or 'history'.
        contexts = {
            "with_alcohol": r"cùng với rượu bia|with alcohol",
            "if_pregnant": r"nếu bạn đang mang thai|if you are pregnant",
            "history": r"tiền sử|history of",
            "empty_stomach": r"lúc đói|on an empty stomach",
            "after_meals": r"sau bữa ăn|after meals",
            "today": r"hôm nay|today",
            "morning": r"sáng nay|this morning",
            "now": r"ngay bây giờ|right now",
            "duration_query": r"bao lâu rồi|how long",
            "appointment": r"khi đi khám|to the appointment",
            "allergic_reaction": r"phản ứng|reaction",
            "disclose_if": r"hãy cho biết nếu|tell us if",
            "right_eye": r"vào mắt phải|in the right eye",
            "left_eye": r"vào mắt trái|in the left eye",
            "worse": r"nặng hơn|worse",
        }
        context, context_spans = [], []
        for key, pattern in contexts.items():
            hits = list(re.finditer(r"(?<!\w)(?:" + pattern + r")(?!\w)", text))
            if hits:
                context.append(key)
                context_spans.extend(m.span() for m in hits)
        if symptom and symptom != "blood_pressure:high" and symptom != "blood_pressure:low":
            for grade in ("high", "low"):
                pattern = r"\b(?:cao|high)\b" if grade == "high" else r"\b(?:thấp|low)\b"
                hits = list(re.finditer(pattern, text))
                if hits:
                    context.append("symptom_" + grade)
        if "disclose_if" in context:
            modality = "conditional"
        duration = list(re.finditer(r"\b(?:trong|over)\s+(" + NUMBER + r")\s+(giờ|hours?|phút|minutes?)\b", text))
        if duration:
            if frequency is not None or len(duration) > 1:
                issues.append("ambiguous_frequency")
            m = duration[0]
            frequency = Frequency("duration", Decimal(m[1]) * (60 if m[2] in ("phút", "minute", "minutes") else 3600), "second")
            context_spans += [m.span() for m in duration]
        if lang == "vi" and re.search(r"gia đình tôi có người", text) or lang == "en" and re.search(r"someone in my family", text):
            subject = "family_member"
            pattern = r"gia đình tôi có người|someone in my family"
            context_spans.extend(m.span() for m in re.finditer(pattern, text))
        spans = [(s, e) for s, e, _ in drugs + symptoms + unit_hits] + frequency_spans + amount_spans + context_spans
        residue = _erase(text, spans)
        for _, matches in action_matches:
            residue = _erase(residue, [m.span() for m in matches])
        allowed = {
            "vi": "tôi bạn bệnh nhân con cô anh ấy chúng với bị đang đã từng có thể lẽ sẽ không chưa đừng chẳng chớ rất quá cực kỳ vô cùng hơi khá dữ dội nặng nhẹ cao thấp cho của là liều theo mỗi lần tối đa ngày nào gì vào sau bữa ăn sáng nay thêm hơn nước nhiều thấy một được nên hãy bác sĩ đây cơn phải bao giờ",
            "en": "i you the patient my child she he we to am is are was were be being been have has had do does did not no never cannot without yet may might could possibly perhaps maybe will very extremely severe severely serious really high low mildly slightly a little somewhat fairly quite intense too more than maximum dose of per each time daily please by an injection and after meals meal today this morning earlier now it your on tablet form water drink more for known any feel feeling should doctor give right with",
        }
        # Remove complete phrases first, then function words; unknown clinical
        # residue (including unbound numbers/unknown units) is unsafe evidence.
        allowed_words = set(allowed[lang].split())
        remaining = re.findall(r"[\w]+(?:[.,]\d+)?", residue)
        if any(token not in allowed_words for token in remaining):
            issues.append("unparsed_critical_clause")
        relation = ClinicalRelation(subject, entity, amount, unit, frequency, action, negation, modality, symptom, limit, tuple(sorted(context)))
        return ParseResult([relation], list(dict.fromkeys(issues)), True)


def compare_relations(source: ParseResult, target: ParseResult) -> list[str]:
    """One-to-one multiset matching preserves entity binding and scoped predicates."""
    reasons = ["source_" + code for code in source.issues] + ["target_" + code for code in target.issues]
    src, tgt = Counter(source.relations), Counter(target.relations)
    if src != tgt:
        if src - tgt:
            reasons.append("clinical_relation_missing_or_changed")
        if tgt - src:
            reasons.append("clinical_relation_added_or_changed")
    return list(dict.fromkeys(reasons))


def parse_clinical_relations(text: str, lang: str, lexicon: dict) -> ParseResult:
    return RelationParser(lexicon).parse(text, lang)
