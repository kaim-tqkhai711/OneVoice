"""Exact decimal parsing for clinical text; no float rounding or guessed dose."""
from decimal import Decimal, InvalidOperation
import re

VI = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
      "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9}
EN = dict(zip(("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split(), range(20)))
EN.update(dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10))))
KO = {"영": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9,
      "한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5,
      "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}


def number(text: str, lang: str) -> Decimal | None:
    s = text.strip().lower()
    try:
        if re.fullmatch(r"[0-9]+/[0-9]+", s):
            a, b = map(Decimal, s.split("/"))
            return a / b if b else None
        if re.fullmatch(r"(?:[0-9]+(?:[.,][0-9]+)?|[.,][0-9]+)", s):
            return Decimal(s.replace(",", "."))
    except InvalidOperation:
        return None
    if s in ("half", "nửa", "반"):
        return Decimal("0.5")
    if s in ("a", "an") and lang == "en":
        return Decimal(1)
    if lang == "vi" and (" phẩy " in s or " chấm " in s):
        a, b = re.split(r" phẩy | chấm ", s, maxsplit=1)
        left = number(a, lang)
        digits = [VI.get(t) for t in b.split()]
        if left is not None and digits and all(v is not None for v in digits):
            return Decimal(str(left) + "." + "".join(map(str, digits)))
        return None
    toks = s.replace("-", " ").split()
    if lang == "vi":
        total = group = pending = 0
        seen = False
        previous_digit = False
        for t in toks:
            if t in ("linh", "lẻ"):
                previous_digit = False
                continue
            if t in VI:
                if previous_digit:
                    return None  # "hai ba viên" is ambiguous, not silently 3
                pending = VI[t]; seen = True
                previous_digit = True
            elif t in ("mười", "mươi"):
                group += (pending if previous_digit else 1) * 10; pending = 0; seen = True
                previous_digit = False
            elif t == "trăm":
                group += (pending if previous_digit else 1) * 100; pending = 0; seen = True
                previous_digit = False
            elif t in ("nghìn", "ngàn"):
                total += (group + pending or 1) * 1000; group = pending = 0; seen = True
                previous_digit = False
            else:
                return None
        return Decimal(total + group + pending) if seen else None
    if lang == "en":
        total = current = 0
        seen = False
        previous_word_number = None
        for t in toks:
            if t == "and" and seen:
                continue
            if t in EN:
                if previous_word_number is not None and not (previous_word_number >= 20 and EN[t] < 10):
                    return None
                current += EN[t]; seen = True
                previous_word_number = EN[t]
            elif t == "hundred":
                current = (current or 1) * 100; seen = True
                previous_word_number = None
            elif t == "thousand":
                total += (current or 1) * 1000; current = 0; seen = True
                previous_word_number = None
            else:
                return None
        return Decimal(total + current) if seen else None
    if lang == "ko":
        if s in KO:
            return Decimal(KO[s])
        total = pending = 0
        previous_digit = False
        for char in s.replace(" ", ""):
            if char in KO:
                if previous_digit:
                    return None
                pending = KO[char]
                previous_digit = True
            elif char in {"십": 10, "백": 100, "천": 1000}:
                total += (pending if previous_digit else 1) * {"십": 10, "백": 100, "천": 1000}[char]; pending = 0
                previous_digit = False
            else:
                return None
        return Decimal(total + pending) if s else None
    return None
