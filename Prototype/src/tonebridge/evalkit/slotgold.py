"""Gold slot-preservation check for the safety set: per-item accepted English forms (written in tools/make_safety_set.py, independent of
the safety-check lexicon). An item is 'preserved' when every PRIMARY slot has at least one accepted form in the hypothesis."""
from __future__ import annotations

import re


def _has(text: str, form: str) -> bool:
    left = "" if form.startswith("n't") else r"(?<![a-z0-9])"
    right = r"(?![a-z0-9])" if (len(form.strip()) <= 3 or form[0].isdigit()) else ""
    # Historical dataset lists "egg" while its own correct reference says "eggs".
    # Repair this known noun inflection; do not turn arbitrary prefixes into gold.
    noun_suffix = r"s?" if form == "egg" else ""
    return re.search(left + re.escape(form) + noun_suffix + right, text) is not None


def norm_en(t: str) -> str:
    return t.lower().replace("’", "'").replace("‘", "'")


def slot_ok(slot: dict, hyp: str) -> bool:
    h = norm_en(hyp)
    return any(_has(h, f) for f in slot["forms"])


def item_preserved(item: dict, hyp: str, primary_only: bool = True) -> bool:
    return all(slot_ok(s, hyp) for s in item["slots"] if (s.get("primary") or not primary_only))


def failed_slots(item: dict, hyp: str) -> list[str]:
    return [s["kind"] for s in item["slots"] if not slot_ok(s, hyp)]
