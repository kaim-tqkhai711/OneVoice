from tonebridge.glossary import Glossary
from tonebridge.evalkit.slotgold import slot_ok


def test_drug_substring_not_a_match():
    rows = Glossary().check("uống paracetamol", "Take paracetamolized medicine.")
    assert not next(row.hit for row in rows if row.vi == "paracetamol")


def test_negation_substring_not_a_match():
    rows = Glossary().check("tôi không thở được", "Known breathing problem.")
    assert not next(row.hit for row in rows if row.vi == "không")


def test_gold_egg_plural_and_boundary():
    slot = {"forms": ["egg"]}
    assert slot_ok(slot, "The patient is allergic to eggs.")
    assert not slot_ok(slot, "The patient likes eggplant.")
