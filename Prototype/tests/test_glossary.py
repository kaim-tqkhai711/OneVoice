from tonebridge.glossary import Glossary

G = Glossary()


def test_size_and_slots():
    assert len(G.terms) >= 60
    assert {t["slot"] for t in G.terms} == {"symptom", "allergy", "medication", "dose_unit", "negation"}


def test_longest_first_consumes_span():
    vi = [t["vi"] for t in G.find("Chúng tôi sẽ tiêm cho bạn năm miligam.")]
    assert "miligam" in vi and "gam" not in vi


def test_unit_miss_detected():  # real NMT error seen in D2: miligam -> millimetres
    r = {c.vi: c.hit for c in G.check("Chúng tôi sẽ tiêm cho bạn năm miligam.", "We'll give you five millimetres.")}
    assert r["miligam"] is False


def test_drug_name_miss_and_hit():
    assert not {c.vi: c.hit for c in G.check("Ngừng uống ibuprofen.", "Stop drinking medrine.")}["ibuprofen"]
    assert {c.vi: c.hit for c in G.check("Tôi bị dị ứng với penicillin.", "I'm allergic to penicillin.")}["penicillin"]
    assert {c.vi: c.hit for c in G.check("Tôi bị dị ứng với penicillin.", "I'm allergic to penicillin.")}["dị ứng"]


def test_negation_hit():
    r = {c.vi: c.hit for c in G.check("Tôi không thở được.", "I can't breathe.")}
    assert r["không"] and r["thở được"]
