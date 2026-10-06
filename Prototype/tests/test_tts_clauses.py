from tonebridge.stages.tts_piper import split_clauses


def test_split_first_clause():
    t = "Take two tablets of paracetamol, and call the doctor if the pain gets worse."
    p = split_clauses(t)
    assert p[0] == "Take two tablets of paracetamol," and " ".join(p) == t


def test_short_clauses_merge_and_single():
    assert split_clauses("Stop. Now.") == ["Stop. Now."]
    assert split_clauses("I can't breathe.") == ["I can't breathe."]
