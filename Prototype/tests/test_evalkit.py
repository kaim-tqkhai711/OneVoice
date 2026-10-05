from tonebridge.evalkit.metrics import ErrorCounts
from tonebridge.evalkit.textnorm import normalize_vi


def test_normalize_keeps_diacritics_drops_punct_case():
    assert normalize_vi("Bạn có bị ĐAU ngực không?") == "bạn có bị đau ngực không"
    assert normalize_vi("  a,  b. ") == "a b"


def test_wer_counts():
    c = ErrorCounts()
    c.add("tôi bị đau đầu", "tôi bị đau")  # 1 deletion of 4 words
    c.add("không dị ứng", "có dị ứng")  # 1 substitution of 3 words
    assert (c.words, c.word_errors) == (7, 2)
    assert abs(c.wer - 2 / 7) < 1e-9


def test_empty_hyp_counts_all_deleted():
    c = ErrorCounts(); c.add("a b c", "")
    assert c.word_errors == 3
