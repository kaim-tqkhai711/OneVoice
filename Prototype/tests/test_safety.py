import pytest

from tonebridge.safety import SemanticSafetyChecker

C = SemanticSafetyChecker()


@pytest.mark.parametrize("vi,n", [("hai mươi lăm", 25), ("năm mươi", 50), ("hai trăm năm mươi", 250), ("một trăm", 100), ("mười", 10), ("mười lăm", 15),
                                  ("năm", 5), ("năm trăm", 500), ("20", 20)])
def test_vi_numbers(vi, n):
    assert C._parse_vi_number(vi.split()) == n


def test_en_numbers():
    assert {25, 5}.issubset(C._en_numbers("take twenty-five and 5 tablets")) and 250 in C._en_numbers("two hundred fifty")


def test_known_d2_errors_flagged():
    r = C.check_texts("chúng tôi sẽ tiêm cho bạn năm miligam", "We will give you five millimetres.")
    assert not r.passed and any("dose_unit_missing" in x for x in r.reasons)
    r = C.check_texts("ngừng uống ibuprofen và uống nhiều nước hơn", "Stop drinking medrine and drink more water.")
    assert not r.passed and any("medication_missing" in x for x in r.reasons)


def test_negation_and_question_particle():
    assert not C.check_texts("bạn không bị dị ứng với penicillin", "You are allergic to penicillin.").passed
    assert C.check_texts("bạn không bị dị ứng với penicillin", "You are not allergic to penicillin.").passed
    assert C.check_texts("bạn có bị đau ngực không", "Do you have chest pain?").passed  # final "không" is a question particle


def test_correct_dose_passes_and_wrong_number_blocks():
    assert C.check_texts("uống hai viên paracetamol", "Take 2 tablets of paracetamol.").passed
    assert not C.check_texts("uống hai viên paracetamol", "Take 5 tablets of paracetamol.").passed


def test_intensity_confirm_not_block():
    r = C.check_texts("huyết áp rất cao", "High blood pressure.")
    assert r.passed and r.confirm and "intensity_missing:rất" in r.reasons
    assert not C.check_texts("huyết áp rất cao", "Very high blood pressure.").confirm


def test_en_number_with_and_and_plural_egg():
    assert 250 in C._en_numbers("two hundred and fifty milliliters")
    assert C.check_texts("bệnh nhân dị ứng với trứng", "The patient is allergic to eggs.").passed


def test_khong_qua_is_a_limit_not_an_intensifier():
    r = C.check_texts("không uống quá hai mươi lăm miligam mỗi ngày", "No more than 25 milligrams a day.")
    assert r.passed and not r.confirm
    assert C.check_texts("tôi thấy quá mệt", "I feel tired.").confirm  # a real intensifier still needs the English counterpart
