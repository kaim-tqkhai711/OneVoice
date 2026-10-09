from copy import deepcopy
import pytest
from tonebridge.clinical_safety import ClinicalSafetyChecker
from tonebridge.clinical_numbers import number
from tonebridge.nmt_evidence import EvidenceMtResult, TranslationEvidence
from tonebridge.contracts import MtResult


BAD = [
    ("uống hai viên paracetamol và một viên aspirin", "Take one tablet of paracetamol and two tablets of aspirin."),
    ("tôi không dị ứng với penicillin nhưng dị ứng với aspirin", "I am allergic to penicillin but not allergic to aspirin."),
    ("uống hai viên paracetamol mỗi sáu giờ", "Take two tablets of paracetamol every twelve hours."),
    ("uống paracetamol", "Take paracetamol and warfarin."),
    ("tôi bị đau ngực", "I have abdominal pain."),
    ("tiêm 0.5 miligam morphin", "Inject 5 mg morphine."),
    ("tiêm 5 mg morphin", "Inject 50 mg morphine."),
    ("ngừng uống ibuprofen", "Continue taking ibuprofen."),
    ("có thể dị ứng với penicillin", "I am allergic to penicillin."),
    ("không uống quá hai viên", "Do not take fewer than 2 tablets."),
    ("uống prednisolone", "Take prednisone."),
]
GOOD = [
    ("uống hai viên paracetamol và một viên aspirin", "Take two tablets of paracetamol and one tablet of aspirin."),
    ("tôi không dị ứng với penicillin nhưng dị ứng với aspirin", "I am not allergic to penicillin but allergic to aspirin."),
    ("uống hai viên paracetamol mỗi sáu giờ", "Take two tablets of paracetamol every six hours."),
    ("tiêm 0.5 miligam morphin", "Inject 0.5 mg morphine."),
    ("tiêm 5 mg morphin", "Inject 5 mg morphine."),
    ("ngừng uống ibuprofen", "Stop taking ibuprofen."),
    ("có thể dị ứng với penicillin", "Possibly allergic to penicillin."),
    ("không uống quá hai viên", "Do not take more than 2 tablets."),
    ("uống prednisolone", "Take prednisolone."),
    ("tôi bị đau ngực", "I have chest pain."),
    ("bệnh nhân dị ứng với trứng", "The patient is allergic to eggs."),
]


@pytest.mark.parametrize("src,tgt", BAD)
def test_critical_changes_never_pass(src, tgt):
    report = ClinicalSafetyChecker().check_texts(src, tgt)
    assert report.status == "FAIL"
    assert not report.passed


@pytest.mark.parametrize("src,tgt", GOOD)
def test_supported_correct_relations_pass(src, tgt):
    report = ClinicalSafetyChecker().check_texts(src, tgt)
    assert report.status == "PASS", report.reasons
    reverse = ClinicalSafetyChecker("en", "vi").check_texts(tgt, src)
    assert reverse.status == "PASS", reverse.reasons


@pytest.mark.parametrize("text,lang,expected", [
    ("0.5", "en", "0.5"), ("0,5", "vi", "0.5"), ("1/2", "en", "0.5"),
    ("half", "en", "0.5"), ("không phẩy năm", "vi", "0.5"),
    ("hai trăm năm mươi", "vi", "250"), ("two hundred and fifty", "en", "250"),
    ("두", "ko", "2"), ("십오", "ko", "15"), ("1/0", "en", None),
])
def test_exact_number_parser(text, lang, expected):
    value = number(text, lang)
    assert (str(value) if value is not None else None) == expected


def result(**overrides):
    evidence = dict(eos_reached=True, truncated=False, source_tokens=3, output_tokens=3)
    evidence.update(overrides)
    return EvidenceMtResult(src_text="xin chào", tgt_text="Hello", src_lang="vi", tgt_lang="en",
                            evidence=TranslationEvidence(**evidence))


@pytest.mark.parametrize("override", [
    {"truncated": True}, {"eos_reached": False}, {"input_truncated": True},
    {"source_unknown_tokens": 1}, {"output_unknown_tokens": 1},
    {"tokenizer_issues": ["bad_mapping"]},
    {"constraints_requested": 1, "constraints_satisfied": 0},
    {"unsatisfied_constraints": ["0"]},
])
def test_unreliable_translation_requires_confirmation(override):
    report = ClinicalSafetyChecker().check(result(**override))
    assert report.status == "CONFIRM" and report.confirm


def test_legacy_result_cannot_bypass_evidence():
    legacy = MtResult(src_text="xin chào", tgt_text="Hello", src_lang="vi", tgt_lang="en")
    assert ClinicalSafetyChecker().check(legacy).confirm


def test_unsupported_language_direction_and_unknown_critical_content():
    assert ClinicalSafetyChecker("vi", "ko").check_texts("xin chào", "안녕하세요").confirm
    assert ClinicalSafetyChecker().check_texts("uống thuốc xyz", "Take xyz medicine.").status != "PASS"
    assert ClinicalSafetyChecker().check_texts("tiêm 5 IU insulin", "Inject 50 IU insulin.").status != "PASS"
    assert ClinicalSafetyChecker().check_texts("nếu dị ứng thì ngừng uống aspirin", "Stop taking aspirin.").status != "PASS"


@pytest.mark.parametrize("src,tgt,a,b", [
    ("en", "ko", "Take 2 tablets of aspirin.", "아스피린 2정을 복용하세요."),
    ("ko", "en", "아스피린 2정을 복용하세요.", "Take 2 tablets of aspirin."),
])
def test_korean_requires_external_review(src, tgt, a, b):
    report = ClinicalSafetyChecker(src, tgt).check_texts(a, b)
    assert report.status == "CONFIRM"
    assert "korean_lexicon_pending_human_review" in report.reasons


def test_legacy_gate_compatibility():
    from tonebridge.gate import decide, GateConfig
    from tonebridge.contracts import AsrResult, UrgencyResult, GateAction
    for status in ("CONFIRM", "FAIL"):
        from tonebridge.nmt_evidence import DetailedSafetyReport
        report = DetailedSafetyReport.decision(status, "vi-en", ["test"])
        decision = decide(AsrResult(text="xin chào", lang="vi", confidence=.9), report,
                          UrgencyResult(label="UNKNOWN"), "Hello", GateConfig())
        assert decision.action == GateAction.CONFIRM and decision.speak_text is None


def test_actor_question_and_ambiguous_numbers_not_erased():
    checker = ClinicalSafetyChecker()
    assert checker.check_texts("tôi dị ứng với penicillin", "You are allergic to penicillin.").status == "FAIL"
    assert checker.check_texts("bạn có bị đau ngực không", "You have chest pain.").status == "FAIL"
    assert checker.check_texts("uống hai ba viên aspirin", "Take three tablets of aspirin.").status != "PASS"
    assert number("one two", "en") is None
    assert number("hai ba", "vi") is None
    assert number("không trăm linh năm", "vi") == 5


def test_legacy_ablation_cannot_disable_runtime_checker():
    from tonebridge.safety import SemanticSafetyChecker
    checker = SemanticSafetyChecker(enable={"dose": False})
    mt = EvidenceMtResult(src_text="uống hai viên paracetamol", tgt_text="Take five tablets of paracetamol.",
                          src_lang="vi", tgt_lang="en",
                          evidence=TranslationEvidence(eos_reached=True, truncated=False, source_tokens=5, output_tokens=6))
    assert checker.check(mt).status == "FAIL"
