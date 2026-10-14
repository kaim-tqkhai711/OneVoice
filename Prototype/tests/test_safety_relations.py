"""Audit section 5: counterexamples and independent safe controls."""
from decimal import Decimal

import numpy as np
import pytest

from tonebridge.contracts import AsrResult, GateAction, UrgencyResult
from tonebridge.gate import GateConfig, decide
from tonebridge.safety import SemanticSafetyChecker

C = SemanticSafetyChecker()
CASES = [
    ("dose_binding", "uống hai miligam aspirin và năm miligam paracetamol",
     "Take five milligrams of aspirin and two milligrams of paracetamol.",
     "Take two milligrams of aspirin and five milligrams of paracetamol."),
    ("negation_scope", "tôi không dị ứng penicillin nhưng dị ứng aspirin",
     "I am allergic to penicillin but not allergic to aspirin.",
     "I am not allergic to penicillin but allergic to aspirin."),
    ("decimal", "uống 0,5 miligam paracetamol", "Take 5 milligrams of paracetamol.",
     "Take 0.5 milligrams of paracetamol."),
    ("added_drug", "tôi uống paracetamol", "I take paracetamol and aspirin.", "I take paracetamol."),
    ("frequency", "uống hai viên paracetamol mỗi sáu giờ",
     "Take two tablets of paracetamol every two hours.", "Take two tablets of paracetamol every six hours."),
    ("modality", "tôi có thể dị ứng penicillin", "I am allergic to penicillin.", "I may be allergic to penicillin."),
    ("action", "ngừng uống aspirin", "Continue taking aspirin.", "Stop taking aspirin."),
    ("symptom", "tôi đau ngực", "I have a headache.", "I have chest pain."),
    ("oov_drug", "tôi dị ứng clopidogrel", "I am allergic to aspirin.", "I am allergic to clopidogrel."),
]


@pytest.mark.parametrize("name,source,bad,good", CASES, ids=[c[0] for c in CASES])
def test_audit_counterexamples_blocked(name, source, bad, good):
    report = C.check_texts(source, bad)
    assert not report.passed, report
    decision = decide(AsrResult(text=source, lang="vi", confidence=.99), report,
                      UrgencyResult(label="UNKNOWN"), bad, GateConfig())
    assert decision.action in (GateAction.CONFIRM, GateAction.ABSTAIN)
    assert decision.speak_text is None


@pytest.mark.parametrize("name,source,bad,good", CASES, ids=[c[0] for c in CASES])
def test_correct_counterparts_pass(name, source, bad, good):
    report = C.check_texts(source, good)
    assert report.passed and not report.confirm, report
    assert decide(AsrResult(text=source, lang="vi", confidence=.99), report,
                  UrgencyResult(label="UNKNOWN"), good, GateConfig()).action == GateAction.SPEAK


@pytest.mark.parametrize("value", ["0,5", "0.5", "0,50", "không phẩy năm"])
def test_decimal_exact_and_bound(value):
    parsed = C.relations.parse(f"uống {value} mg aspirin", "vi")
    assert not parsed.issues
    assert parsed.relations[0].amount == Decimal("0.5")
    assert parsed.relations[0].drug == "aspirin"
    assert len(parsed.relations[0].as_tuple()) == 8
    assert C.check_texts(f"uống {value} miligam aspirin", "Take 0.5 milligrams of aspirin.").passed


@pytest.mark.parametrize("source,target", [
    ("tôi chưa uống aspirin", "I have not taken aspirin yet."),
    ("tôi không dị ứng penicillin và aspirin", "I am not allergic to penicillin and aspirin."),
    ("uống 2 mg aspirin mỗi 6 giờ", "Take 2 mg aspirin every 360 minutes."),
    ("uống 2 mg aspirin và 5 mg paracetamol", "Take 5 mg paracetamol and 2 mg aspirin."),
    ("uống 0,5mg aspirin", "Take 0.5mg aspirin."),
    ("tôi đau ngực. bạn đang uống aspirin phải không", "I have chest pain. You are taking aspirin, right?"),
    ("không uống quá 25 miligam mỗi ngày", "Maximum 25 milligrams a day."),
])
def test_safe_variations(source, target):
    assert C.check_texts(source, target).passed


@pytest.mark.parametrize("source,target", [
    ("tôi chưa uống aspirin", "I did not take aspirin."),
    ("bạn uống aspirin", "I take aspirin."),
    ("tôi dị ứng clopidogrel", "I am allergic to prasugrel."),
    ("tôi uống aspirin khi cần", "I take aspirin."),
    ("tôi uống aspirin", "I take aspirin with unknownmed."),
    ("uống 0,5 mg aspirin", "Take 0.5 mg aspirin and 0.5 mg aspirin."),
    ("tôi uống aspirin", "I take aspirin except when fasting."),
    ("tôi uống aspirin", "I take aspirin and clopidogrel."),
    ("tôi đau bụng", "I have abdominal pain."),
    ("uống 0,5,5 mg aspirin", "Take 0.5 mg aspirin."),
    ("không uống aspirin cùng với rượu bia", "Do not take aspirin with water."),
    ("bệnh nhân không có tiền sử đau ngực", "The patient has no chest pain."),
    ("không dùng aspirin nếu bạn đang mang thai", "Do not use aspirin."),
    ("nhỏ 2 giọt vào mắt phải", "Put 2 drops in the left eye."),
    ("truyền 500 mililít amoxicillin trong 1 giờ", "Infuse 500 ml of amoxicillin over 2 hours."),
    ("tôi đau ngực. bạn có dị ứng aspirin không", "Do I have chest pain? You are allergic to aspirin."),
    ("tôi đau ngực", "I have chest pain and clopidogrel."),
    ("cơn sốt cao", "The fever is low."),
    ("uống hai năm mg aspirin", "Take 5 mg aspirin."),
    ("uống 3 mg aspirin", "Take one two mg aspirin."),
])
def test_unsupported_or_changed_critical_relation_fails_closed(source, target):
    assert not C.check_texts(source, target).passed


def test_existing_synthetic_references_do_not_false_alarm():
    """Regression over known templates, not an independent clinical evaluation."""
    import json
    from tonebridge.config import ROOT
    rows = [json.loads(line) for line in (ROOT / "configs/safety/safety_set_v1.jsonl").read_text(encoding="utf8").splitlines()]
    assert len(rows) == 1224
    for row in rows:
        report = C.check_texts(row["vi"], row["en_ref"])
        assert report.passed and not report.confirm, (row["id"], report.reasons)


@pytest.mark.parametrize("number", ["0,5", "0.5", "không phẩy năm", "nửa", "một nửa"])
def test_decoder_number_constraints_preserve_decimals(number):
    from tonebridge.nmt_constraints import GlossaryConstrainer
    forms = GlossaryConstrainer(groups=("number",)).forms(f"uống {number} miligam aspirin")
    assert forms == [["0.5"]]


@pytest.mark.parametrize("name,source,bad,good", CASES, ids=[c[0] for c in CASES])
def test_pipeline_never_calls_tts_for_counterexamples(name, source, bad, good):
    # Existing test doubles exercise the real safety and gate, no new runtime stub.
    from tonebridge.config import PipelineConfig
    from tonebridge.contracts import MtResult
    from tonebridge.pipeline import Pipeline
    from tonebridge.stages.factory import build_stub_stages
    from tonebridge.gate import Gate
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg)
    stages.asr.transcribe = lambda wav, lang: AsrResult(text=source, lang=lang, confidence=.99)
    stages.nmt.translate = lambda text, sl, tl: MtResult(src_text=text, tgt_text=bad, src_lang=sl, tgt_lang=tl)
    stages.safety = C
    stages.gate = Gate()
    def forbidden(*args, **kwargs):
        pytest.fail("TTS must not run for unsafe translation")
    stages.tts.stream = forbidden
    wav = (.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
    result = Pipeline(cfg, stages).run(wav, utt_id="audit")
    assert result.record.gate.action in (GateAction.CONFIRM, GateAction.ABSTAIN)
    assert result.out_wav.size == 0
