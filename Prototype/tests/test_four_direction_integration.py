import json
import numpy as np
import pytest

from tonebridge.cli import build_stub_stages, public_row
from tonebridge.clinical_safety import ClinicalSafetyChecker
from tonebridge.config import PipelineConfig
from tonebridge.contracts import AsrResult, GateAction, GateDecision
from tonebridge.gate import Gate
from tonebridge.nmt_evidence import EvidenceMtResult, TranslationEvidence
from tonebridge.pipeline import Pipeline


@pytest.mark.parametrize("direction,source,target,action", [
    ("vi-en", "xin chào", "Hello.", GateAction.SPEAK),
    ("en-vi", "Hello.", "xin chào", GateAction.SPEAK),
    ("en-ko", "Hello.", "안녕하세요.", GateAction.CONFIRM),
    ("ko-en", "안녕하세요.", "Hello.", GateAction.CONFIRM),
])
def test_real_checker_four_direction_bridge(direction, source, target, action):
    cfg = PipelineConfig(direction=direction, urgency_enabled=False)
    stages = build_stub_stages(cfg, None)
    src, tgt = direction.split("-")
    stages.asr.transcribe = lambda *args: AsrResult(text=source, lang=src, confidence=.95)
    stages.nmt.translate = lambda *args: EvidenceMtResult(src_text=source, tgt_text=target, src_lang=src, tgt_lang=tgt,
         evidence=TranslationEvidence(eos_reached=True, truncated=False, source_tokens=2, output_tokens=2))
    stages.safety = ClinicalSafetyChecker(src, tgt)
    stages.gate = Gate()
    result = Pipeline(cfg, stages).run(np.full(16000, .1, np.float32), "bridge")
    assert result.record.gate.action == action
    assert bool(result.out_wav.size) == (action == GateAction.SPEAK)
    assert result.record.nmt_evidence["eos_reached"]
    assert result.record.safety_status == ("PASS" if action == GateAction.SPEAK else "CONFIRM")


@pytest.mark.parametrize("fields", [dict(truncated=True), dict(source_unknown_tokens=1),
    dict(output_unknown_tokens=1), dict(tokenizer_issues=["mapping:private"]),
    dict(constraints_requested=1), dict(constraints_satisfied=1)])
def test_rich_evidence_cannot_be_bypassed_by_custom_checker_or_gate(fields):
    cfg = PipelineConfig(direction="en-vi")
    stages = build_stub_stages(cfg, None)
    def translate(text, src, tgt):
        ev = TranslationEvidence.model_validate(dict(eos_reached=True, truncated=False, source_tokens=2, output_tokens=2) | fields)
        return EvidenceMtResult(src_text=text, tgt_text="Xin chào.", src_lang=src, tgt_lang=tgt, evidence=ev)
    stages.nmt.translate = translate
    stages.gate = lambda *args: GateDecision(action=GateAction.SPEAK, speak_text="Xin chào.")
    result = Pipeline(cfg, stages).run(np.full(16000, .1, np.float32), "unsafe")
    assert result.record.gate.action == GateAction.CONFIRM and not result.out_wav.size
    assert result.record.safety_status == "FAIL"


def test_checker_diagnostics_do_not_leak_content_to_default_logs():
    cfg = PipelineConfig(direction="en-ko")
    result = Pipeline(cfg, build_stub_stages(cfg, None)).run(np.full(16000, .1, np.float32), "privacy")
    result.record.safety_reasons = ["unparsed_content:private medical sentence"]
    result.record.gate.reasons = list(result.record.safety_reasons)
    result.record.nmt_evidence = TranslationEvidence(eos_reached=True, truncated=False, source_tokens=2, output_tokens=2,
       unsatisfied_constraints=["private drug"], tokenizer_issues=["probe_contains_source_unk:private medical sentence"]).model_dump()
    assert "private" not in json.dumps(public_row(result.record))
    assert "private" in json.dumps(public_row(result.record, True))


def test_source_preparation_preserves_medication_numbers_and_existing_questions():
    from tonebridge.nmt_source import prepare_source
    source = "do not take 2.5 mg of ABC"
    assert prepare_source(source, "en", True)[0] == "Do not take 2.5 mg of ABC."
    assert prepare_source("Are you allergic?", "en", True) == ("Are you allergic?", [])
    assert prepare_source("xin chào", "vi", True) == ("xin chào", [])
    assert prepare_source(source, "en", False) == (source, [])


def test_branch_thread_state_is_reused_and_closed_pipeline_is_safe():
    import threading
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg, None)
    local = threading.local()
    initialized = []
    def transcribe(wav, lang):
        if not getattr(local, "initialized", False):
            initialized.append(threading.current_thread())
            local.initialized = True
        return AsrResult(text="hello there", lang=lang, confidence=.9)
    stages.asr.transcribe = transcribe
    pipe = Pipeline(cfg, stages)
    signal = np.full(16000, .1, np.float32)
    for i in range(10):
        assert pipe.run(signal, str(i)).record.status == "ok"
    assert len(initialized) <= 2  # bounded branch workers, not a new thread per turn
    pipe.close()
    result = pipe.run(signal, "closed")
    assert result.record.status == "error" and not result.out_wav.size


@pytest.mark.parametrize("backend", ["marian", "argos"])
def test_pretrained_english_asr_formatting_and_real_budget_stop(backend):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    path = root / ("models/nmt-laptop/en-vi" if backend == "marian" else "models/nmt-laptop/en-ko-argos")
    if not (path / "asset_manifest.json").exists():
        pytest.skip("local pretrained NMT assets absent")
    if backend == "marian":
        from tonebridge.stages.nmt_marian import MarianTextAdapter
        nmt = MarianTextAdapter(path, "en", "vi", allow_partial_vocabulary=True, format_asr_source=True)
        target = "vi"
    else:
        from tonebridge.stages.nmt_argos import ArgosEnKoNmt
        nmt = ArgosEnKoNmt(path, format_asr_source=True)
        target = "ko"
    result = nmt.translate("hello", "en", target)
    assert result.src_text == "hello" and result.evidence.eos_reached
    assert result.evidence.source_preparation == ["english_initial_case", "english_terminal_period"]
    assert not result.evidence.source_unknown_tokens
    assert result.tgt_text in ("Xin chào.", "안녕하세요.")
    if backend == "marian":
        assert nmt.translate("|", "en", target).evidence.source_unknown_tokens
    nmt.max_new_tokens = 1
    limited = nmt.translate("hello", "en", target)
    assert limited.evidence.truncated and not limited.evidence.eos_reached
