import numpy as np

from tonebridge.config import PipelineConfig
from tonebridge.contracts import AsrResult, GateAction, MtResult, SafetyReport, UrgencyResult
from tonebridge.gate import Gate, GateConfig, decide
from tonebridge.pipeline import Pipeline, Stages
from tonebridge.safety import SemanticSafetyChecker
from tonebridge.stages import stubs

CFG = GateConfig()
OK = SafetyReport(passed=True)
LOW, UNK, HI = UrgencyResult(label="LOW"), UrgencyResult(label="UNKNOWN"), UrgencyResult(label="HIGH", score=0.9, voicing_quality=0.9)


def asr(conf=0.9, text="tôi bị đau ngực"):
    return AsrResult(text=text, lang="vi", confidence=conf)


def test_speak():
    d = decide(asr(), OK, LOW, "I have chest pain.", CFG)
    assert d.action == GateAction.SPEAK and d.speak_text


def test_speak_when_urgency_unknown_still_speaks():
    assert decide(asr(), OK, UNK, "I have chest pain.", CFG).action == GateAction.SPEAK


def test_speak_cue():
    d = decide(asr(), OK, HI, "I have chest pain.", CFG)
    assert d.action == GateAction.SPEAK_CUE and d.speak_text
    weak = UrgencyResult(label="HIGH", score=0.9, voicing_quality=0.3)  # weak voicing: no cue
    assert decide(asr(), OK, weak, "x y", CFG).action == GateAction.SPEAK


def test_repeat_low_asr_and_empty():
    assert decide(asr(conf=0.4), OK, LOW, "x", CFG).action == GateAction.REPEAT
    d = decide(asr(text=""), OK, LOW, "", CFG)
    assert d.action == GateAction.REPEAT and d.speak_text is None


def test_confirm_critical_and_severity():
    d = decide(asr(), SafetyReport(passed=False, reasons=["negation_missing"]), LOW, "You are allergic.", CFG)
    assert d.action == GateAction.CONFIRM and d.speak_text is None
    d = decide(asr(), SafetyReport(passed=True, confirm=True, reasons=["intensity_missing:rất"]), LOW, "High pressure.", CFG)
    assert d.action == GateAction.CONFIRM and d.speak_text is None


def test_abstain_cases():
    assert decide(asr(conf=0.1), OK, LOW, "x", CFG).action == GateAction.ABSTAIN  # floor
    bad = SafetyReport(passed=False, reasons=["medication_missing:ibuprofen"])
    d = decide(asr(conf=0.4), bad, LOW, "stop medrine", CFG)  # weak ASR AND failed safety
    assert d.action == GateAction.ABSTAIN and d.speak_text is None
    assert decide(asr(), OK, LOW, "", CFG).action == GateAction.ABSTAIN  # empty translation


def test_config_loads_from_json():
    c = GateConfig.load()
    assert 0 < c.asr_abstain_floor < c.asr_confidence_min < 1


class _Asr:
    def __init__(self, text, conf=0.9):
        self.t, self.c = text, conf

    def transcribe(self, wav, lang):
        return AsrResult(text=self.t, lang=lang, confidence=self.c)


class _Nmt:
    def __init__(self, out):
        self.out = out

    def translate(self, text, src, tgt):
        return MtResult(src_text=text, tgt_text=self.out, src_lang=src, tgt_lang=tgt, hops=["vi>en"])


class _CountTts(stubs.ToneTts):
    calls = 0

    def stream(self, text, lang):
        type(self).calls += 1
        yield from super().stream(text, lang)


def _run(src_text, mt_out, conf=0.9):
    cfg = PipelineConfig()
    tts = _CountTts(cfg.tts_sample_rate)
    st = Stages(frontend=stubs.PassthroughFrontEnd(), vad=stubs.WholeClipVad(), denoiser=stubs.PassthroughDenoiser(), asr=_Asr(src_text, conf),
                nmt=_Nmt(mt_out), safety=SemanticSafetyChecker(), branch_b=stubs.UnknownBranchB(), tts=tts, gate=Gate())
    return Pipeline(cfg, st).run((0.1 * np.sin(np.linspace(0, 300, 16000))).astype(np.float32), "u"), tts


def test_seeded_error_blocked_from_tts_end_to_end():
    _CountTts.calls = 0
    res, _ = _run("uống hai viên paracetamol mỗi sáu giờ", "Take five tablets of paracetamol every six hours.")  # dose number changed 2 -> 5
    assert res.record.gate.action == GateAction.CONFIRM and res.out_wav.size == 0 and _CountTts.calls == 0
    res, _ = _run("bạn không bị dị ứng với penicillin", "You are allergic to penicillin.")  # negation dropped
    assert res.record.gate.action == GateAction.CONFIRM and res.out_wav.size == 0 and _CountTts.calls == 0


def test_correct_translation_reaches_tts():
    _CountTts.calls = 0
    res, _ = _run("uống hai viên paracetamol mỗi sáu giờ", "Take two tablets of paracetamol every six hours.")
    assert res.record.gate.action in (GateAction.SPEAK, GateAction.SPEAK_CUE) and res.out_wav.size > 0 and _CountTts.calls == 1


def test_abstain_end_to_end():
    _CountTts.calls = 0
    res, _ = _run("uống hai viên paracetamol", "Take five tablets of paracetamol.", conf=0.3)  # weak ASR + failed safety
    assert res.record.gate.action == GateAction.ABSTAIN and res.out_wav.size == 0 and _CountTts.calls == 0
