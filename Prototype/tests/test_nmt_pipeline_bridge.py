"""Integration with the unchanged VI-EN orchestrator using deterministic text.
No audio-model/translation-accuracy claims: this proves gate/TTS wiring only.
"""
import numpy as np
import pytest
from tonebridge.config import PipelineConfig
from tonebridge.contracts import AsrResult, GateAction
from tonebridge.pipeline import Pipeline, Stages
from tonebridge.gate import Gate
from tonebridge.stages import stubs
from tonebridge.nmt_evidence import EvidenceMtResult, TranslationEvidence
from tonebridge.safety import SemanticSafetyChecker


@pytest.mark.parametrize("target,incomplete,expected", [
    ("Take two tablets of paracetamol.", False, GateAction.SPEAK),
    ("Take five tablets of paracetamol.", False, GateAction.CONFIRM),
    ("Take two tablets of paracetamol.", True, GateAction.CONFIRM),
])
def test_evidence_bridge_controls_tts(target, incomplete, expected):
    class TextAsr:
        def transcribe(self, wav, lang):
            return AsrResult(text="uống hai viên paracetamol", lang=lang, confidence=.9)
    class TextNmt:
        def translate(self, text, src, tgt):
            return EvidenceMtResult(src_text=text, tgt_text=target, src_lang=src, tgt_lang=tgt,
                                    evidence=TranslationEvidence(eos_reached=not incomplete,
                                        truncated=incomplete, source_tokens=5, output_tokens=6))
    class CountTts(stubs.ToneTts):
        calls = 0
        def stream(self, text, lang):
            self.calls += 1
            yield from super().stream(text, lang)
    cfg = PipelineConfig()
    tts = CountTts(cfg.tts_sample_rate)
    stages = Stages(frontend=stubs.PassthroughFrontEnd(), vad=stubs.WholeClipVad(),
                    denoiser=stubs.PassthroughDenoiser(), asr=TextAsr(), nmt=TextNmt(),
                    safety=SemanticSafetyChecker(), branch_b=stubs.UnknownBranchB(), tts=tts, gate=Gate())
    result = Pipeline(cfg, stages).run(np.zeros(16000, np.float32), "test")
    assert result.record.gate.action == expected
    assert tts.calls == (1 if expected == GateAction.SPEAK else 0)
    assert bool(result.out_wav.size) == (expected == GateAction.SPEAK)
