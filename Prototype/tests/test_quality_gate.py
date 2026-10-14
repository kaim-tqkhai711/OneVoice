"""Acoustic rejection controls; synthetic fixtures are not ASR calibration."""
import numpy as np
import pytest
from pydantic import ValidationError

from tonebridge.config import PipelineConfig, QualitySettings
from tonebridge.contracts import AcousticEvidence, AsrResult, GateAction, SafetyReport, UrgencyResult
from tonebridge.gate import Gate, GateConfig, decide
from tonebridge.quality_gate import acoustic_evidence, assess_quality


def evidence(**overrides):
    return AcousticEvidence(duration_s=2, rms_dbfs=-20, snr_proxy_db=15, clipping_fraction=0,
        vad_ratio=.8, voicing_quality=.9, words_per_second=2, repeated_bigram_fraction=0,
        ).model_copy(update=overrides)


@pytest.mark.parametrize("field,value,reason", [
    ("rms_dbfs", -65, "acoustic_energy_low"),
    ("snr_proxy_db", 1, "acoustic_snr_proxy_low"),
    ("clipping_fraction", .3, "acoustic_clipping"),
    ("vad_ratio", .05, "acoustic_vad_ratio_low"),
    ("voicing_quality", .1, "acoustic_voicing_low"),
    ("words_per_second", 12, "asr_text_rate_implausible"),
    ("repeated_bigram_fraction", .8, "asr_repetition_suspected"),
])
def test_confident_hallucination_still_rejected(field, value, reason):
    asr = AsrResult(text="tôi đau ngực", lang="vi", confidence=.999, quality=evidence(**{field: value}))
    decision = decide(asr, SafetyReport(passed=True), UrgencyResult(label="UNKNOWN"), "I have chest pain.", GateConfig())
    assert decision.action == GateAction.REPEAT and reason in decision.reasons
    assert decision.speak_text is None


def test_missing_and_uncalibrated_evidence_never_grants_full_speak():
    cfg = PipelineConfig().prepare("full", verify_assets=False)
    gate = Gate()
    asr = AsrResult(text="xin chào", lang="vi", confidence=.99)
    args = (SafetyReport(passed=True), UrgencyResult(label="UNKNOWN"), "Hello", cfg)
    assert gate(asr, *args).action == GateAction.REPEAT
    asr.quality = evidence()
    assert gate(asr, *args).action == GateAction.CONFIRM
    asr.quality.voicing_quality = None
    assert "acoustic_context_incomplete" in gate(asr, *args).reasons


def test_calibrated_policy_and_fingerprint_use_effective_settings():
    cfg = QualitySettings(calibrated=True, calibration_id="unit-test-only")
    assert assess_quality(evidence(), cfg) == (None, [])
    pipeline = PipelineConfig(quality=cfg).prepare("full", verify_assets=False)
    original = pipeline.config_hash()
    pipeline.quality.min_snr_proxy_db = 20
    assert pipeline.config_hash() != original
    assert assess_quality(evidence(), pipeline.quality)[0] == GateAction.REPEAT
    with pytest.raises(ValidationError):
        QualitySettings(calibrated=True)


def test_energy_proxy_rejects_stationary_noise_and_degraded_speech():
    rng = np.random.default_rng(61)
    # Continuous stationary noise and a low-contrast 0 dB mixture; no measured
    # babble recall is inferred from these deterministic regression fixtures.
    noise = rng.normal(0, .04, 32000).astype(np.float32)
    time = np.arange(32000) / 16000
    voiced = (.04 * np.sqrt(2) * np.sin(2 * np.pi * 170 * time)).astype(np.float32)
    for wav in (noise, noise + voiced):
        acoustic = acoustic_evidence(wav, text="confident hallucination", vad_ratio=.9, voicing_quality=.8)
        assert acoustic.snr_proxy_db < 6
        assert assess_quality(acoustic, QualitySettings())[0] == GateAction.REPEAT


def test_clean_dynamic_fixture_has_finite_independent_evidence():
    time = np.arange(32000) / 16000
    envelope = np.tile(np.r_[np.ones(8000), np.full(8000, .01)], 2)
    wav = (.15 * envelope * np.sin(2 * np.pi * 170 * time)).astype(np.float32)
    acoustic = acoustic_evidence(wav, text="xin chào", vad_ratio=.5, voicing_quality=.9)
    assert acoustic.snr_proxy_db > 30
    assert acoustic.words_per_second == 1
    assert assess_quality(acoustic, QualitySettings())[0] == GateAction.CONFIRM


@pytest.mark.parametrize("data", [np.zeros(0), np.array([np.nan]), np.array([2.]), np.zeros((2, 2))])
def test_malformed_quality_audio_is_rejected(data):
    with pytest.raises(ValueError):
        acoustic_evidence(data)


def test_pipeline_uses_raw_audio_and_vad_coverage_before_tts(monkeypatch):
    from tonebridge.pipeline import Pipeline
    from tonebridge.stages.factory import build_stub_stages
    stages = build_stub_stages(PipelineConfig())
    # Replace only test boundaries to exercise full policy without model assets.
    stages.mode = "full"
    stages.effective_config._mode = "full"
    stages.gate = Gate()
    stages.vad.last_speech_ratio = .05
    monkeypatch.setattr(stages.vad, "segment", lambda wav: (0., .5))
    monkeypatch.setattr(stages.denoiser, "process", lambda wav: .1 * np.sin(np.arange(len(wav)) * .1).astype(np.float32))
    monkeypatch.setattr(stages.tts, "stream", lambda *args: pytest.fail("weak audio cannot call TTS"))
    wav = np.random.default_rng(2).normal(0, .03, 16000).astype(np.float32)
    result = Pipeline(PipelineConfig(), stages).run(wav, "quality")
    assert result.record.gate.action == GateAction.REPEAT and not result.out_wav.size
    assert "acoustic_vad_ratio_low" in result.record.gate.reasons
    assert "acoustic_snr_proxy_low" in result.record.gate.reasons
