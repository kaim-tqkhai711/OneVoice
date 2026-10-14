"""Runtime settings and fingerprints must describe the resources actually used."""
import json
import shutil

import numpy as np
import pytest

from tonebridge.assets import read_manifest, selected_assets, verify_files
from tonebridge.config import ROOT, PipelineConfig
from tonebridge.contracts import AsrResult, GateAction, SafetyReport, UrgencyResult
from tonebridge.gate import Gate
from tonebridge.pipeline import Pipeline
from tonebridge.stages.factory import build_stages, build_stub_stages, capabilities


@pytest.fixture
def config_tree(tmp_path):
    directory = tmp_path / "configs"
    directory.mkdir()
    for name in ("pipeline.json", "safety_lexicon_vi_en.json", "runtime_files.json"):
        shutil.copy2(ROOT / "configs" / name, directory / name)
    return directory / "pipeline.json"


def test_config_missing_malformed_unknown_and_unsupported_audio(config_tree):
    with pytest.raises(FileNotFoundError):
        PipelineConfig.load(config_tree.with_name("missing.json"))
    for data in ("{", '{"typo_threshold": 0.4}', '{"sample_rate":44100}', '{"max_utterance_s":0}',
                 '{"thresholds":{"asr_confidence_min":0.1,"asr_abstain_floor":0.2}}'):
        config_tree.write_text(data, encoding="utf8")
        with pytest.raises(ValueError):
            PipelineConfig.load(config_tree)


def test_pipeline_thresholds_override_gate_and_fingerprint_changes():
    first = PipelineConfig()
    second = first.model_copy(deep=True)
    second.thresholds.asr_confidence_min = .99
    evidence = (AsrResult(text="tôi đau ngực", lang="vi", confidence=.9), SafetyReport(passed=True),
                UrgencyResult(label="UNKNOWN"), "I have chest pain.")
    gate = Gate()
    assert gate(*evidence, first).action == GateAction.SPEAK
    assert gate(*evidence, second).action == GateAction.REPEAT
    assert first.config_hash() != second.config_hash()


def test_child_configs_are_not_read(config_tree):
    (config_tree.parent / "gate.json").write_text("not json")
    (config_tree.parent / "branch_b.json").write_text("not json")
    cfg = PipelineConfig.load(config_tree).prepare("full", verify_assets=False)
    assert cfg.thresholds.voicing_quality_min == .7
    assert cfg.branch_b.min_voiced_frames == 20


@pytest.mark.parametrize("field,value", [("glossary_bonus", 6), ("max_new_tokens", 30),
                                       ("eos_penalty", 2), ("threads", 3), ("asr_active_paths", 8)])
def test_decode_settings_fingerprinted(field, value):
    cfg = PipelineConfig()
    other = cfg.model_copy(deep=True)
    setattr(other.decode, field, value)
    assert cfg.prepare("stub").config_hash() != other.prepare("stub").config_hash()


def test_lexicon_bytes_and_version_are_snapshotted(config_tree):
    cfg = PipelineConfig.load(config_tree)
    first = cfg.prepare("full", verify_assets=False)
    lexicon = config_tree.parent / "safety_lexicon_vi_en.json"
    data = json.loads(lexicon.read_text(encoding="utf8"))
    data["version"] = "new-version"
    lexicon.write_text(json.dumps(data), encoding="utf8")
    second = cfg.prepare("full", verify_assets=False)
    assert first.config_hash() != second.config_hash()
    assert first._lexicon_data["version"] != second._lexicon_data["version"]


def test_invalid_lexicon_fails_preflight(config_tree):
    cfg = PipelineConfig.load(config_tree)
    lexicon = config_tree.parent / "safety_lexicon_vi_en.json"
    for data in ({}, {"version": "broken"}, []):
        lexicon.write_text(json.dumps(data), encoding="utf8")
        with pytest.raises(ValueError, match="lexicon"):
            cfg.prepare("stub")


def test_manifest_hash_and_mode_change_fingerprint(config_tree):
    cfg = PipelineConfig.load(config_tree)
    full = cfg.prepare("full", verify_assets=False)
    bench = cfg.prepare("branch-a-bench", verify_assets=False)
    assert full.config_hash() != bench.config_hash()
    manifest = config_tree.parent / "runtime_files.json"
    data = json.loads(manifest.read_text(encoding="utf8"))
    data[0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data), encoding="utf8")
    assert full.config_hash() != cfg.prepare("full", verify_assets=False).config_hash()


def test_asset_selection_respects_modes_and_missing_assets(config_tree):
    cfg = PipelineConfig.load(config_tree)
    full = selected_assets(cfg, "full")
    bench = selected_assets(cfg, "branch-a-bench")
    assert len(full) == 370 and len(bench) == 368
    assert selected_assets(cfg, "stub") == []
    cfg.denoise_mode = "on"
    assert len(selected_assets(cfg, "full")) == 371
    with pytest.raises(ValueError, match="missing asset"):
        build_stages(cfg, "full")


def test_unused_pack_hash_is_not_a_full_mode_prerequisite(config_tree):
    cfg = PipelineConfig.load(config_tree)
    cfg.models["denoise"].sha256 = "0" * 64
    cfg.prepare("full", verify_assets=False)
    cfg.denoise_mode = "on"
    with pytest.raises(ValueError, match="sha256 does not match"):
        cfg.prepare("full", verify_assets=False)


def test_manifest_duplicate_escape_and_invalid_hash_fail(config_tree):
    cfg = PipelineConfig.load(config_tree)
    manifest = config_tree.parent / "runtime_files.json"
    original = json.loads(manifest.read_text(encoding="utf8"))
    cases = [original + [original[0]], [{**original[0], "path": "../../escape.onnx"}],
             [{**original[0], "sha256": "not-a-hash"}]]
    for rows in cases:
        manifest.write_text(json.dumps(rows), encoding="utf8")
        with pytest.raises(ValueError):
            read_manifest(cfg)


def test_size_and_hash_mismatch_detected(tmp_path):
    path = tmp_path / "asset.onnx"
    path.write_bytes(b"abc")
    entry = {"path": str(path), "bytes": 3, "sha256": "0" * 64}
    assert "sha256 mismatch" in verify_files([entry])[0]
    entry["bytes"] = 4
    assert "size mismatch" in verify_files([entry])[0]


def test_model_path_is_applied_and_unpinned_path_fails(config_tree):
    cfg = PipelineConfig.load(config_tree)
    first = cfg.config_hash()
    cfg.models["asr_vi"].path = "models/asr/other"
    assert cfg.model_path("asr_vi") == config_tree.parent.parent / "models/asr/other"
    assert cfg.config_hash() != first
    with pytest.raises(ValueError, match="not pinned"):
        cfg.prepare("full", verify_assets=False)


def test_pipeline_uses_copy_of_effective_config_and_detects_internal_mutation():
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg)
    pipeline = Pipeline(cfg, stages)
    cfg.thresholds.asr_confidence_min = .99
    assert pipeline.cfg.thresholds.asr_confidence_min == .5
    pipeline.cfg.thresholds.asr_confidence_min = .99
    result = pipeline.run(np.ones(16000, np.float32) * .1, "changed")
    assert result.record.error_type == "ChangedAfterInitialization"
    assert not result.out_wav.size


def test_full_capabilities_do_not_claim_trained_urgency():
    cfg = PipelineConfig()
    assert "MLP not trained" in capabilities(cfg, "full")["branch_b"]
    assert "stub" in capabilities(cfg, "branch-a-bench")["safety"]


def test_settings_changed_inside_gate_cannot_reach_tts(monkeypatch):
    cfg = PipelineConfig()
    stages = build_stub_stages(cfg)
    pipeline = Pipeline(cfg, stages)
    original = stages.gate

    def changed_gate(*args):
        pipeline.cfg.decode.glossary_bonus = 6
        return original(*args)

    stages.gate = changed_gate
    monkeypatch.setattr(stages.tts, "stream", lambda *args: pytest.fail("changed config reached TTS"))
    result = pipeline.run(np.ones(16000, np.float32) * .1, "changed-in-gate")
    assert result.record.error_type == "ChangedDuringTurn" and not result.out_wav.size


def test_high_urgency_requires_score_and_quality():
    cfg = PipelineConfig()
    asr = AsrResult(text="tôi đau ngực", lang="vi", confidence=.95)
    for urgency in (UrgencyResult(label="HIGH"), UrgencyResult(label="HIGH", score=.1, voicing_quality=.9)):
        assert Gate()(asr, SafetyReport(passed=True), urgency, "I have chest pain", cfg).action == GateAction.SPEAK
