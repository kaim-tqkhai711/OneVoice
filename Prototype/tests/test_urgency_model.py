"""Artifact contract tests with tiny ONNX graphs, never claimed trained weights."""
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from tonebridge.branch_b import BranchB, FEATURE_NAMES
from tonebridge.config import BranchBSettings, PipelineConfig, sha256_file
from tonebridge.contracts import AsrResult, GateAction, SafetyReport, UrgencyResult
from tonebridge.gate import GateConfig, decide
from tonebridge.urgency_model import SCHEMA_VERSION, UrgencyArtifactError, UrgencyModel, softmax_high
from tonebridge.urgency_train import export_standardizer


@pytest.fixture
def artifacts(tmp_path):
    weights = np.zeros((21, 2), np.float32)
    weights[0, 1], weights[20, 1] = 1., 3.
    graph = helper.make_graph([helper.make_node("MatMul", ["z", "w"], ["logits"])], "contract_fixture",
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, [None, 21])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 2])],
        initializer=[numpy_helper.from_array(weights, "w")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    helper.set_model_props(model, {"tonebridge.schema_version": SCHEMA_VERSION,
        "tonebridge.feature_names": json.dumps(FEATURE_NAMES), "tonebridge.class_names": json.dumps(["LOW", "HIGH"])})
    mlp, scaler = tmp_path / "mlp.onnx", tmp_path / "standardizer.json"
    onnx.save(model, mlp)
    export_standardizer(str(scaler), str(mlp), np.arange(21), np.full(21, 2.))
    return mlp, scaler


def test_loader_standardizes_in_schema_order_then_softmax(artifacts):
    model = UrgencyModel(*artifacts)
    vector = np.arange(21, dtype=float)
    vector[0] += 4  # z0 = 2
    vector[20] -= 2  # z20 = -1 => HIGH logit -1, LOW logit 0
    features = {name: value for name, value in reversed(list(zip(FEATURE_NAMES, vector)))}
    assert model.predict_high(features) == pytest.approx(1 / (1 + np.exp(1)))
    assert not model.mean.flags.writeable and not model.scale.flags.writeable


@pytest.mark.parametrize("logits,expected", [([[0., 0.]], .5), ([[1000., 1001.]], 1 / (1 + np.exp(-1))), ([[-1000., 1000.]], 1.)])
def test_softmax_is_probability_and_stable(logits, expected):
    assert softmax_high(np.asarray(logits)) == pytest.approx(expected)


@pytest.mark.parametrize("logits", [np.array([1, 2]), np.zeros((2, 2)), np.array([[np.nan, 1]]), np.array([[1, np.inf]])])
def test_invalid_logits_fail_closed(logits):
    with pytest.raises(UrgencyArtifactError):
        softmax_high(logits)


@pytest.mark.parametrize("change", ["order", "classes", "shape", "zero_scale", "nan", "hash", "schema", "corrupt"])
def test_invalid_scaler_or_binding_rejected(artifacts, change):
    mlp, scaler = artifacts
    data = json.loads(scaler.read_text())
    if change == "order": data["feature_names"].reverse()
    elif change == "classes": data["class_names"].reverse()
    elif change == "shape": data["mean"] = [0]
    elif change == "zero_scale": data["scale"][0] = 0
    elif change == "nan": data["mean"][0] = float("nan")
    elif change == "hash": data["model_sha256"] = "0" * 64
    elif change == "schema": data["schema_version"] = "unexpected"
    scaler.write_text("{" if change == "corrupt" else json.dumps(data), encoding="utf8")
    with pytest.raises(UrgencyArtifactError):
        UrgencyModel(mlp, scaler)


def test_missing_configured_artifact_fails_load(artifacts):
    mlp, scaler = artifacts
    mlp.unlink()
    with pytest.raises(UrgencyArtifactError):
        UrgencyModel(mlp, scaler)


def test_onnx_metadata_mismatch_is_rejected_even_with_new_hash(artifacts):
    mlp, scaler = artifacts
    graph = onnx.load(mlp)
    helper.set_model_props(graph, {})
    onnx.save(graph, mlp)
    data = json.loads(scaler.read_text()); data["model_sha256"] = sha256_file(mlp)
    scaler.write_text(json.dumps(data))
    with pytest.raises(UrgencyArtifactError, match="metadata"):
        UrgencyModel(mlp, scaler)


def _branch(monkeypatch, cfg, root, feature_values=None):
    # Test boundaries avoid loading SwiftF0/band assets; classifier itself is ORT.
    import tonebridge.branch_b as bb
    monkeypatch.setattr(bb, "load_swiftf0", lambda *args: None)
    import onnxruntime as ort
    real_session = ort.InferenceSession
    monkeypatch.setattr(ort, "InferenceSession", lambda path, *args, **kwargs:
                        None if str(path) == "unused-band" else real_session(path, *args, **kwargs))
    branch = BranchB(cfg, band_model=Path("unused-band"), root=root)
    features = {name: 1. for name in FEATURE_NAMES}
    features.update(voiced_frac=.9, voicing_quality=.9)
    features.update(feature_values or {})
    monkeypatch.setattr(branch, "features", lambda wav: (features, 100))
    return branch


@pytest.mark.parametrize("paths", [(None, None), ("not-loaded.onnx", None), (None, "not-loaded.json")])
def test_null_artifacts_report_unconfigured_and_disable_cue(monkeypatch, tmp_path, paths):
    branch = _branch(monkeypatch, BranchBSettings(mlp_path=paths[0], standardizer=paths[1]), tmp_path)
    result = branch.analyze(np.full(16000, .1, np.float32))
    assert result.label == "UNCONFIGURED" and result.classifier_status == "UNCONFIGURED"
    assert not result.cue_enabled and result.score is None and result.reasons
    assert result.voicing_quality == .9  # evidence is still available for quality gate
    silence = branch.analyze(np.zeros(16000, np.float32))
    assert silence.label == "UNCONFIGURED" and "silence" in silence.reasons


def test_real_loader_branch_high_low_and_untuned_cue(monkeypatch, artifacts):
    mlp, scaler = artifacts
    cfg = BranchBSettings(mlp_path=mlp.name, standardizer=scaler.name)
    branch = _branch(monkeypatch, cfg, mlp.parent, {FEATURE_NAMES[0]: 20., FEATURE_NAMES[20]: 22.})
    result = branch.analyze(np.full(16000, .1, np.float32))
    assert result.label == "HIGH" and 0 <= result.score <= 1
    assert result.classifier_status == "READY" and not result.cue_enabled
    branch.cfg.tuned = True
    assert branch.analyze(np.full(16000, .1, np.float32)).cue_enabled
    branch.features = lambda wav: ({name: -100. for name in FEATURE_NAMES} | {"voiced_frac": .9, "voicing_quality": .9}, 100)
    assert branch.analyze(np.full(16000, .1, np.float32)).label == "LOW"


def test_missing_or_bad_features_cannot_infer(artifacts):
    classifier = UrgencyModel(*artifacts)
    features = {name: 1. for name in FEATURE_NAMES}
    features.pop(FEATURE_NAMES[0])
    with pytest.raises(UrgencyArtifactError): classifier.predict_high(features)
    features[FEATURE_NAMES[0]] = np.nan
    with pytest.raises(UrgencyArtifactError): classifier.predict_high(features)


@pytest.mark.parametrize("status", ["UNCONFIGURED", "DISABLED", "ERROR"])
def test_gate_cannot_emit_cue_for_unavailable_classifier(status):
    result = UrgencyResult(label="HIGH", score=.99, voicing_quality=.99, classifier_status=status, cue_enabled=True)
    decision = decide(AsrResult(text="xin chào", lang="vi", confidence=.99), SafetyReport(passed=True), result, "Hello", GateConfig())
    assert decision.action == GateAction.SPEAK


def test_config_fingerprints_artifact_paths_and_cue_tuning():
    cfg = PipelineConfig()
    before = cfg.config_hash()
    cfg.branch_b.mlp_path = "models/urgency/mlp.onnx"
    assert cfg.config_hash() != before
    before = cfg.config_hash(); cfg.branch_b.standardizer = "models/urgency/standardizer.json"
    assert cfg.config_hash() != before
    before = cfg.config_hash(); cfg.branch_b.tuned = True
    assert cfg.config_hash() != before


def test_configured_classifier_assets_must_be_pinned_and_snapshotted(artifacts, tmp_path):
    import shutil
    from tonebridge.config import ROOT
    from tonebridge.assets import selected_assets
    configs = tmp_path / "configs"; configs.mkdir()
    for name in ("pipeline.json", "runtime_files.json", "safety_lexicon_vi_en.json"):
        shutil.copyfile(ROOT / "configs" / name, configs / name)
    cfg = PipelineConfig.load(configs / "pipeline.json")
    mlp, scaler = artifacts
    cfg.branch_b.mlp_path, cfg.branch_b.standardizer = mlp.name, scaler.name
    with pytest.raises(ValueError, match="not pinned"):
        cfg.prepare("full", verify_assets=False)
    manifest = json.loads((configs / "runtime_files.json").read_text())
    for path in (mlp, scaler):
        manifest.append({"path": path.name, "reader": "urgency_model", "required": True,
                         "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    (configs / "runtime_files.json").write_text(json.dumps(manifest))
    effective = cfg.prepare("full", verify_assets=False)
    selected = selected_assets(effective, "full")
    assert {str(mlp), str(scaler)}.issubset({entry["path"] for entry in selected})
    assert effective._resources["urgency_schema"] == SCHEMA_VERSION
    assert len(selected) == 372  # two additional active assets; no hidden config reload
