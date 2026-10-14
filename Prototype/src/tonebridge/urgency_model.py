"""Strict offline loader for the 21-feature urgency MLP and population scaler.

Artifacts are a two-class logits ONNX graph and a JSON standardizer binding its
SHA-256, feature order and class order. Invalid configured artifacts fail fast;
no raw logits are ever treated as P(HIGH).
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

SCHEMA_VERSION = "tonebridge-urgency-v1"


class UrgencyArtifactError(ValueError):
    """Invalid weights, scaler, feature schema or model input/output contract."""


def softmax_high(logits: np.ndarray) -> float:
    """Stable two-class softmax; output order is strictly LOW, HIGH."""
    x = np.asarray(logits, dtype=np.float64)
    if x.shape != (1, 2) or not np.isfinite(x).all():
        raise UrgencyArtifactError("expected finite [1,2] logits")
    exp = np.exp(x[0] - np.max(x[0]))
    return float(exp[1] / exp.sum())


class UrgencyModel:
    """A verified, CPU-only inference session plus immutable standardizer."""

    def __init__(self, mlp_path: Path, standardizer_path: Path, threads: int = 1):
        import onnxruntime as ort
        from tonebridge.branch_b import FEATURE_NAMES
        from tonebridge.config import sha256_file

        try:
            data = json.loads(Path(standardizer_path).read_text(encoding="utf8"))
            if data.get("schema_version") != SCHEMA_VERSION or data.get("feature_names") != FEATURE_NAMES:
                raise UrgencyArtifactError("standardizer feature schema/order mismatch")
            if data.get("class_names") != ["LOW", "HIGH"]:
                raise UrgencyArtifactError("class order must be LOW,HIGH")
            if data.get("model_sha256") != sha256_file(Path(mlp_path)):
                raise UrgencyArtifactError("standardizer/model hash mismatch")
            self.mean = np.asarray(data["mean"], dtype=np.float64)
            self.scale = np.asarray(data["scale"], dtype=np.float64)
            if self.mean.shape != (21,) or self.scale.shape != (21,) or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
                raise UrgencyArtifactError("standardizer must contain 21 finite means and positive scales")
            self.mean.setflags(write=False)
            self.scale.setflags(write=False)
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            self.session = ort.InferenceSession(str(mlp_path), options, providers=["CPUExecutionProvider"])
            inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
            if (len(inputs) != 1 or inputs[0].name != "z" or inputs[0].type != "tensor(float)"
                    or len(inputs[0].shape) != 2 or inputs[0].shape[1] != 21
                    or (isinstance(inputs[0].shape[0], int) and inputs[0].shape[0] != 1)
                    or len(outputs) != 1 or outputs[0].name != "logits" or outputs[0].type != "tensor(float)"
                    or len(outputs[0].shape) != 2 or outputs[0].shape[1] != 2):
                raise UrgencyArtifactError("ONNX must map float z[B,21] to logits[B,2]")
            metadata = self.session.get_modelmeta().custom_metadata_map
            if (metadata.get("tonebridge.schema_version") != SCHEMA_VERSION
                    or json.loads(metadata.get("tonebridge.feature_names", "null")) != FEATURE_NAMES
                    or json.loads(metadata.get("tonebridge.class_names", "null")) != ["LOW", "HIGH"]):
                raise UrgencyArtifactError("ONNX metadata disagrees with standardizer schema")
        except UrgencyArtifactError:
            raise
        except Exception as exc:
            raise UrgencyArtifactError("cannot load configured urgency artifacts") from exc

    def predict_high(self, features: dict[str, float]) -> float:
        """Order by the schema, standardize once, infer logits and apply softmax."""
        from tonebridge.branch_b import FEATURE_NAMES
        if set(features) != set(FEATURE_NAMES):
            raise UrgencyArtifactError("feature set mismatch")
        vector = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)
        if not np.isfinite(vector).all():
            raise UrgencyArtifactError("nonfinite features")
        with np.errstate(over="ignore", invalid="ignore"):
            z = ((vector - self.mean) / self.scale).astype(np.float32)[None, :]
        if not np.isfinite(z).all():
            raise UrgencyArtifactError("nonfinite standardized features")
        return softmax_high(self.session.run(["logits"], {"z": z})[0])
