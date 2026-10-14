"""Branch B: raw acoustic evidence and an optional verified urgency classifier.

Absent configured weights/scaler -> UNCONFIGURED, cue disabled. Configured model
with weak acoustic evidence -> UNKNOWN with explicit reasons. Loaded, untuned
classifiers may produce LOW/HIGH probabilities but cannot enable an urgency cue.

Inference path = SwiftF0 ONNX (pitch/confidence/loudness per 256-sample hop) + ``band_feats.onnx`` (STFT band energies, built by
branch_b_graph.py) + numpy statistics + optional MLP ONNX. No librosa / torch / pyworld here (those are test references only).

Deviation from the design table, stated: F0 statistics are in semitones relative to the MEDIAN F0 OF THE SAME UTTERANCE (owner instruction,
2026-10-07), so feature 4 (f0_p50_st) would be identically 0 and is dropped: 21 features instead of 22. No feature was added.
UNKNOWN thresholds (0.3 s, 20 voiced frames, voiced_frac 0.15, voicing_quality 0.70) are STARTING VALUES, NOT TUNED (no team recordings yet).
Branch B must never see denoised audio: ``analyze`` raises ProvenanceError when the buffer is tagged "denoised".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tonebridge.contracts import UrgencyResult
from tonebridge.config import BranchBSettings

ROOT = Path(__file__).resolve().parents[2]
HOP, SR = 256, 16000
HOP_S = HOP / SR
FEATURE_NAMES = [
    "f0_mean_st", "f0_std_st", "f0_p20_st", "f0_p80_st", "f0_range_st", "f0_slope_rise", "f0_slope_fall",
    "voiced_frac", "voiced_seg_per_s", "voiced_len_mean_s", "unvoiced_len_mean_s", "voicing_quality",
    "loud_mean", "loud_std", "loud_p20", "loud_p80", "loud_range", "loud_rise_slope",
    "alpha_ratio_db", "hammarberg_db", "tilt_db",
]


class ProvenanceError(RuntimeError):
    """Raised when Branch B is handed audio that went through the Branch A denoiser."""


BranchBConfig = BranchBSettings


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of consecutive True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q, method="linear"))


def compute_features(pitch_hz: np.ndarray, conf: np.ndarray, loud_db: np.ndarray, bands: np.ndarray, voiced_conf: float = 0.5) -> dict[str, float]:
    """All inputs per 256-sample hop: pitch_hz [N], conf [N], loud_db [N], bands [N,6] (E50-1000, E1000-5000, E0-500, E500-1500, Pmax0-2000, Pmax2000-5000)."""
    N = len(conf)
    v = conf >= voiced_conf
    nv = int(v.sum())
    f: dict[str, float] = {k: 0.0 for k in FEATURE_NAMES}
    f["voiced_frac"] = nv / max(N, 1)
    if nv == 0:
        return f
    f0 = pitch_hz[v]
    st = 12.0 * np.log2(f0 / np.median(f0))  # semitones re the utterance's own median F0
    f["f0_mean_st"], f["f0_std_st"] = float(st.mean()), float(st.std())
    f["f0_p20_st"], f["f0_p80_st"] = _pct(st, 20), _pct(st, 80)
    f["f0_range_st"] = f["f0_p80_st"] - f["f0_p20_st"]
    both = v[1:] & v[:-1]  # consecutive voiced pairs
    st_all = np.zeros(N)
    st_all[v] = st
    slope = (st_all[1:] - st_all[:-1])[both] / HOP_S
    rise, fall = slope[slope > 0], slope[slope < 0]
    f["f0_slope_rise"] = float(rise.mean()) if len(rise) else 0.0
    f["f0_slope_fall"] = float(fall.mean()) if len(fall) else 0.0
    runs = _runs(v)
    f["voiced_seg_per_s"] = len(runs) / (N * HOP_S)
    f["voiced_len_mean_s"] = float(np.mean([e - s for s, e in runs])) * HOP_S
    gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
    f["unvoiced_len_mean_s"] = float(np.mean(gaps)) * HOP_S if gaps else 0.0
    f["voicing_quality"] = float(conf[v].mean())
    L = loud_db[v]
    f["loud_mean"], f["loud_std"] = float(L.mean()), float(L.std())
    f["loud_p20"], f["loud_p80"] = _pct(L, 20), _pct(L, 80)
    f["loud_range"] = f["loud_p80"] - f["loud_p20"]
    dl = (loud_db[1:] - loud_db[:-1])[both] / HOP_S
    pos = dl[dl > 0]
    f["loud_rise_slope"] = float(pos.mean()) if len(pos) else 0.0
    b = bands[v].astype(np.float64) + 1e-20
    f["alpha_ratio_db"] = float(np.mean(10 * np.log10(b[:, 0] / b[:, 1])))
    f["hammarberg_db"] = float(np.mean(10 * np.log10(b[:, 4] / b[:, 5])))
    f["tilt_db"] = float(np.mean(10 * np.log10(b[:, 2] / b[:, 3])))
    return f


def unknown_reasons(cfg: BranchBConfig, duration_s: float, peak: float, feats: dict[str, float], n_voiced: int) -> list[str]:
    r = []
    if peak < cfg.silence_peak:
        r.append("silence")
    if duration_s < cfg.min_duration_s:
        r.append("too_short")
    if n_voiced < cfg.min_voiced_frames:
        r.append("too_few_voiced_frames")
    if feats["voiced_frac"] < cfg.min_voiced_frac:
        r.append("voiced_frac_low")
    if feats["voicing_quality"] < cfg.min_voicing_quality:
        r.append("voicing_quality_low")
    return r


def load_swiftf0(model: Path, threads: int):
    """SwiftF0 detector whose ONNX session is built from OUR copy under models/ (byte-identical to the wheel's model.onnx, hash checked in
    tests/test_offline.py); the wheel's bundled file is never opened. Only ``SwiftF0.session`` is used by ``detect``."""
    import onnxruntime as ort
    from swift_f0 import SwiftF0

    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    det = SwiftF0.__new__(SwiftF0)  # skip __init__: it would open the copy inside site-packages
    det.session = ort.InferenceSession(str(model), o, providers=["CPUExecutionProvider"])
    return det


class BranchB:
    """Raw acoustic evidence plus an explicitly configured/offline urgency classifier."""

    def __init__(self, cfg: BranchBConfig | None = None, band_model: Path = ROOT / "models/branch_b/band_feats.onnx", threads: int = 1,
                 swiftf0_model: Path = ROOT / "models/branch_b/swiftf0_model.onnx",
                 root: Path = ROOT) -> None:
        import onnxruntime as ort

        self.cfg = cfg.model_copy(deep=True) if cfg is not None else BranchBConfig()
        self.classifier = None
        self.classifier_status = "UNCONFIGURED"
        self.configuration_reasons = []
        for name in ("mlp_path", "standardizer"):
            if getattr(self.cfg, name) is None:
                self.configuration_reasons.append(name + "_unconfigured")
        if not self.configuration_reasons:
            from tonebridge.urgency_model import UrgencyModel
            def resolve(path):
                value = Path(path)
                return value if value.is_absolute() else root / value
            self.classifier = UrgencyModel(resolve(self.cfg.mlp_path), resolve(self.cfg.standardizer), threads)
            self.classifier_status = "READY"
        self.f0 = load_swiftf0(swiftf0_model, threads)
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.log_severity_level = threads, 3
        self.band = ort.InferenceSession(str(band_model), so, providers=["CPUExecutionProvider"])
        self.last_features: dict[str, float] | None = None

    def features(self, wav: np.ndarray) -> tuple[dict[str, float], int]:
        x = np.asarray(wav, np.float32)
        r = self.f0.detect(x, SR)
        n = len(r.confidence)
        bands = self.band.run(None, {"audio": x[None, None, :]})[0]  # [N,6]
        m = min(n, len(bands))
        f = compute_features(r.pitch_hz[:m], r.confidence[:m], r.loudness_db[:m], bands[:m], self.cfg.voiced_conf)
        return f, int((r.confidence[:m] >= self.cfg.voiced_conf).sum())

    def analyze(self, wav: np.ndarray, provenance: str = "raw16k") -> UrgencyResult:  # [T_seg] -> urgency
        if provenance != "raw16k":
            raise ProvenanceError(f"Branch B accepts only raw16k audio, got {provenance!r}")
        x = np.asarray(wav, np.float32)
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        def unavailable(why, quality):
            return UrgencyResult(label="UNCONFIGURED" if self.classifier is None else "UNKNOWN",
                classifier_status=self.classifier_status, cue_enabled=False, voicing_quality=quality,
                reasons=self.configuration_reasons + why)
        if len(x) < HOP or peak < self.cfg.silence_peak:
            return unavailable(["silence" if peak < self.cfg.silence_peak else "too_short"], 0.0)
        f, nv = self.features(x)
        self.last_features = f
        why = unknown_reasons(self.cfg, len(x) / SR, peak, f, nv)
        if why:
            return unavailable(why, f["voicing_quality"])
        if self.classifier is None:
            return unavailable([], f["voicing_quality"])
        from tonebridge.urgency_model import UrgencyArtifactError
        try:
            p = self.classifier.predict_high(f)
        except (UrgencyArtifactError, RuntimeError):
            return UrgencyResult(label="UNKNOWN", classifier_status="ERROR", cue_enabled=False,
                voicing_quality=f["voicing_quality"], reasons=["urgency_inference_failed"])
        return UrgencyResult(label="HIGH" if p >= self.cfg.high_threshold else "LOW", score=p,
            classifier_status="READY", cue_enabled=self.cfg.tuned, voicing_quality=f["voicing_quality"],
            reasons=[] if self.cfg.tuned else ["urgency_threshold_untuned"])
