"""Branch B equivalence tests against independent references (librosa / plain-python loops) with the tolerance table of
docs/BRANCH_B_DESIGN.md section 4. librosa is a TEST reference only."""
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import pytest

from tonebridge import branch_b_graph as G
from tonebridge.branch_b import FEATURE_NAMES, HOP_S, BranchB, BranchBConfig, ProvenanceError, compute_features

ROOT = Path(__file__).resolve().parents[1]
SR = 16000


@pytest.fixture(scope="module")
def band_session(tmp_path_factory):
    p = tmp_path_factory.mktemp("g") / "band_feats.onnx"
    G.build(str(p))
    return ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])


def _speechy(seed=0, dur=3.0):
    """Harmonic glide 120->220 Hz with an amplitude contour + a little noise (voiced), then a gap and a second voiced run."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    f0 = np.where(t < 1.4, 120 + 100 * t / 1.4, 160 + 20 * np.sin(2 * np.pi * (t - 1.8)))
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(h * ph) / h for h in range(1, 12)) * (0.1 + 0.05 * np.sin(2 * np.pi * 2 * t))
    x[int(1.4 * SR):int(1.8 * SR)] = 0.0005 * rng.standard_normal(int(0.4 * SR))
    return (x + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


def _ref_power(x):
    xp = np.concatenate([np.zeros(256), x]).astype(np.float64)
    return np.abs(librosa.stft(xp, n_fft=512, hop_length=256, window="hann", center=False)) ** 2  # [257, N]


def _ref_bands(x):
    S, M = _ref_power(x), G.band_masks().astype(np.float64)
    return np.concatenate([S.T @ M[:, :4], np.stack([(S.T * M[:, 4]).max(1), (S.T * M[:, 5]).max(1)], 1)], 1)


@pytest.mark.parametrize("n_extra", [0, 100, 255])
def test_band_graph_vs_librosa(band_session, n_extra):
    x = _speechy()[: 3 * SR + n_extra] if n_extra == 0 else np.concatenate([_speechy(), _speechy(1)[:n_extra]])
    out = band_session.run(None, {"audio": x[None, None, :]})[0]
    ref = _ref_bands(x)
    assert out.shape[0] == len(x) // 256 and out.shape[0] <= ref.shape[0]
    n = out.shape[0]
    m = ref[:n] > 1e-10
    assert np.max(np.abs(out[:n][m] - ref[:n][m]) / ref[:n][m]) <= 1e-4
    db_o, db_r = 10 * np.log10(out[:n] + 1e-20), 10 * np.log10(ref[:n] + 1e-20)
    loud = ref[:n, :4].sum(1) > 10 ** (-80 / 10)
    assert np.max(np.abs(db_o - db_r)[loud]) <= 0.01


def test_features_20_22_vs_full_reference(band_session):
    x = _speechy()
    from swift_f0 import SwiftF0
    r = SwiftF0(threads=1).detect(x, SR)
    bands = band_session.run(None, {"audio": x[None, None, :]})[0]
    n = min(len(r.confidence), len(bands))
    f = compute_features(r.pitch_hz[:n], r.confidence[:n], r.loudness_db[:n], bands[:n])
    S = _ref_power(x)[:, :n]
    fr = np.arange(257) * SR / 512
    v = r.confidence[:n] >= 0.5
    E = lambda lo, hi: S[(fr >= lo) & (fr < hi)].sum(0)[v]
    Pm = lambda lo, hi: S[(fr >= lo) & (fr < hi)].max(0)[v]
    assert abs(f["alpha_ratio_db"] - np.mean(10 * np.log10(E(50, 1000) / E(1000, 5000)))) <= 0.02
    assert abs(f["hammarberg_db"] - np.mean(10 * np.log10(Pm(0, 2000) / Pm(2000, 5000)))) <= 0.02
    assert abs(f["tilt_db"] - np.mean(10 * np.log10(E(0, 500) / E(500, 1500)))) <= 0.02


def _loop_reference(pitch, conf, loud):
    """Plain-python re-derivation of features 1-19 (no vectorised numpy shortcuts) from identical SwiftF0 outputs."""
    N = len(conf)
    voiced = [i for i in range(N) if conf[i] >= 0.5]
    f0 = sorted(pitch[i] for i in voiced)
    med = f0[len(f0) // 2] if len(f0) % 2 else 0.5 * (f0[len(f0) // 2 - 1] + f0[len(f0) // 2])
    st = {i: 12 * np.log2(pitch[i] / med) for i in voiced}
    vals = [st[i] for i in voiced]
    mean = sum(vals) / len(vals)
    std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
    def pct(a, q):
        a = sorted(a); k = (len(a) - 1) * q / 100; lo = int(np.floor(k)); hi = min(lo + 1, len(a) - 1)
        return a[lo] + (a[hi] - a[lo]) * (k - lo)
    rise, fall, dl = [], [], []
    for i in range(N - 1):
        if conf[i] >= 0.5 and conf[i + 1] >= 0.5:
            s = (st[i + 1] - st[i]) / HOP_S
            (rise if s > 0 else fall if s < 0 else []).append(s)
            d = (loud[i + 1] - loud[i]) / HOP_S
            if d > 0:
                dl.append(d)
    runs, i = [], 0
    while i < N:
        if conf[i] >= 0.5:
            j = i
            while j < N and conf[j] >= 0.5:
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    gaps = [runs[k + 1][0] - runs[k][1] for k in range(len(runs) - 1)]
    L = [loud[i] for i in voiced]
    lm = sum(L) / len(L)
    return {"f0_mean_st": mean, "f0_std_st": std, "f0_p20_st": pct(vals, 20), "f0_p80_st": pct(vals, 80), "f0_range_st": pct(vals, 80) - pct(vals, 20),
            "f0_slope_rise": float(np.mean(rise)) if rise else 0.0, "f0_slope_fall": float(np.mean(fall)) if fall else 0.0,
            "voiced_frac": len(voiced) / N, "voiced_seg_per_s": len(runs) / (N * HOP_S), "voiced_len_mean_s": float(np.mean([b - a for a, b in runs])) * HOP_S,
            "unvoiced_len_mean_s": float(np.mean(gaps)) * HOP_S if gaps else 0.0, "voicing_quality": sum(conf[i] for i in voiced) / len(voiced),
            "loud_mean": lm, "loud_std": (sum((x - lm) ** 2 for x in L) / len(L)) ** 0.5, "loud_p20": pct(L, 20), "loud_p80": pct(L, 80),
            "loud_range": pct(L, 80) - pct(L, 20), "loud_rise_slope": float(np.mean(dl)) if dl else 0.0}


def test_features_1_19_vs_loop_reference(band_session):
    x = _speechy(3)
    from swift_f0 import SwiftF0
    r = SwiftF0(threads=1).detect(x, SR)
    bands = band_session.run(None, {"audio": x[None, None, :]})[0]
    n = min(len(r.confidence), len(bands))
    f = compute_features(r.pitch_hz[:n], r.confidence[:n], r.loudness_db[:n], bands[:n])
    ref = _loop_reference(r.pitch_hz[:n], r.confidence[:n], r.loudness_db[:n])
    for k, v in ref.items():
        assert abs(f[k] - v) <= 1e-4 * max(1.0, abs(v)), (k, f[k], v)
    assert set(FEATURE_NAMES) == set(f)


def test_f0_semitone_is_relative_to_own_median():
    pitch = np.array([100.0, 110, 120, 130, 140]); conf = np.ones(5)
    f = compute_features(pitch, conf, np.full(5, -30.0), np.ones((5, 6)))
    st = 12 * np.log2(pitch / 120.0)
    assert abs(f["f0_mean_st"] - st.mean()) < 1e-9
    # scaling the whole utterance (a different speaker's register) leaves the statistics unchanged
    f2 = compute_features(pitch * 1.7, conf, np.full(5, -30.0), np.ones((5, 6)))
    assert abs(f2["f0_mean_st"] - f["f0_mean_st"]) < 1e-9 and abs(f2["f0_range_st"] - f["f0_range_st"]) < 1e-9


def test_swiftf0_sweep_median_error_under_1pct():
    from swift_f0 import SwiftF0
    t = np.arange(2 * SR) / SR
    f0 = 100 * (4.0 ** (t / 2.0))  # 100 -> 400 Hz exponential glide
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = (sum(np.sin(h * ph) / h for h in range(1, 9)) * 0.2).astype(np.float32)
    r = SwiftF0(threads=1).detect(x, SR)
    v = r.confidence >= 0.5
    ref = 100 * (4.0 ** (r.timestamps / 2.0))
    err = np.abs(r.pitch_hz[v] - ref[v]) / ref[v]
    assert v.mean() > 0.8 and np.median(err) <= 0.01


@pytest.fixture(scope="module")
def bb(tmp_path_factory):
    return BranchB(band_model=tmp_path_factory.mktemp("b") / "band_feats.onnx")


def test_unknown_rules_and_provenance(bb):
    assert bb.analyze(np.zeros(SR, np.float32)).label == "UNKNOWN"  # silence
    r = bb.analyze(_speechy()[: int(0.2 * SR)])
    assert r.label == "UNKNOWN" and "too_short" in r.reasons
    r = bb.analyze(_speechy())
    assert r.label == "UNKNOWN" and r.reasons == ["mlp_not_trained"] and r.voicing_quality and r.voicing_quality > 0.7  # features fine, model absent
    with pytest.raises(ProvenanceError):
        bb.analyze(_speechy(), provenance="denoised")


def test_gate_input_never_denoised_in_pipeline():
    """Pipeline tap: Branch B receives exactly the pre-denoiser segment (same array hash), even when the denoiser changes the audio."""
    from tonebridge.config import PipelineConfig
    from tonebridge.pipeline import Pipeline, Stages
    from tonebridge.stages import stubs

    seen = {}

    class Spy:
        def analyze(self, wav):
            seen["h"] = hashlib.sha256(np.ascontiguousarray(wav).tobytes()).hexdigest()
            from tonebridge.contracts import UrgencyResult
            return UrgencyResult(label="UNKNOWN")

    class Loud:
        def process(self, wav):
            return (wav * 0.5 + 0.01).astype(np.float32)

    cfg = PipelineConfig()
    st = Stages(frontend=stubs.PassthroughFrontEnd(), vad=stubs.WholeClipVad(), denoiser=Loud(), asr=stubs.SidecarAsr(), nmt=stubs.TagNmt(cfg.direction),
                safety=stubs.AlwaysPassSafety(), branch_b=Spy(), tts=stubs.ToneTts(cfg.tts_sample_rate))
    x = _speechy()
    Pipeline(cfg, st).run(x, "u1")
    assert seen["h"] == hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
