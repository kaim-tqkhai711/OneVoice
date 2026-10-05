from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tonebridge.evalkit.noise import measured_snr_db, mix_at_snr
from tonebridge.stages.denoise import GtcrnDenoiser, oa_mix

ROOT = Path(__file__).resolve().parents[1]
WAV = ROOT / "models/asr/zipformer-vi-int8/test_wavs/1.wav"
GTCRN = ROOT / "models/denoise/gtcrn_simple.onnx"


def lag(a, b):  # samples by which a is delayed vs b
    c = np.correlate(a, b, "full")
    return int(np.argmax(c) - (len(b) - 1))


def test_oa_endpoints_and_midpoint():
    n, e = np.array([1, 1, 1], np.float32), np.array([0, 2, 4], np.float32)
    assert np.allclose(oa_mix(n, e, 0.0), n) and np.allclose(oa_mix(n, e, 1.0), e)
    assert np.allclose(oa_mix(n, e, 0.5), [0.5, 1.5, 2.5])


def test_oa_rejects_misaligned_lengths():
    with pytest.raises(ValueError):
        oa_mix(np.zeros(10, np.float32), np.zeros(11, np.float32), 0.5)


@pytest.mark.parametrize("snr", [10, 5, 0])
def test_mix_hits_requested_snr(snr):
    rng = np.random.default_rng(0)
    clean = np.sin(2 * np.pi * 220 * np.arange(16000) / 16000).astype(np.float32) * 0.3
    noise = rng.standard_normal(48000).astype(np.float32) * 0.1
    noisy = mix_at_snr(clean, noise, snr, rng)
    assert abs(measured_snr_db(clean, noisy) - snr) < 0.05


@pytest.mark.skipif(not (GTCRN.exists() and WAV.exists()), reason="models not downloaded")
def test_gtcrn_output_is_sample_aligned_with_input():
    """OA mixes noisy and enhanced sample by sample: GTCRN must neither delay nor change length (checked on noisy speech)."""
    x, _ = sf.read(WAV, dtype="float32")
    rng = np.random.default_rng(1)
    noisy = mix_at_snr(x, rng.standard_normal(len(x)).astype(np.float32), 5, rng)
    y = GtcrnDenoiser().process(noisy)
    assert len(y) == len(noisy)
    assert lag(y, noisy) == 0
