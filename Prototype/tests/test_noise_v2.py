import numpy as np
import pytest

from tonebridge.evalkit.noise import measured_snr_masked_db, mix_at_snr_masked, rms_power, speech_mask
from tonebridge.evalkit.noisegen import make_alarm, make_babble

SR = 16000


def _speechlike(seed=0, dur=6.0, speech=(1.5, 3.0)):
    """Silence / loud burst / silence: whole-file power is far below speech power, so a whole-file SNR definition would be wrong by several dB."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(dur * SR), np.float32)
    a, b = int(speech[0] * SR), int(speech[1] * SR)
    x[a:b] = (0.3 * rng.standard_normal(b - a) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * np.arange(b - a) / SR) ** 2)).astype(np.float32)
    return x, speech_mask(len(x), [speech])


@pytest.mark.parametrize("snr", [10, 5, 0, -5])
@pytest.mark.parametrize("kind", ["white", "alarm", "babble"])
def test_masked_snr_within_half_db(snr, kind):
    clean, mask = _speechlike()
    rng = np.random.default_rng(3)
    noise = {"white": lambda: rng.standard_normal(SR * 30).astype(np.float32) * 0.1, "alarm": lambda: make_alarm(30, 1),
             "babble": lambda: make_babble([clean[int(1.5 * SR):int(3 * SR)]], 30, 4, 1)}[kind]()
    y = mix_at_snr_masked(clean, noise, snr, mask, np.random.default_rng(5))
    assert abs(measured_snr_masked_db(clean, y, mask) - snr) <= 0.5


def test_whole_file_definition_would_be_wrong():
    clean, mask = _speechlike()
    noise = np.random.default_rng(1).standard_normal(SR * 30).astype(np.float32)
    y = mix_at_snr_masked(clean, noise, 5.0, mask, np.random.default_rng(2))
    whole = 10 * np.log10(rms_power(clean) / rms_power(y - clean))
    assert whole < 5.0 - 3.0  # whole-file power underestimates speech SNR by > 3 dB when 75% of the clip is silence


def test_generators_deterministic():
    assert np.array_equal(make_alarm(10, 7), make_alarm(10, 7)) and not np.array_equal(make_alarm(10, 7), make_alarm(10, 8))
    assert np.max(np.abs(make_alarm(10, 7))) <= 0.5 + 1e-6
