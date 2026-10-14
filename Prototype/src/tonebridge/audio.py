"""Strict 16-kHz mono audio boundary; no implicit downmixing or resampling."""
from __future__ import annotations

from pathlib import Path
from math import gcd
from scipy.signal import resample_poly

import numpy as np
from scipy.io import wavfile


class AudioInputError(ValueError):
    """A malformed or unsupported input that must never reach inference/TTS."""


def validate_audio(wav: np.ndarray, sample_rate: int, max_utterance_s: float) -> np.ndarray:
    """Validate an already-normalized [T] float buffer and return float32 mono."""
    x = np.asarray(wav)
    if sample_rate != 16000:
        raise AudioInputError("sample rate must be 16000 Hz")
    if x.ndim != 1:
        raise AudioInputError("audio must be mono (one-dimensional); downmix explicitly before input")
    if not x.size:
        raise AudioInputError("audio buffer is empty")
    if x.size > int(max_utterance_s * sample_rate):
        raise AudioInputError(f"audio exceeds max_utterance_s={max_utterance_s}")
    if x.dtype.kind != "f":
        raise AudioInputError("pipeline input must be normalized floating-point audio")
    if not np.isfinite(x).all():
        raise AudioInputError("audio contains NaN or infinity")
    if np.max(np.abs(x)) > 1.0:
        raise AudioInputError("audio amplitude is outside [-1, 1]")
    return np.ascontiguousarray(x, dtype=np.float32)


def load_wav(path: Path, expect_sr: int = 16000, max_utterance_s: float = 15.0,
             *, convert: bool = False) -> np.ndarray:
    """Load PCM uint8/int16/int32 (including left-aligned PCM24), or IEEE float.

    Unsigned PCM silence is 128. Signed PCM is divided by 2**(bits-1),
    in float64 first to avoid overflow/rounding at the int32 extremes.
    Invalid files fail at the boundary, before any model is loaded.
    """
    try:
        sr, data = wavfile.read(path)
    except (OSError, ValueError, EOFError) as exc:
        raise AudioInputError(f"cannot read WAV ({type(exc).__name__}): {path}") from exc
    if convert:
        if data.shape[0] > int(max_utterance_s * sr):
            raise AudioInputError("audio exceeds max_utterance_s before conversion")
        return validate_audio(prepare_audio(data, sr, expect_sr), expect_sr, max_utterance_s)
    if sr != expect_sr or expect_sr != 16000:
        raise AudioInputError(f"sample rate {sr} != required 16000 Hz")
    if data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    elif data.dtype.kind == "i" and data.dtype.itemsize in (2, 4):
        x = data.astype(np.float64) / float(2 ** (data.dtype.itemsize * 8 - 1))
    elif data.dtype.kind == "f":
        x = data
    else:
        raise AudioInputError(f"unsupported WAV dtype: {data.dtype}")
    return validate_audio(x, sr, max_utterance_s)


def prepare_audio(data: np.ndarray, sr: int, expect_sr: int = 16000) -> np.ndarray:
    if sr <= 0 or data.ndim not in (1, 2) or not data.size:
        raise ValueError("empty_or_invalid_wav")
    if data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128) / 128
    elif np.issubdtype(data.dtype, np.signedinteger):
        x = data.astype(np.float64) / float(2 ** (data.dtype.itemsize * 8 - 1))
    elif np.issubdtype(data.dtype, np.floating):
        x = data.astype(np.float32)
    else:
        raise ValueError("unsupported_wav_encoding")
    if not np.isfinite(x).all() or np.max(np.abs(x)) > 1.001:
        raise ValueError("nonfinite_or_out_of_range_audio")
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != expect_sr:
        divisor = gcd(sr, expect_sr)
        x = resample_poly(x, expect_sr // divisor, sr // divisor).astype(np.float32)
    return np.clip(x, -1, 1).astype(np.float32, copy=False)


def record_microphone(seconds: float, expect_sr: int = 16000) -> np.ndarray:
    """Explicit opt-in timed capture; raw audio stays in memory."""
    import sys
    import sounddevice as sd
    if not 0 < seconds <= 60:
        raise ValueError("record_seconds_out_of_range")
    info = sd.query_devices(kind="input")
    sr = int(info["default_samplerate"])
    print(f"RECORDING for {seconds:g} seconds", file=sys.stderr, flush=True)
    data = sd.rec(round(seconds * sr), samplerate=sr, channels=1, dtype="float32", blocking=True)
    print("RECORDING finished", file=sys.stderr, flush=True)
    return prepare_audio(data, sr, expect_sr)


def write_wav_new(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    """Publish a complete PCM16 WAV exclusively; failed writes never expose a partial final file."""
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            wavfile.write(output, sample_rate, np.rint(np.clip(audio, -1, 1) * 32767).astype(np.int16))
        os.link(temporary, path)  # fails if the final path exists; no overwrite race
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
