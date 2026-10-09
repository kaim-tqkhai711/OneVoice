"""Laptop audio boundary: local WAV decoding and one conversion to 16 kHz."""
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def load_wav(path: Path, expect_sr: int = 16000) -> np.ndarray:
    sr, data = wavfile.read(path)
    return prepare_audio(data, sr, expect_sr)


def prepare_audio(data: np.ndarray, sr: int, expect_sr: int = 16000) -> np.ndarray:
    if sr <= 0 or data.ndim not in (1, 2) or not data.size:
        raise ValueError("empty_or_invalid_wav")
    if data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128) / 128
    elif np.issubdtype(data.dtype, np.signedinteger):
        x = data.astype(np.float32) / float(2 ** (data.dtype.itemsize * 8 - 1))
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
