"""Explicit live audio sink; optional sounddevice/PortAudio dependency.

Run inside the CLI's disposable inference process so device hangs share its
watchdog. Submission/driver latency is not a microphone-measured audible onset.
Already played, fully approved chunks cannot be retracted after a later fault.
"""
from __future__ import annotations

import numpy as np


class StreamingPlayer:
    """Lazy mono output stream, opened only for the first gate-approved chunk."""
    def __init__(self):
        self.stream = None
        self.sample_rate = None
        self.underflows = 0

    def write(self, chunk: np.ndarray, sample_rate: int) -> None:
        import sounddevice as sd
        if self.stream is None:
            self.sample_rate = sample_rate
            self.stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32", latency="low")
            self.stream.start()
        if sample_rate != self.sample_rate:
            raise ValueError("playback sample rate changed during turn")
        self.underflows += int(self.stream.write(np.ascontiguousarray(chunk[:, None], dtype=np.float32)))

    def close(self, *, abort: bool = False) -> None:
        """Drain completed turns; abort queued output on failure, then close."""
        if self.stream is not None:
            try:
                self.stream.abort() if abort else self.stream.stop()
            finally:
                self.stream.close()
                self.stream = None
