"""Blind acoustic evidence and conservative ASR rejection, independent of tokens.

Top/bottom frame-energy contrast is only an SNR proxy: continuous speech, babble
and gain changes confound it. It cannot validate transcript truth. The default
uncalibrated policy never authorizes automatic speech in the full runtime.
"""
from collections import Counter
import numpy as np

from tonebridge.config import QualitySettings
from tonebridge.contracts import AcousticEvidence, GateAction

QUALITY_VERSION = "acoustic-evidence-v1"


def acoustic_evidence(wav: np.ndarray, sample_rate: int = 16000, *, text: str = "",
                      speech_duration_s: float | None = None, vad_ratio: float | None = None,
                      voicing_quality: float | None = None) -> AcousticEvidence:
    """Measure raw pre-denoise audio; no clean reference or transcript logging."""
    x = np.asarray(wav, dtype=np.float64)
    if sample_rate != 16000 or x.ndim != 1 or not x.size or not np.isfinite(x).all() or np.max(np.abs(x)) > 1:
        raise ValueError("invalid quality audio")
    n = int(.02 * sample_rate)
    frames = x[:len(x) // n * n].reshape(-1, n) if len(x) >= n else x[None, :]
    energy = np.sort(np.mean(frames ** 2, axis=1)) + 1e-12
    contrast = float(10 * np.log10(np.mean(energy[int(.7 * len(energy)):]) /
                                  np.mean(energy[:max(1, int(.2 * len(energy)))])))
    duration = len(x) / sample_rate
    words = text.lower().split()
    bigrams = list(zip(words, words[1:]))
    repeats = sum(count - 1 for count in Counter(bigrams).values()) / max(1, len(bigrams))
    return AcousticEvidence(duration_s=duration,
        rms_dbfs=max(-120., float(10 * np.log10(max(1e-12, np.mean(x ** 2))))),
        snr_proxy_db=max(0., contrast), clipping_fraction=float(np.mean(np.abs(x) >= .99)),
        vad_ratio=vad_ratio, voicing_quality=voicing_quality,
        words_per_second=len(words) / max(.02, speech_duration_s or duration),
        repeated_bigram_fraction=repeats)


def assess_quality(evidence: AcousticEvidence | None, cfg: QualitySettings,
                   *, required: bool = False) -> tuple[GateAction | None, list[str]]:
    """Reject weak evidence; missing/uncalibrated signals require confirmation."""
    if evidence is None:
        return (GateAction.REPEAT, ["acoustic_evidence_missing"]) if required else (None, [])
    reasons = []
    if evidence.rms_dbfs < cfg.min_rms_dbfs:
        reasons.append("acoustic_energy_low")
    if evidence.snr_proxy_db < cfg.min_snr_proxy_db:
        reasons.append("acoustic_snr_proxy_low")
    if evidence.clipping_fraction > cfg.max_clipping_fraction:
        reasons.append("acoustic_clipping")
    if evidence.vad_ratio is not None and evidence.vad_ratio < cfg.min_vad_ratio:
        reasons.append("acoustic_vad_ratio_low")
    if evidence.voicing_quality is not None and evidence.voicing_quality < cfg.min_voicing_quality:
        reasons.append("acoustic_voicing_low")
    if evidence.words_per_second > cfg.max_words_per_second:
        reasons.append("asr_text_rate_implausible")
    if evidence.repeated_bigram_fraction > cfg.max_repeated_bigram_fraction:
        reasons.append("asr_repetition_suspected")
    if reasons:
        return GateAction.REPEAT, reasons
    if evidence.vad_ratio is None or evidence.voicing_quality is None:
        return GateAction.CONFIRM, ["acoustic_context_incomplete"]
    if not cfg.calibrated:
        return GateAction.CONFIRM, ["acoustic_policy_uncalibrated"]
    return None, []
