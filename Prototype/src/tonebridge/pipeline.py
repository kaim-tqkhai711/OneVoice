"""Bounded dual-branch turns with fail-closed output and explicit error records.

The library returns promptly on a worker timeout and retires that Pipeline.
Python cannot cancel native inference in a thread: CLI real modes additionally
run in a disposable child process (runtime.py) for hard cancellation.
"""
from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from threading import Lock, Thread
from queue import Queue
import weakref
import time
from typing import Callable

import numpy as np

from tonebridge.audio import AudioInputError, validate_audio
from tonebridge.config import PipelineConfig
from tonebridge.contracts import (AsrResult, GateAction, GateDecision, Lang, MtResult, SafetyReport,
                                  TurnRecord, UrgencyResult, Utterance)
from tonebridge.stages.base import Asr, BranchB, Denoiser, FrontEnd, Nmt, SafetyChecker, Tts, Vad
from tonebridge.stages.stubs import v0_gate
from tonebridge.telemetry import TurnClock, peak_rss_mb, rtf
from tonebridge.quality_gate import acoustic_evidence


@dataclass
class Stages:
    frontend: FrontEnd
    vad: Vad
    denoiser: Denoiser
    asr: Asr
    nmt: Nmt
    safety: SafetyChecker
    branch_b: BranchB
    tts: Tts
    gate: Callable[..., GateDecision] = v0_gate
    mode: str = "custom"
    capabilities: dict[str, str] = field(default_factory=dict)
    effective_config: PipelineConfig | None = None


@dataclass
class TurnResult:
    record: TurnRecord
    out_wav: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))


class StageFailure(RuntimeError):
    """Non-content failure evidence; do not leak model input in exception logs."""

    def __init__(self, stage: str, error_type: str, *, timeout: bool = False):
        self.stage, self.error_type, self.timeout = stage, error_type, timeout
        super().__init__(f"{stage}:{error_type}")


def _submit(function, *args) -> Future:
    """Use daemon workers so an abandoned native call cannot block interpreter exit."""
    future = Future()

    def invoke():
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)

    Thread(target=invoke, daemon=True, name="tonebridge-stage").start()
    return future


class _DaemonPool:
    """Reuse native thread-local workspaces without blocking exit on a hang."""
    def __init__(self, count: int):
        self.jobs = Queue()
        self.closed = False
        for _ in range(count):
            Thread(target=self._worker, daemon=True, name="tonebridge-stage").start()

    def _worker(self):
        while True:
            task = self.jobs.get()
            if task is None:
                return
            future, function, args = task
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        future.set_result(function(*args))
                    except BaseException as exc:
                        future.set_exception(exc)
            finally:
                del task, future, function, args  # do not retain a completed Pipeline

    def submit(self, function, *args):
        if self.closed:
            raise RuntimeError("pipeline pool closed")
        future = Future()
        self.jobs.put((future, function, args))
        return future

    def shutdown(self):
        if not self.closed:
            self.closed = True
            # Two sentinels suffice for either the serial or branch pool.
            self.jobs.put(None)
            self.jobs.put(None)


def _close_pools(*pools):
    for pool in pools:
        pool.shutdown()


def failure_result(cfg: PipelineConfig, utt_id: str, session_id: str, audio_s: float,
                   stage: str, error_type: str, *, status: str = "error", elapsed_ms: float = 0,
                   mode: str = "custom", capabilities: dict | None = None) -> TurnResult:
    """A failure never contains approved text/audio, including process failures."""
    return TurnResult(TurnRecord(
        session_id=session_id, utterance_id=utt_id, config_hash=cfg.config_hash(), direction=cfg.direction,
        audio_s=audio_s, stage_ms={}, endpoint_to_first_audio_ms=None, total_ms=elapsed_ms,
        rtf=rtf(elapsed_ms, audio_s), peak_rss_mb=peak_rss_mb(),
        gate=GateDecision(action=GateAction.ABSTAIN, reasons=[f"{stage}:{error_type}"]),
        asr_text="", nmt_text="", mode=mode, capabilities=capabilities or {}, status=status,
        error_stage=stage, error_type=error_type))


class Pipeline:
    def __init__(self, cfg: PipelineConfig, stages: Stages) -> None:
        self.cfg = (stages.effective_config or cfg).model_copy(deep=True)
        self.s = stages
        self._fingerprint = self.cfg.config_hash()
        self.manifest_hash = self.cfg._resources.get("laptop_manifest_sha256")
        self._unusable = False
        self._lock = Lock()
        self._closed = False
        self._serial = _DaemonPool(1)
        self._workers = _DaemonPool(2)
        self._cleanup = weakref.finalize(self, _close_pools, self._serial, self._workers)
        self.cfg.seed_everything()

    def close(self):
        """Stop accepting work; daemon native calls cannot delay interpreter exit."""
        self._closed = True
        self._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _stage(name: str, clk: TurnClock, function, *args):
        with clk.stage(name):
            try:
                return function(*args)
            except Exception as exc:
                raise StageFailure(name, type(exc).__name__) from exc

    @staticmethod
    def _await(future: Future, name: str, deadline: float):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise StageFailure(name, "TimeoutError", timeout=True)
        try:
            return future.result(timeout=remaining)
        except FutureTimeout as exc:
            raise StageFailure(name, "TimeoutError", timeout=True) from exc
        except StageFailure:
            raise
        except Exception as exc:
            raise StageFailure(name, type(exc).__name__) from exc

    def _branch_a(self, seg: np.ndarray, src: Lang, tgt: Lang, clk: TurnClock):
        # An in-place denoiser must not mutate Branch B's corresponding segment.
        x = self._stage("denoise", clk, self.s.denoiser.process, seg.copy())
        x = validate_audio(x, self.cfg.sample_rate, self.cfg.max_utterance_s)
        if len(x) != len(seg):
            raise StageFailure("denoise", "LengthMismatch")
        asr = self._stage("asr", clk, self.s.asr.transcribe, x, src)
        if not isinstance(asr, AsrResult) or asr.lang != src:
            raise StageFailure("asr", "TypeError")
        if not asr.text.strip():
            return asr, MtResult(src_text="", tgt_text="", src_lang=src, tgt_lang=tgt), SafetyReport(passed=False)
        mt = self._stage("nmt", clk, self.s.nmt.translate, asr.text, src, tgt)
        if not isinstance(mt, MtResult) or (mt.src_lang, mt.tgt_lang, mt.src_text) != (src, tgt, asr.text):
            raise StageFailure("nmt", "TypeError")
        if mt.truncated or mt.terminated_by_eos is False or mt.constraints_satisfied is False:
            raise StageFailure("nmt", "TruncatedTranslation")
        evidence = getattr(mt, "evidence", None)
        if evidence is not None and (evidence.truncated or evidence.input_truncated or not evidence.eos_reached
                or evidence.source_unknown_tokens or evidence.output_unknown_tokens or evidence.tokenizer_issues
                or evidence.unsatisfied_constraints or evidence.constraints_requested != evidence.constraints_satisfied
                or evidence.source_unknown_tokens > evidence.source_tokens
                or evidence.output_unknown_tokens > evidence.output_tokens):
            return asr, mt, SafetyReport(passed=False, reasons=["translation_evidence_invalid_or_incomplete"])
        if self.s.mode == "full" and evidence is None and mt.terminated_by_eos is None:
            return asr, mt, SafetyReport(passed=False, reasons=["translation_evidence_missing"])
        safety = self._stage("safety", clk, self.s.safety.check, mt)
        return asr, mt, safety

    def _branch_b(self, seg: np.ndarray, clk: TurnClock):
        return self._stage("branch_b", clk, self.s.branch_b.analyze, seg.copy())

    def _synthesize(self, text: str, tgt: Lang, clk: TurnClock, on_audio=None) -> np.ndarray:
        chunks = []
        with clk.stage("tts"):
            try:
                for chunk in self.s.tts.stream(text, tgt):
                    c = np.asarray(chunk)
                    if c.ndim != 1 or c.dtype.kind != "f" or not np.isfinite(c).all() or (c.size and np.max(np.abs(c)) > 1):
                        raise ValueError("invalid TTS samples")
                    if not c.size:
                        continue
                    if not chunks:
                        clk.mark("first_audio")
                    approved = c.astype(np.float32, copy=True)
                    if self.cfg.config_hash() != self._fingerprint:
                        raise StageFailure("config", "ChangedDuringTurn")
                    if on_audio is not None:
                        # Explicit streaming consumers receive approved audio now,
                        # not after the remaining synthesis finishes. Already
                        # delivered audio cannot be retracted after a later fault.
                        on_audio(approved.copy(), self.cfg.tts_sample_rate)
                        if len(chunks) == 0:
                            clk.mark("first_delivery")
                    chunks.append(approved)
                if not chunks:
                    raise ValueError("TTS returned no audio")
                return np.concatenate(chunks)
            except Exception as exc:
                raise StageFailure("tts", type(exc).__name__) from exc

    def run(self, wav: np.ndarray, utt_id: str, session_id: str = "s0", src: Lang | None = None, *, on_audio=None) -> TurnResult:
        """Process one turn, returning ABSTAIN/REPEAT instead of propagating stage errors.

        Timeout permanently retires this instance. Use the isolated CLI runner
        for actual native-process termination; direct callers must replace the
        instance and own their process lifecycle.
        ``on_audio(chunk, sample_rate)`` explicitly opts into live delivery after
        full-translation approval. Buffered default discards partial failed audio;
        a live sink must abort queued playback on failure (played audio is final).
        """
        start = time.monotonic()
        audio_s = len(wav) / self.cfg.sample_rate if isinstance(wav, np.ndarray) and wav.ndim == 1 else 0.0
        if not self._lock.acquire(blocking=False):
            return failure_result(self.cfg, utt_id, session_id, audio_s, "pipeline", "ConcurrentTurn",
                                  mode=self.s.mode, capabilities=self.s.capabilities)
        clk = TurnClock()
        clk.mark("endpoint")
        pending = []
        asr = mt = safety = None
        src = src if src is not None else self.cfg.direction.split("-")[0]
        asr_text = nmt_text = ""
        out = np.zeros(0, np.float32)
        status, error_stage, error_type = "ok", None, None
        decision = GateDecision(action=GateAction.ABSTAIN)
        deadline = start + self.cfg.turn_timeout_s
        try:
            if self._unusable:
                raise StageFailure("pipeline", "RetiredAfterTimeout")
            if self._closed:
                raise StageFailure("input", "RuntimeError")
            if self.cfg.config_hash() != self._fingerprint:
                raise StageFailure("config", "ChangedAfterInitialization")
            x = validate_audio(wav, self.cfg.sample_rate, self.cfg.max_utterance_s)
            if src != self.cfg.direction.split("-")[0]:
                raise StageFailure("input", "UnsupportedLanguage")
            if np.max(np.abs(x)) <= self.cfg.silence_peak:
                status = "no_speech"
                decision = GateDecision(action=GateAction.REPEAT, reasons=["silence", "no_speech_or_empty_asr"])
            else:
                ff = self._serial.submit(self._stage, "frontend", clk, self.s.frontend.process, x.copy())
                pending.append(ff)
                y = self._await(ff, "frontend", deadline)
                y = validate_audio(y, self.cfg.sample_rate, self.cfg.max_utterance_s)
                if len(y) != len(x):
                    raise StageFailure("frontend", "LengthMismatch")
                fv = self._serial.submit(self._stage, "vad", clk, self.s.vad.segment, y)
                pending.append(fv)
                t0, t1 = self._await(fv, "vad", deadline)
                if not np.isfinite([t0, t1]).all() or not 0 <= t0 <= t1 <= audio_s:
                    raise StageFailure("vad", "InvalidSegment")
                seg = y[int(t0 * self.cfg.sample_rate):int(t1 * self.cfg.sample_rate)]
                if not seg.size or len(seg) / self.cfg.sample_rate < self.cfg.vad.min_speech_s or np.max(np.abs(seg)) <= self.cfg.silence_peak:
                    status = "no_speech"
                    reason = "speech_too_short" if seg.size and len(seg) / self.cfg.sample_rate < self.cfg.vad.min_speech_s else "no_speech"
                    decision = GateDecision(action=GateAction.REPEAT, reasons=[reason])
                else:
                    tgt = self.cfg.direction.split("-")[1]
                    branch_deadline = min(deadline, time.monotonic() + self.cfg.branch_timeout_s)
                    fa = self._workers.submit(self._branch_a, seg, src, tgt, clk)
                    fb = self._workers.submit(self._branch_b, seg, clk)
                    pending.extend((fa, fb))
                    asr, mt, safety = self._await(fa, "branch_a", branch_deadline)
                    urgency = self._await(fb, "branch_b", branch_deadline)
                    if self.s.mode in ("full", "branch-a-bench"):
                        # Raw PTT evidence must survive denoising/cropping: a quiet
                        # window and VAD holes otherwise disappear from the proxy.
                        asr.quality = acoustic_evidence(x, self.cfg.sample_rate, text=asr.text,
                            speech_duration_s=len(seg) / self.cfg.sample_rate,
                            vad_ratio=getattr(self.s.vad, "last_speech_ratio", None),
                            voicing_quality=urgency.voicing_quality)
                    asr_text, nmt_text = asr.text, mt.tgt_text
                    fg = self._serial.submit(self._stage, "gate", clk, self.s.gate, asr, safety, urgency, mt.tgt_text, self.cfg)
                    pending.append(fg)
                    decision = self._await(fg, "gate", deadline)
                    if (not safety.passed or safety.confirm) and decision.action in (GateAction.SPEAK, GateAction.SPEAK_CUE):
                        decision = GateDecision(action=GateAction.CONFIRM, reasons=safety.reasons, urgency=urgency)
                    if self.s.mode == "full" and decision.action in (GateAction.SPEAK, GateAction.SPEAK_CUE):
                        from tonebridge.quality_gate import assess_quality
                        action, reasons = assess_quality(asr.quality, self.cfg.quality, required=True)
                        if action is not None:
                            decision = GateDecision(action=action, reasons=reasons, urgency=urgency)
                        elif decision.action == GateAction.SPEAK_CUE and not (
                                urgency.cue_enabled and urgency.classifier_status == "READY"
                                and self.cfg.branch_b.tuned and self.cfg.branch_b.mlp_path
                                and self.cfg.branch_b.standardizer):
                            decision = GateDecision(action=GateAction.SPEAK, speak_text=mt.tgt_text,
                                                    reasons=["urgency_cue_disabled"], urgency=urgency)
                    if self.cfg.config_hash() != self._fingerprint:
                        raise StageFailure("config", "ChangedDuringTurn")
                    clk.mark("text_ready")
                    if decision.action in (GateAction.SPEAK, GateAction.SPEAK_CUE) and decision.speak_text:
                        if decision.speak_text != mt.tgt_text:
                            raise StageFailure("gate", "ValueError")
                        ft = self._serial.submit(self._synthesize, decision.speak_text, tgt, clk, on_audio)
                        pending.append(ft)
                        out = self._await(ft, "tts", deadline)
        except AudioInputError:
            status, error_stage, error_type = "invalid_audio", "input", "AudioInputError"
            why = "utterance_too_long" if audio_s > self.cfg.max_utterance_s else "no_speech_or_empty_asr" if audio_s == 0 else "invalid_audio"
            decision = GateDecision(action=GateAction.ABSTAIN, reasons=["invalid_audio", why])
        except Exception as exc:
            failure = exc if isinstance(exc, StageFailure) else StageFailure("pipeline", type(exc).__name__)
            status = "timeout" if failure.timeout else "error"
            error_stage, error_type = failure.stage, failure.error_type
            action = GateAction.CONFIRM if failure.error_type == "TruncatedTranslation" else GateAction.ABSTAIN
            decision = GateDecision(action=action, reasons=[str(failure)])
            self._unusable = failure.timeout or any(not f.done() for f in pending)
        finally:
            for future in pending:
                future.cancel()
            clk.freeze()
            self._lock.release()
            if self._unusable:
                self._cleanup()
        total_ms = (time.monotonic() - start) * 1000
        if self.cfg.config_hash() != self._fingerprint and error_stage != "config":
            status, error_stage, error_type = "error", "config", "ChangedDuringTurn"
            decision = GateDecision(action=GateAction.ABSTAIN, reasons=["config:ChangedDuringTurn"])
        success = status == "ok"
        record = TurnRecord(
            session_id=session_id, utterance_id=utt_id, config_hash=self._fingerprint, direction=self.cfg.direction,
            audio_s=audio_s, stage_ms=dict(clk.stage_ms), total_ms=total_ms, rtf=rtf(total_ms, audio_s),
            endpoint_to_first_audio_ms=clk.between_ms("endpoint", "first_audio") if success else None,
            endpoint_to_first_delivery_ms=clk.between_ms("endpoint", "first_delivery") if success else None,
            endpoint_to_text_ms=clk.between_ms("endpoint", "text_ready") if success else None,
            peak_rss_mb=peak_rss_mb(), gate=decision, asr_text=asr_text, nmt_text=nmt_text,
            mode=self.s.mode, capabilities=dict(self.s.capabilities), status=status,
            error_stage=error_stage, error_type=error_type,
            runtime_manifest_sha256=self.cfg._resources.get("laptop_manifest_sha256"),
            nmt_evidence=getattr(mt, "evidence", None).model_dump(mode="json") if getattr(mt, "evidence", None) is not None else None,
            safety_status=(getattr(safety, "status", None) or ("FAIL" if not safety.passed else "CONFIRM" if safety.confirm else "PASS")) if safety is not None else None,
            safety_reasons=safety.reasons if safety is not None else [])
        return TurnResult(record, out if success else np.zeros(0, np.float32))


def utterance_meta(session_id: str, utt_id: str, n: int, sr: int, end_s: float, src: Lang) -> Utterance:
    return Utterance(session_id=session_id, utterance_id=utt_id, sample_rate=sr, n_samples=n,
                     speech_end_s=end_s, src_lang=src)
