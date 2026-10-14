"""Disposable process watchdog for CLI turns, including native inference hangs."""
from __future__ import annotations

from contextlib import nullcontext, contextmanager, redirect_stdout, redirect_stderr
import os
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from tonebridge.config import Mode, PipelineConfig
from tonebridge.contracts import TurnRecord
from tonebridge.pipeline import Pipeline, TurnResult, failure_result


def _stop_process(process) -> None:
    """Bound every cleanup wait; terminate/kill native workers still running."""
    process.join(timeout=0.1)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)
    if process.is_alive():
        raise RuntimeError("cannot terminate inference worker")
    process.close()


@contextmanager
def quiet_worker_output():
    """Suppress Python and native stdout/stderr inside the disposable worker only.

    Third-party adapters may print transcripts or input-containing exceptions.
    IPC carries structured results separately; evaluation also uses that channel.
    """
    with open(os.devnull, "w", encoding="utf8") as sink:
        saved = [os.dup(fd) for fd in (1, 2)]
        handles = None
        try:
            for fd in (1, 2):
                os.dup2(sink.fileno(), fd)
            if os.name == "nt":
                import ctypes
                import msvcrt
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.GetStdHandle.restype = ctypes.c_void_p
                kernel.SetStdHandle.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
                handles = [(code, kernel.GetStdHandle(code)) for code in (0xFFFFFFF5, 0xFFFFFFF4)]
                for code, _ in handles:
                    kernel.SetStdHandle(code, msvcrt.get_osfhandle(sink.fileno()))
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            if handles is not None:
                for code, handle in handles:
                    kernel.SetStdHandle(code, handle)
            for fd, original in zip((1, 2), saved):
                os.dup2(original, fd)
                os.close(original)


def _worker(connection, *args) -> None:
    """All model imports and inference occur behind the output privacy boundary."""
    with quiet_worker_output():
        _worker_body(connection, *args)


def _worker_body(connection, cfg: PipelineConfig, mode: Mode, wav: np.ndarray,
            utt_id: str, transcript: Path | None, assert_offline: bool, playback: bool = False) -> None:
    """Own all native sessions and network auditing inside the killable process."""
    from tonebridge.stages.factory import build_stages

    guard = None
    if assert_offline:
        from tonebridge.offline_guard import OfflineGuard
        guard = OfflineGuard()
    try:
        with guard if guard is not None else nullcontext():
            stages = build_stages(cfg, mode, transcript)
            pipe = Pipeline(cfg, stages)
            connection.send(("ready", pipe.cfg.config_hash()))
            player = None
            if playback:
                from tonebridge.playback import StreamingPlayer
                player = StreamingPlayer()
            try:
                result = pipe.run(wav, utt_id, on_audio=player.write if player else None)
            finally:
                if player is not None:
                    player.close(abort="result" not in locals() or result.record.status != "ok")
        audit = guard.report() if guard is not None else None
        if audit and (audit["net_attempts_python"] or audit["net_conns_os_seen"]):
            result = failure_result(pipe.cfg, utt_id, "s0", len(wav) / cfg.sample_rate,
                                    "offline", "NetworkActivity", mode=mode, capabilities=stages.capabilities)
        # Local IPC is internal state, not a log. Parent serializes under its
        # explicit runtime/evaluation policy before writing any output.
        connection.send(("result", result.record.model_dump(context={"eval_mode": True}), result.out_wav, audit))
    except BaseException as exc:
        # Only the type is exposed; an exception message may contain input text.
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


def run_isolated(cfg: PipelineConfig, mode: Mode, wav: np.ndarray, utt_id: str,
                 transcript: Path | None = None, *, assert_offline: bool = False, playback: bool = False) -> tuple[TurnResult, dict | None]:
    """Model-load and turn deadlines have separate budgets; every worker is reaped.

    This runner is deliberately one process per CLI invocation. Model load time
    is excluded from endpoint latency and is not a warm-serving benchmark.
    """
    from tonebridge.stages.factory import capabilities

    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child, cfg, mode, wav, utt_id, transcript, assert_offline, playback),
                              daemon=True, name="tonebridge-inference")
    started = time.monotonic()
    stage, timeout_s = "model_load", cfg.model_load_timeout_s
    try:
        process.start()
    except Exception as exc:
        parent.close()
        child.close()
        process.close()
        return failure_result(cfg, utt_id, "s0", len(wav) / cfg.sample_rate, "worker_start", type(exc).__name__,
                              mode=mode, capabilities=capabilities(cfg, mode)), None
    child.close()
    try:
        if not parent.poll(timeout_s):
            return failure_result(cfg, utt_id, "s0", len(wav) / cfg.sample_rate, stage, "TimeoutError",
                                  status="timeout", mode=mode, capabilities=capabilities(cfg, mode)), None
        message = parent.recv()
        if message[0] == "ready":
            if mode != "stub" and message[1] != cfg.config_hash():
                return failure_result(cfg, utt_id, "s0", len(wav) / cfg.sample_rate, "config", "FingerprintChangedAtLoad",
                                      mode=mode, capabilities=capabilities(cfg, mode)), None
            stage, timeout_s = "turn", cfg.turn_timeout_s + 0.5
            if not parent.poll(timeout_s):
                return failure_result(cfg, utt_id, "s0", len(wav) / cfg.sample_rate, stage, "TimeoutError",
                                      status="timeout", elapsed_ms=timeout_s * 1000,
                                      mode=mode, capabilities=capabilities(cfg, mode)), None
            message = parent.recv()
        if message[0] == "result":
            return TurnResult(TurnRecord.model_validate(message[1]), message[2]), message[3]
        return failure_result(cfg, utt_id, "s0", len(wav) / cfg.sample_rate, stage, message[1],
                              elapsed_ms=(time.monotonic() - started) * 1000,
                              mode=mode, capabilities=capabilities(cfg, mode)), None
    except (EOFError, OSError):
        return failure_result(cfg, utt_id, "s0", len(wav) / cfg.sample_rate, stage, "WorkerExited",
                              mode=mode, capabilities=capabilities(cfg, mode)), None
    finally:
        parent.close()
        _stop_process(process)
