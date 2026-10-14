"""Registered laptop CLI with mandatory modes and metadata-only defaults."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from tonebridge.audio import load_wav, record_microphone, write_wav_new
from tonebridge.config import PipelineConfig
from tonebridge.pipeline import Pipeline, Stages
from tonebridge.stages.factory import build_stub_stages
from tonebridge.cli import PrivateArgumentParser, public_row
from tonebridge.runtime import quiet_worker_output
from tonebridge.telemetry import JsonlLogger

ROOT = Path(__file__).resolve().parents[2]
DIRECTIONS = ("vi-en", "en-vi", "en-ko", "ko-en")


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = PrivateArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path)
    source.add_argument("--record-seconds", type=float, help="explicit microphone capture; raw audio is not saved")
    ap.add_argument("--out", type=Path, required=True, help="new output path; existing files are refused")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.json")
    ap.add_argument("--direction", choices=DIRECTIONS)
    ap.add_argument("--text-factory", help="partner adapter module:function(cfg, threads) -> (Nmt, SafetyChecker)")
    ap.add_argument("--mode", required=True, choices=("stub", "full"))
    ap.add_argument("--assert-offline", action="store_true")
    ap.add_argument("--eval-mode", action="store_true", help="explicit text logging for evaluation")
    ap.add_argument("--play", action="store_true", help="play completed approved WAV on Windows")
    ap.add_argument("--log", type=Path, default=ROOT / "results/laptop_current/turns.jsonl")
    a = ap.parse_args(argv)
    guard, cfg, res, pipe, logger = None, None, None, None, None
    try:
        if a.out.exists():
            raise FileExistsError("output_already_exists")
        cfg = PipelineConfig.load(a.config)
        if a.direction:
            cfg.direction = a.direction  # validated assignment retains project-root snapshots
        if a.text_factory:
            cfg.text_factory = a.text_factory
        print(json.dumps({"event": "startup", "mode": a.mode, "direction": cfg.direction,
                          "capabilities": {"runtime": "registered laptop packs", "urgency": "disabled" if not cfg.urgency_enabled else "configured"}}), file=sys.stderr)
        logger = JsonlLogger(a.log, eval_mode=a.eval_mode)
        if a.assert_offline:
            from tonebridge.offline_guard import OfflineGuard
            guard = OfflineGuard().__enter__()
        x = load_wav(a.wav, cfg.sample_rate, cfg.max_utterance_s, convert=True) if a.wav else None
        with quiet_worker_output():
            if (a.mode == "stub"):
                stages = build_stub_stages(cfg, a.wav.with_suffix(".txt") if a.wav else None)
            else:
                from tonebridge.stages.factory import real_full
                stages = real_full(cfg)
            if a.record_seconds is not None:
                if not 0 < a.record_seconds <= cfg.max_utterance_s:
                    raise ValueError("record_seconds_exceeds_config_limit")
                x = record_microphone(a.record_seconds, cfg.sample_rate)
            else:
                assert x is not None
            pipe = Pipeline(cfg, stages)
            res = pipe.run(x, utt_id=a.wav.stem if a.wav else "microphone")
        row = public_row(res.record, a.eval_mode)
        row["mode"] = a.mode
        row["output_written"] = False
        if res.out_wav.size:
            write_wav_new(a.out, stages.tts.sample_rate, res.out_wav)
            row["output_written"] = True
            if a.play:
                import winsound
                winsound.PlaySound(str(a.out.resolve()), winsound.SND_FILENAME)
    except Exception as error:
        row = {"status": "error", "error_type": type(error).__name__,
               "error_stage": "setup_or_io" if res is None else "output",
               "direction": cfg.direction if cfg else a.direction,
               "gate": {"action": "ABSTAIN", "reasons": ["runtime_failure"]}, "output_written": False}
        if res is None:
            print(f"{type(error).__name__}: setup_or_io_failed", file=sys.stderr)
    finally:
        if pipe:
            pipe.close()
        if guard:
            guard.__exit__(None, None, None)
    if guard:
        row["offline_audit"] = guard.report()
    try:
        if logger is None:
            logger = JsonlLogger(a.log, eval_mode=a.eval_mode)
        row = logger.write(row)
    except (OSError, ValueError):
        ap.error("cannot write turn output; select a compatible writable log")
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 1 if row["status"] in ("error", "timeout", "invalid_audio") else 0


if __name__ == "__main__":
    raise SystemExit(main())
