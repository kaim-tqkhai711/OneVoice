"""Guarded laptop CLI. Real pipeline by default; test doubles require --stub."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from tonebridge.audio import load_wav, record_microphone, write_wav_new
from tonebridge.config import PipelineConfig
from tonebridge.pipeline import Pipeline, Stages
from tonebridge.stages import stubs
from tonebridge.telemetry import JsonlLogger

ROOT = Path(__file__).resolve().parents[2]
DIRECTIONS = ("vi-en", "en-vi", "en-ko", "ko-en")


def build_stub_stages(cfg: PipelineConfig, transcript: Path | None) -> Stages:
    return Stages(frontend=stubs.PassthroughFrontEnd(), vad=stubs.WholeClipVad(cfg.sample_rate),
                  denoiser=stubs.PassthroughDenoiser(), asr=stubs.SidecarAsr(transcript),
                  nmt=stubs.TagNmt(cfg.direction), safety=stubs.AlwaysPassSafety(),
                  branch_b=stubs.UnknownBranchB(), tts=stubs.ToneTts(cfg.tts_sample_rate))


def public_row(record, log_content=False):
    row = record.model_dump(mode="json")
    if not log_content:
        row.pop("asr_text", None)
        row.pop("nmt_text", None)
        row["gate"].pop("speak_text", None)
        # Checker diagnostics and constraint/tokenizer details can contain source words.
        row["gate"]["reasons"] = [reason.split(":", 1)[0] for reason in row["gate"]["reasons"]]
        row["safety_reasons"] = [reason.split(":", 1)[0] for reason in row.get("safety_reasons", [])]
        if row.get("nmt_evidence"):
            evidence = row["nmt_evidence"]
            evidence["unsatisfied_constraint_count"] = len(evidence.pop("unsatisfied_constraints", []))
            evidence["tokenizer_issues"] = [reason.split(":", 1)[0] for reason in evidence["tokenizer_issues"]]
    return row


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path)
    source.add_argument("--record-seconds", type=float, help="explicit microphone capture; raw audio is not saved")
    ap.add_argument("--out", type=Path, required=True, help="new output path; existing files are refused")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.json")
    ap.add_argument("--direction", choices=DIRECTIONS)
    ap.add_argument("--text-factory", help="partner adapter module:function(cfg, threads) -> (Nmt, SafetyChecker)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--stub", action="store_true", help="test doubles; output is a tone, not translated speech")
    mode.add_argument("--full", action="store_true", help="alias for default guarded pipeline")
    mode.add_argument("--real", action="store_true", help="alias; now includes safety and gate")
    ap.add_argument("--assert-offline", action="store_true")
    ap.add_argument("--log-content", action="store_true", help="include transcript/translation in logs")
    ap.add_argument("--play", action="store_true", help="play completed approved WAV on Windows")
    ap.add_argument("--log", type=Path, default=ROOT / "results/laptop_current/turns.jsonl")
    a = ap.parse_args()
    guard, cfg, res, pipe = None, None, None, None
    try:
        if a.out.exists():
            raise FileExistsError("output_already_exists")
        cfg = PipelineConfig.load(a.config)
        if a.direction:
            cfg = PipelineConfig.model_validate({**cfg.model_dump(), "direction": a.direction})
        if a.text_factory:
            cfg = PipelineConfig.model_validate({**cfg.model_dump(), "text_factory": a.text_factory})
        if a.assert_offline:
            from tonebridge.offline_guard import OfflineGuard
            guard = OfflineGuard().__enter__()
        x = load_wav(a.wav, cfg.sample_rate) if a.wav else None
        if a.stub:
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
        row = public_row(res.record, a.log_content or cfg.log_content)
        row["mode"] = "stub" if a.stub else "guarded"
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
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
    finally:
        if pipe:
            pipe.close()
        if guard:
            guard.__exit__(None, None, None)
    if guard:
        row["offline_audit"] = guard.report()
    JsonlLogger(a.log).write(row)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 1 if row["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
