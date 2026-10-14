"""Explicit one-turn CLI with fail-closed configuration and bounded inference.

python -m tonebridge.cli --mode full --wav in.wav --out out.wav
Configuration/preflight errors exit 2; safe runtime/input failures exit 1.
REPEAT/CONFIRM/ABSTAIN are valid gate decisions and never produce audio.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy.io import wavfile

from tonebridge.audio import AudioInputError, load_wav
from tonebridge.config import ROOT, PipelineConfig
from tonebridge.pipeline import failure_result
from tonebridge.stages.factory import build_stub_stages, capabilities
from tonebridge.telemetry import JsonlLogger, PrivacyLogError


class PrivateArgumentParser(argparse.ArgumentParser):
    """Argparse must not echo user-supplied values or filenames on errors."""
    def error(self, message):
        if message.startswith(("unrecognized arguments", "argument ")):
            message = "invalid command-line arguments; check --mode and option values"
        super().error(message)


def public_row(record, log_content=False):
    """Compatibility API; content serialization still requires explicit opt-in."""
    return record.model_dump(mode="json", context={"eval_mode": bool(log_content)})


def main(argv: list[str] | None = None) -> int:
    """Validate before loading models; emit mode/capabilities to stderr at startup."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf8", errors="replace")
    # Laptop configs explicitly select the four-direction audio registry. Keep
    # that upstream entrypoint, with the same mandatory modes and privacy rules.
    probe = PrivateArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.json")
    location, _ = probe.parse_known_args(argv)
    try:
        laptop_selected = json.loads(location.config.read_text(encoding="utf8")).get("audio_config") is not None
    except (OSError, ValueError, AttributeError):
        laptop_selected = False  # the strict parser/loader below reports the error
    if laptop_selected:
        from tonebridge.laptop_cli import main as laptop_main
        return laptop_main(argv)
    parser = PrivateArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("stub", "branch-a-bench", "full"))
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.json")
    parser.add_argument("--assert-offline", action="store_true")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--eval-mode", action="store_true", help="explicitly retain transcript/translation text for evaluation")
    parser.add_argument("--playback", action="store_true", help="stream approved chunks to audio hardware; requires requirements-playback.txt")
    args = parser.parse_args(argv)
    if args.playback:
        import importlib.util
        if importlib.util.find_spec("sounddevice") is None:
            parser.error("playback dependency missing; install requirements-playback.txt")
    # Missing/malformed config is an error, never a request for defaults.
    try:
        cfg = PipelineConfig.load(args.config)
        caps = capabilities(cfg, args.mode)
        print(json.dumps({"event": "startup", "mode": args.mode, "capabilities": caps}, ensure_ascii=False), file=sys.stderr)
        cfg = cfg.prepare(args.mode, verify_assets=False)
    except Exception:
        parser.error("configuration error: invalid or unavailable effective config")
    if args.out.resolve() == args.wav.resolve():
        parser.error("input and output WAV paths must differ")
    audit = None
    try:
        audio = load_wav(args.wav, cfg.sample_rate, cfg.max_utterance_s)
    except AudioInputError:
        result = failure_result(cfg, args.wav.stem, "s0", 0, "input", "AudioInputError",
                                status="invalid_audio", mode=args.mode, capabilities=caps)
    else:
        from tonebridge.preflight import check_dependencies
        environment = check_dependencies(args.mode)
        if environment["errors"]:
            parser.error("dependency preflight failed; run tools/check_runtime.py for environment diagnostics")
        # Parent verifies assets before creating the native worker. The child
        # verifies them again at load, and never reads gate/branch config files.
        try:
            cfg = cfg.prepare(args.mode)
        except Exception:
            parser.error("asset preflight failed: missing asset or invalid asset/configuration")
        from tonebridge.runtime import run_isolated
        try:
            result, audit = run_isolated(cfg, args.mode, audio, args.wav.stem,
                                         args.wav.with_suffix(".txt"), assert_offline=args.assert_offline, playback=args.playback)
        except (OSError, RuntimeError):
            parser.error("worker lifecycle failure")
    row = result.record.model_dump(mode="json", context={"eval_mode": args.eval_mode})
    if audit is not None:
        row["offline_audit"] = audit
    # Never overwrite an old successful file with a failed/blocked turn or leave
    # a misleading pre-existing output path for callers to play back.
    if args.out.exists() and not result.out_wav.size:
        parser.error("blocked turn: output path already exists; choose a fresh path")
    try:
        log_path = args.log or Path("results/eval-turns.jsonl" if args.eval_mode else "results/turns.jsonl")
        row = JsonlLogger(log_path, eval_mode=args.eval_mode).write(row)
        if result.out_wav.size:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(args.out, cfg.tts_sample_rate, result.out_wav)
    except (OSError, PrivacyLogError):
        parser.error("cannot write turn output: output unavailable or incompatible log privacy policy")
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 1 if result.record.status in ("invalid_audio", "error", "timeout") else 0


if __name__ == "__main__":
    raise SystemExit(main())
