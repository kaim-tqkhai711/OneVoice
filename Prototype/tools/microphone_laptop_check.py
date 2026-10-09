"""Coordinated live hardware check. Waits for Enter after models are ready.

Run explicitly when the speaker is ready; source audio stays in memory. Only
guard-approved speech is saved/played. Human listening confirmation is separate.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonebridge.audio import record_microphone, write_wav_new
from tonebridge.cli import public_row
from tonebridge.config import PipelineConfig
from tonebridge.offline_guard import OfflineGuard
from tonebridge.pipeline import Pipeline
from tonebridge.stages.factory import real_full
from tonebridge.telemetry import JsonlLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/laptop_vi-en.json")
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=ROOT / "results/laptop_current/microphone.jsonl")
    args = ap.parse_args()
    cfg = PipelineConfig.load(args.config)
    if not 0 < args.seconds <= cfg.max_utterance_s or args.out.exists():
        raise ValueError("invalid duration or output exists")
    with OfflineGuard() as guard:
        stages = real_full(cfg)
        pipe = Pipeline(cfg, stages)
        print("READY: models loaded; press Enter to begin microphone capture", flush=True)
        input()
        audio = record_microphone(args.seconds, cfg.sample_rate)
        result = pipe.run(audio, "live-hardware")
        row = public_row(result.record)
        row["output_written"] = False
        row["input_peak"] = float(np.max(np.abs(audio))) if audio.size else 0.0
        row["input_rms"] = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        row["playback_api_completed"] = False
        row["human_listening_confirmed"] = None
        if result.out_wav.size:
            write_wav_new(args.out, stages.tts.sample_rate, result.out_wav)
            row["output_written"] = True
            import winsound
            winsound.PlaySound(str(args.out.resolve()), winsound.SND_FILENAME)
            row["playback_api_completed"] = True
        pipe.close()
    row["offline_audit"] = guard.report()
    JsonlLogger(args.log).write(row)
    print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    return int(result.record.status == "error")


if __name__ == "__main__":
    raise SystemExit(main())
