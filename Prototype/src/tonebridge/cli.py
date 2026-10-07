"""CLI: one wav in -> one wav out + JSONL turn record.

Run:
    python -m tonebridge.cli --wav in.wav --out out.wav --config configs/pipeline.json --log results/turns.jsonl
Pass: exit 0, out.wav written (when gate approves), one JSONL row appended.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from tonebridge.config import PipelineConfig
from tonebridge.pipeline import Pipeline, Stages
from tonebridge.stages import stubs
from tonebridge.telemetry import JsonlLogger


def load_wav(path: Path, expect_sr: int) -> np.ndarray:
    """-> wav [T] float32 in [-1, 1], mono @ expect_sr. Raises if the rate differs (no silent resampling)."""
    sr, data = wavfile.read(path)
    if sr != expect_sr:
        raise ValueError(f"{path}: sample rate {sr} != {expect_sr}")
    x = data.astype(np.float32)
    if data.dtype == np.int16:
        x /= 32768.0
    return x.mean(axis=1) if x.ndim == 2 else x  # [T, C] -> [T]


def build_stub_stages(cfg: PipelineConfig, transcript: Path | None) -> Stages:
    return Stages(frontend=stubs.PassthroughFrontEnd(), vad=stubs.WholeClipVad(cfg.sample_rate),
                  denoiser=stubs.PassthroughDenoiser(), asr=stubs.SidecarAsr(transcript),
                  nmt=stubs.TagNmt(cfg.direction), safety=stubs.AlwaysPassSafety(),
                  branch_b=stubs.UnknownBranchB(), tts=stubs.ToneTts(cfg.tts_sample_rate))


def main() -> None:
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles/pipes default to cp1258/cp1252: Vietnamese text would crash the final print
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/pipeline.json"))
    ap.add_argument("--real", action="store_true", help="real VAD/denoise/ASR/NMT/TTS, Branch A only (Branch B, safety, gate stubbed)")
    ap.add_argument("--full", action="store_true", help="all stages real (0 stubs)")
    ap.add_argument("--assert-offline", action="store_true", help="block + count network use during the run; adds net_attempts_python / net_conns_os_seen to the JSONL row (expect 0)")
    ap.add_argument("--log", type=Path, default=Path("results/turns.jsonl"))
    a = ap.parse_args()
    cfg = PipelineConfig.load(a.config) if a.config.exists() else PipelineConfig()
    guard = None
    if a.assert_offline:
        from tonebridge.offline_guard import OfflineGuard
        guard = OfflineGuard().__enter__()  # active from before the models load until after the turn
    try:
        if a.full:
            from tonebridge.stages.factory import real_full
            stages = real_full(cfg)
        elif a.real:
            from tonebridge.stages.factory import real_branch_a
            stages = real_branch_a(cfg)
        else:
            stages = build_stub_stages(cfg, a.wav.with_suffix(".txt"))
        pipe = Pipeline(cfg, stages)
        res = pipe.run(load_wav(a.wav, cfg.sample_rate), utt_id=a.wav.stem)
    finally:
        if guard:
            guard.__exit__(None, None, None)
    row = json.loads(res.record.model_dump_json())
    if guard:
        row["offline_audit"] = guard.report()
    JsonlLogger(a.log).write(row)
    if res.out_wav.size:
        wavfile.write(a.out, getattr(stages.tts, 'sample_rate', cfg.tts_sample_rate), res.out_wav)  # [T_out] float32
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
