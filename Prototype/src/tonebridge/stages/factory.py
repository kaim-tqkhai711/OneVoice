"""Builds the Stages bundle. `real_branch_a` = real VAD/denoise/ASR/NMT/TTS; Branch B, safety and gate stay stubs until D3."""
from __future__ import annotations

from tonebridge.config import PipelineConfig
from tonebridge.pipeline import Stages
from tonebridge.stages import stubs
from tonebridge.stages.asr_sherpa import SherpaZipformerVi
from tonebridge.stages.denoise import GtcrnDenoiser, OaDenoiser
from tonebridge.stages.nmt_ort import OrtMarianNmt
from tonebridge.stages.tts_piper import PiperEn
from tonebridge.stages.vad_silero import SileroVad

ASR_DECODER = "decoder-epoch-12-avg-8.int8.onnx"  # locked shipped file (PROGRESS.md, D2)


def build_denoiser(cfg: PipelineConfig):
    if cfg.denoise_mode == "off":
        return stubs.PassthroughDenoiser()
    g = GtcrnDenoiser()
    return g if cfg.denoise_mode == "on" else OaDenoiser(g, cfg.oa_beta)


def real_branch_a(cfg: PipelineConfig, threads: int = 2) -> Stages:
    assert cfg.direction == "vi-en", "only vi-en is built (ADR-002)"
    return Stages(frontend=stubs.PassthroughFrontEnd(), vad=SileroVad(sample_rate=cfg.sample_rate), denoiser=build_denoiser(cfg),
                  asr=SherpaZipformerVi(decoder=ASR_DECODER, threads=threads), nmt=OrtMarianNmt(threads=threads),
                  safety=stubs.AlwaysPassSafety(), branch_b=stubs.UnknownBranchB(), tts=PiperEn(threads=threads))
