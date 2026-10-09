"""Build guarded runtime stages. real_branch_a is retained only for historical research tools."""
from __future__ import annotations

from tonebridge.config import PipelineConfig
from tonebridge.pipeline import Stages
from tonebridge.stages import stubs

ASR_DECODER = "decoder-epoch-12-avg-8.int8.onnx"  # locked shipped file (PROGRESS.md, D2)


def build_denoiser(cfg: PipelineConfig):
    if cfg.denoise_mode == "off":
        return stubs.PassthroughDenoiser()
    from tonebridge.stages.denoise import GtcrnDenoiser, OaDenoiser
    g = GtcrnDenoiser()
    return g if cfg.denoise_mode == "on" else OaDenoiser(g, cfg.oa_beta)


GLOSSARY_BONUS = 5.0  # locked on dev, M2 round 1 (results/optim_log.jsonl, tag m2_r1)


def real_branch_a(cfg: PipelineConfig, threads: int = 2, glossary: bool = True) -> Stages:
    from tonebridge.stages.asr_sherpa import SherpaZipformerVi
    from tonebridge.stages.nmt_ort import OrtMarianNmt
    from tonebridge.stages.tts_piper import PiperEn
    from tonebridge.stages.vad_silero import SileroVad
    assert cfg.direction == "vi-en", "only vi-en is built (ADR-002)"
    nmt = OrtMarianNmt(threads=threads)
    if glossary:
        from tonebridge.nmt_constraints import GlossaryConstrainer
        nmt.constrainer, nmt.constraint_bonus = GlossaryConstrainer(), GLOSSARY_BONUS
    return Stages(frontend=stubs.PassthroughFrontEnd(), vad=SileroVad(sample_rate=cfg.sample_rate), denoiser=build_denoiser(cfg),
                  asr=SherpaZipformerVi(decoder=ASR_DECODER, threads=threads), nmt=nmt,
                  safety=stubs.AlwaysPassSafety(), branch_b=stubs.UnknownBranchB(), tts=PiperEn(threads=threads))


def real_full(cfg: PipelineConfig, threads: int = 2) -> Stages:
    """Guarded audio + registered text stages. Urgency is optional and remains UNKNOWN without a trained MLP."""
    from tonebridge.gate import Gate, GateConfig
    from tonebridge.stages.audio_factory import ROOT, build_asr, build_tts
    from tonebridge.stages.vad_silero import SileroVad
    nmt, safety = build_text_stages(cfg, threads)
    branch_b = stubs.UnknownBranchB()
    if cfg.urgency_enabled:
        from tonebridge.branch_b import BranchB
        branch_b = BranchB(threads=1)
    return Stages(frontend=stubs.PassthroughFrontEnd(), vad=SileroVad(model=ROOT / "models/silero_vad.onnx"),
                  denoiser=build_denoiser(cfg), asr=build_asr(cfg, threads), nmt=nmt, safety=safety,
                  branch_b=branch_b, tts=build_tts(cfg, threads),
                  gate=Gate(GateConfig(asr_confidence_min=cfg.thresholds.asr_confidence_min,
                                       urgency_high=cfg.thresholds.urgency_high,
                                       voicing_quality_min=cfg.thresholds.voicing_quality_min)))


def build_text_stages(cfg: PipelineConfig, threads: int = 2):
    """Friend-owned handoff: module:function(cfg, threads) -> (Nmt, SafetyChecker)."""
    if cfg.text_factory:
        from importlib import import_module
        module, separator, name = cfg.text_factory.partition(":")
        if not separator:
            raise ValueError("text_factory must be module:function")
        nmt, safety = getattr(import_module(module), name)(cfg, threads)
        if not callable(getattr(nmt, "translate", None)) or not callable(getattr(safety, "check", None)):
            raise TypeError("text_factory must return (Nmt, SafetyChecker)")
        return nmt, safety
    from tonebridge.clinical_safety import ClinicalSafetyChecker
    source, target = cfg.direction.split("-")
    if cfg.direction != "vi-en":
        import json
        from pathlib import Path
        from tonebridge.stages.audio_factory import ROOT
        registry_path = Path(cfg.nmt_config)
        if not registry_path.is_absolute():
            registry_path = ROOT / registry_path
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        spec = registry["models"].get(cfg.direction)
        if spec is None:
            raise ValueError(f"NMT/safety for {cfg.direction} not registered")
        if spec.get("status", "").startswith("quarantined"):
            raise ValueError("nmt_quarantined:" + cfg.direction)
        assets = Path(spec["assets"])
        if not assets.is_absolute():
            assets = ROOT / assets
        if spec.get("backend") == "argos":
            from tonebridge.stages.nmt_argos import ArgosEnKoNmt
            return ArgosEnKoNmt(assets, threads=threads, format_asr_source=cfg.format_asr_source), ClinicalSafetyChecker(source, target)
        from tonebridge.stages.nmt_marian import MarianTextAdapter
        return MarianTextAdapter(assets, source, target, threads=threads,
                                 allow_partial_vocabulary=spec.get("allow_partial_vocabulary", False),
                                 format_asr_source=cfg.format_asr_source), ClinicalSafetyChecker(source, target)
    from tonebridge.stages.nmt_ort import OrtMarianNmt
    from tonebridge.nmt_constraints import GlossaryConstrainer
    nmt = OrtMarianNmt(threads=threads)
    nmt.constrainer, nmt.constraint_bonus = GlossaryConstrainer(), GLOSSARY_BONUS
    return nmt, ClinicalSafetyChecker(source, target)
