"""Build guarded runtime stages. real_branch_a is retained only for historical research tools."""
from __future__ import annotations

from tonebridge.config import PipelineConfig
from tonebridge.pipeline import Stages
from tonebridge.stages import stubs

ASR_DECODER = "decoder-epoch-12-avg-8.int8.onnx"  # locked shipped file (PROGRESS.md, D2)


def build_denoiser(cfg: PipelineConfig):
    from tonebridge.stages.factory import build_denoiser as configured
    return configured(cfg)


GLOSSARY_BONUS = 5.0  # locked on dev, M2 round 1 (results/optim_log.jsonl, tag m2_r1)


def real_branch_a(cfg: PipelineConfig, threads: int = 2, glossary: bool = True) -> Stages:
    from tonebridge.stages.factory import real_branch_a as configured
    return configured(cfg, threads, glossary)


def real_full(cfg: PipelineConfig, threads: int = 2) -> Stages:
    """Guarded audio + registered text stages. Urgency is optional and remains UNKNOWN without a trained MLP."""
    from tonebridge.gate import Gate
    from tonebridge.laptop_config import prepare_laptop
    from tonebridge.stages.audio_factory import ROOT, build_asr, build_tts
    from tonebridge.stages.vad_silero import SileroVad
    cfg = prepare_laptop(cfg, threads)
    nmt, safety = build_text_stages(cfg, threads)
    branch_b = stubs.UnknownBranchB()
    if cfg.urgency_enabled:
        from tonebridge.branch_b import BranchB
        branch_b = BranchB(cfg=cfg.branch_b, threads=cfg.decode.branch_b_threads,
                           band_model=cfg.model_path("band_features"), swiftf0_model=cfg.model_path("swift_f0"), root=cfg.root)
    from tonebridge.stages.factory import build_denoiser as configured_denoiser
    tts = build_tts(cfg, threads)
    cfg.tts_sample_rate = tts.sample_rate
    v = cfg.vad
    return Stages(frontend=stubs.PassthroughFrontEnd(), vad=SileroVad(model=cfg.resolve_path("models/silero_vad.onnx"),
                  sample_rate=cfg.sample_rate, min_silence_s=v.min_silence_s, min_speech_s=v.min_speech_s,
                  threshold=v.threshold, pad_s=v.pad_s),
                  denoiser=configured_denoiser(cfg), asr=build_asr(cfg, threads), nmt=nmt, safety=safety,
                  branch_b=branch_b, tts=tts, gate=Gate(cfg.thresholds), mode="full", effective_config=cfg,
                  capabilities={"asr": "local " + cfg.direction.split("-")[0], "nmt": cfg.direction,
                                "gate": "clinical + uncalibrated acoustic confirmation", "branch_b": "configured" if cfg.urgency_enabled else "DISABLED"})


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
    if cfg.direction != "vi-en" or cfg.audio_config is not None:
        import json
        from pathlib import Path
        from tonebridge.stages.audio_factory import ROOT
        registry_path = cfg.resolve_path(cfg.nmt_config)
        registry = cfg._nmt_registry if cfg._nmt_registry is not None else json.loads(registry_path.read_text(encoding="utf-8"))
        spec = registry["models"].get(cfg.direction)
        if spec is None:
            raise ValueError(f"NMT/safety for {cfg.direction} not registered")
        if spec.get("status", "").startswith("quarantined"):
            raise ValueError("nmt_quarantined:" + cfg.direction)
        assets = cfg.resolve_path(spec["assets"])
        checker = ClinicalSafetyChecker(source, target, lexicon=cfg._clinical_lexicon_data or cfg.resolve_path("configs/clinical_lexicon.json"))
        if cfg.direction == "vi-en":
            from tonebridge.stages.nmt_ort import OrtMarianNmt
            nmt = OrtMarianNmt(model_dir=assets, threads=threads, max_new_tokens=cfg.decode.max_new_tokens,
                               num_beams=cfg.decode.num_beams, length_penalty=cfg.decode.length_penalty)
            from tonebridge.nmt_constraints import GlossaryConstrainer
            if cfg.decode.glossary_enabled:
                nmt.constrainer = GlossaryConstrainer(groups=cfg.decode.glossary_groups,
                    preferred_only_meds=cfg.decode.preferred_only_meds, lexicon=cfg._lexicon_data or cfg.resolve_path(cfg.safety_lexicon))
                nmt.constraint_bonus, nmt.eos_penalty = cfg.decode.glossary_bonus, cfg.decode.eos_penalty
            return nmt, checker
        if spec.get("backend") == "argos":
            from tonebridge.stages.nmt_argos import ArgosEnKoNmt
            return ArgosEnKoNmt(assets, threads=threads, max_new_tokens=cfg.decode.max_new_tokens,
                                format_asr_source=cfg.format_asr_source), checker
        if spec.get("backend") not in (None, "marian", "hf-cpu"):
            raise ValueError("unsupported_nmt_backend")
        from tonebridge.stages.nmt_marian import MarianTextAdapter
        return MarianTextAdapter(assets, source, target, threads=threads,
                                 max_new_tokens=cfg.decode.max_new_tokens, num_beams=cfg.decode.num_beams,
                                 allow_partial_vocabulary=spec.get("allow_partial_vocabulary", False),
                                 format_asr_source=cfg.format_asr_source), checker
    from tonebridge.stages.nmt_ort import OrtMarianNmt
    from tonebridge.nmt_constraints import GlossaryConstrainer
    from tonebridge.assets import verify_files
    effective = cfg.prepare("full", verify_assets=False)
    model_dir = effective.model_path("nmt_vi_en")
    entries = [r for r in effective._resources["assets"] if Path(r["path"]).is_relative_to(model_dir)]
    issues = verify_files(entries)
    if issues:
        raise ValueError("NMT asset preflight failed: " + issues[0])
    d = effective.decode
    nmt = OrtMarianNmt(model_dir=model_dir, threads=threads, max_new_tokens=d.max_new_tokens,
                       num_beams=d.num_beams, length_penalty=d.length_penalty,
                       enc_file=d.nmt_encoder, dec_file=d.nmt_decoder)
    nmt.provenance_issues = []
    if d.glossary_enabled:
        nmt.constrainer = GlossaryConstrainer(groups=d.glossary_groups, preferred_only_meds=d.preferred_only_meds,
                                             lexicon=effective._lexicon_data)
        nmt.constraint_bonus, nmt.eos_penalty = d.glossary_bonus, d.eos_penalty
    return nmt, ClinicalSafetyChecker(source, target)
