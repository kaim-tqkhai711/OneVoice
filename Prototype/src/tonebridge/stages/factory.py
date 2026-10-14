"""Explicit stage modes, constructed exclusively from the effective config."""
from __future__ import annotations

from pathlib import Path

from tonebridge.config import Mode, PipelineConfig
from tonebridge.pipeline import Stages
from tonebridge.stages import stubs


def capabilities(cfg: PipelineConfig, mode: Mode) -> dict[str, str]:
    """Describe disabled/experimental stages honestly before model loading."""
    if mode == "stub":
        return {"frontend": "passthrough", "vad": "stub: whole clip", "denoiser": "passthrough",
                "asr": "stub: sidecar or fixed phrase", "nmt": "stub: tag only",
                "safety": "stub: always pass", "branch_b": "stub: DISABLED (no urgency classifier)", "gate": "stub: v0",
                "tts": "stub: sine tone"}
    return {"frontend": "passthrough: APM disabled", "vad": "Silero CPU",
            "denoiser": "disabled: OFF passthrough" if cfg.denoise_mode == "off" else f"GTCRN CPU ({cfg.denoise_mode})",
            "asr": "Zipformer VI CPU", "nmt": "Marian VI->EN ORT CPU",
            "safety": "scoped clinical relations; unsupported syntax blocks" if mode == "full" else "stub: always pass (benchmark only)",
            "branch_b": ("SwiftF0 + band features; " + ("ONNX classifier configured; " + ("cue threshold tuned" if cfg.branch_b.tuned else "SPEAK_CUE disabled (untuned)")
                if cfg.branch_b.mlp_path and cfg.branch_b.standardizer else "UNCONFIGURED (MLP not trained); SPEAK_CUE disabled")) if mode == "full" else "stub: DISABLED (benchmark only; cue disabled)",
            "gate": ("clinical + acoustic quality gate; " + ("explicit calibration" if cfg.quality.calibrated else "uncalibrated: CONFIRM/REPEAT")) if mode == "full" else "stub: v0 (benchmark only)",
            "tts": "Piper EN CPU; first-clause synthesis" if cfg.decode.tts_first_clause else "Piper EN CPU; whole-utterance synthesis"}


def build_stub_stages(cfg: PipelineConfig, transcript: Path | None = None) -> Stages:
    """Use existing skeleton stages only when the caller explicitly selects stub."""
    effective = cfg.prepare("stub")
    if transcript is not None and transcript.is_file():
        from tonebridge.config import sha256_file
        effective._resources["stub_transcript"] = {"path": str(transcript.resolve()), "sha256": sha256_file(transcript)}
    return Stages(frontend=stubs.PassthroughFrontEnd(), vad=stubs.WholeClipVad(effective.sample_rate),
                  denoiser=stubs.PassthroughDenoiser(), asr=stubs.SidecarAsr(transcript),
                  nmt=stubs.TagNmt(effective.direction), safety=stubs.AlwaysPassSafety(),
                  branch_b=stubs.UnknownBranchB(), tts=stubs.ToneTts(effective.tts_sample_rate),
                  mode="stub", capabilities=capabilities(effective, "stub"), effective_config=effective)


def build_denoiser(cfg: PipelineConfig):
    if cfg.denoise_mode == "off":
        return stubs.PassthroughDenoiser()
    from tonebridge.stages.denoise import GtcrnDenoiser, OaDenoiser
    model = GtcrnDenoiser(model=cfg.model_path("denoise"), threads=cfg.decode.threads)
    return model if cfg.denoise_mode == "on" else OaDenoiser(model, cfg.oa_beta)


def build_stages(cfg: PipelineConfig, mode: Mode, transcript: Path | None = None) -> Stages:
    """Verify all active assets before constructing any native model session."""
    if mode == "stub":
        return build_stub_stages(cfg, transcript)
    effective = cfg.prepare(mode)
    from tonebridge.stages.asr_sherpa import SherpaZipformerVi
    from tonebridge.stages.nmt_ort import OrtMarianNmt
    from tonebridge.stages.tts_piper import PiperEn
    from tonebridge.stages.vad_silero import SileroVad

    d, v = effective.decode, effective.vad
    nmt = OrtMarianNmt(model_dir=effective.model_path("nmt_vi_en"), threads=d.threads,
                      enc_file=d.nmt_encoder, dec_file=d.nmt_decoder, max_new_tokens=d.max_new_tokens,
                      num_beams=d.num_beams, length_penalty=d.length_penalty)
    # The central manifest has authenticated every model byte before this load;
    # a separate laptop asset_manifest.json is not a second trust requirement.
    nmt.provenance_issues = []
    nmt.model_id = effective.models["nmt_vi_en"].name
    nmt.revision = effective.models["nmt_vi_en"].version
    if d.glossary_enabled:
        from tonebridge.nmt_constraints import GlossaryConstrainer
        nmt.constrainer = GlossaryConstrainer(groups=d.glossary_groups, preferred_only_meds=d.preferred_only_meds,
                                             lexicon=effective._lexicon_data)
        nmt.constraint_bonus, nmt.eos_penalty = d.glossary_bonus, d.eos_penalty
    tts = PiperEn(model_dir=effective.model_path("tts_en"), threads=d.threads,
                  chunk_s=d.tts_chunk_s, model_file=d.tts_model, first_clause=d.tts_first_clause)
    if tts.sample_rate != effective.tts_sample_rate:
        raise ValueError("tts_sample_rate does not match the selected voice")
    stages = Stages(
        frontend=stubs.PassthroughFrontEnd(),
        vad=SileroVad(model=effective.model_path("vad"), sample_rate=effective.sample_rate,
                      min_silence_s=v.min_silence_s, min_speech_s=v.min_speech_s, threshold=v.threshold, pad_s=v.pad_s),
        denoiser=build_denoiser(effective),
        asr=SherpaZipformerVi(model_dir=effective.model_path("asr_vi"), decoder=d.asr_decoder,
                             encoder=d.asr_encoder, joiner=d.asr_joiner, threads=d.threads,
                             decoding_method=d.asr_method, num_active_paths=d.asr_active_paths),
        nmt=nmt, safety=stubs.AlwaysPassSafety(), branch_b=stubs.UnknownBranchB(), tts=tts,
        mode=mode, capabilities=capabilities(effective, mode), effective_config=effective)
    if mode == "full":
        from tonebridge.branch_b import BranchB
        from tonebridge.gate import Gate
        from tonebridge.safety import SemanticSafetyChecker
        stages.safety = SemanticSafetyChecker(lexicon=effective._lexicon_data)
        stages.branch_b = BranchB(cfg=effective.branch_b, threads=d.branch_b_threads,
                                  band_model=effective.model_path("band_features"),
                                  swiftf0_model=effective.model_path("swift_f0"), root=effective.root)
        stages.gate = Gate(effective.thresholds)
    return stages


def _overrides(cfg: PipelineConfig, threads: int | None, glossary: bool | None = None) -> PipelineConfig:
    """Retain programmatic benchmark entry points while hashing their overrides."""
    out = cfg.model_copy(deep=True)
    if threads is not None:
        out.decode.threads = threads
    if glossary is not None:
        out.decode.glossary_enabled = glossary
    return out


def real_branch_a(cfg: PipelineConfig, threads: int | None = None, glossary: bool | None = None) -> Stages:
    return build_stages(_overrides(cfg, threads, glossary), "branch-a-bench")


def real_full(cfg: PipelineConfig, threads: int | None = None) -> Stages:
    if cfg.audio_config is not None:
        from tonebridge.stages.laptop_factory import real_full as laptop_full
        return laptop_full(cfg, threads or cfg.decode.threads)
    return build_stages(_overrides(cfg, threads), "full")


def build_text_stages(cfg: PipelineConfig, threads: int = 2):
    """Preserve the upstream registered, guarded text-adapter handoff."""
    from tonebridge.stages.laptop_factory import build_text_stages as registered
    return registered(cfg, threads)
