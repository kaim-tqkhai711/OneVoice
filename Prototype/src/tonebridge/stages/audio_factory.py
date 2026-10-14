"""Audio ownership boundary. Paths resolve against Prototype, never caller cwd."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def audio_specs(cfg):
    if cfg._laptop_audio_specs is not None:
        return cfg._laptop_audio_specs
    return json.loads(cfg.resolve_path(cfg.audio_config).read_text(encoding="utf-8"))


def build_asr(cfg, threads=2):
    from tonebridge.stages.asr_sherpa import SherpaTransducer
    lang = cfg.direction.split("-")[0]
    spec = audio_specs(cfg)["asr"][lang]
    return SherpaTransducer(lang, **{key: cfg.resolve_path(spec[key]) for key in ("encoder", "decoder", "joiner", "tokens")},
                            threads=threads, decoding_method=cfg.decode.asr_method, num_active_paths=cfg.decode.asr_active_paths)


def build_tts(cfg, threads=2):
    from tonebridge.stages.tts_local import LocalTts
    lang = cfg.direction.split("-")[1]
    return LocalTts(lang, audio_specs(cfg)["tts"][lang], cfg.root, threads,
                    chunk_s=cfg.decode.tts_chunk_s, first_clause=cfg.decode.tts_first_clause)
