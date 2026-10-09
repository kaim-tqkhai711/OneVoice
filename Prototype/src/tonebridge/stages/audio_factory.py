"""Audio ownership boundary. Paths resolve against Prototype, never caller cwd."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def audio_specs(cfg):
    return json.loads((ROOT / cfg.audio_config).read_text(encoding="utf-8"))


def build_asr(cfg, threads=2):
    from tonebridge.stages.asr_sherpa import SherpaTransducer
    lang = cfg.direction.split("-")[0]
    spec = audio_specs(cfg)["asr"][lang]
    return SherpaTransducer(lang, **{key: ROOT / spec[key] for key in ("encoder", "decoder", "joiner", "tokens")}, threads=threads)


def build_tts(cfg, threads=2):
    from tonebridge.stages.tts_local import LocalTts
    lang = cfg.direction.split("-")[1]
    return LocalTts(lang, audio_specs(cfg)["tts"][lang], ROOT, threads)
