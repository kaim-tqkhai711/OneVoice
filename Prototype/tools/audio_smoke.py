"""Real audio-component smoke, not a clinical accuracy benchmark or independent test set."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonebridge.audio import load_wav
from tonebridge.config import PipelineConfig
from tonebridge.offline_guard import OfflineGuard
from tonebridge.stages.audio_factory import build_asr, build_tts


def run(destination):
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for lang, direction, phrase in [("vi", "en-vi", "Xin chào, bạn có khỏe không?"), ("en", "vi-en", "Hello, how are you today?"),
                                    ("ko", "en-ko", "안녕하세요. 오늘 기분이 어떠세요?")]:
        with OfflineGuard() as guard:
            cfg = PipelineConfig(direction=direction, urgency_enabled=False)
            t0 = time.perf_counter()
            tts = build_tts(cfg)
            load_tts = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            audio = np.concatenate(list(tts.stream(phrase, lang)))
            synth_ms = (time.perf_counter() - t0) * 1000
            if not audio.size or not np.isfinite(audio).all():
                raise RuntimeError("Invalid TTS output")
            output = destination / f"tts_{lang}.wav"
            sample_rate = tts.sample_rate
            wavfile.write(output, sample_rate, audio)
            del tts
            asr_cfg = PipelineConfig(direction={"vi": "vi-en", "en": "en-vi", "ko": "ko-en"}[lang])
            t0 = time.perf_counter()
            asr = build_asr(asr_cfg)
            load_asr = (time.perf_counter() - t0) * 1000
            x = load_wav(ROOT / f"models/asr/zipformer-{lang}-int8/test_wavs/0.wav")
            t0 = time.perf_counter()
            result = asr.transcribe(x, lang)
            decode_ms = (time.perf_counter() - t0) * 1000
            if not result.text:
                raise RuntimeError(f"Empty {lang} transcript")
            row = {"lang": lang, "tts_sample_rate": sample_rate,
                   "tts_audio_s": len(audio) / sample_rate, "tts_load_ms": load_tts, "tts_synth_ms": synth_ms,
                   "asr_load_ms": load_asr, "asr_decode_ms": decode_ms, "asr_confidence": result.confidence,
                   "asr_sample_text": result.text, "rss_mb": psutil.Process().memory_info().rss / 2**20}
            del asr
        row["offline_audit"] = guard.report()
        rows.append(row)
        print(lang, "ASR/TTS OK", flush=True)
    report = {"label": "audio-component smoke only; model-provided samples; no medical quality claim", "platform": platform.platform(), "rows": rows}
    (destination / "audio_smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/laptop_current/audio")
    run(ap.parse_args().out_dir)
