"""Local CPU TTS for laptop language packs. Full synthesis, then chunk yielding."""
from pathlib import Path

import numpy as np

from tonebridge.contracts import Lang


class LocalTts:
    def __init__(self, lang: Lang, spec: dict, root: Path, threads: int = 2,
                 chunk_s: float = .25, first_clause: bool = True):
        import sherpa_onnx as so

        self.lang, self.kind, self.spec = lang, spec["kind"], spec

        def asset(key):
            path = root / spec[key]
            if not path.exists():
                raise FileNotFoundError(f"TTS asset missing: {path}")
            return str(path)

        if self.kind == "vits":
            model = so.OfflineTtsModelConfig(
                vits=so.OfflineTtsVitsModelConfig(model=asset("model"), tokens=asset("tokens"), data_dir=asset("data_dir")),
                num_threads=threads, provider="cpu")
        elif self.kind == "supertonic":
            model = so.OfflineTtsModelConfig(supertonic=so.OfflineTtsSupertonicModelConfig(
                **{k: asset(k) for k in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder", "tts_json", "unicode_indexer", "voice_style")}),
                num_threads=threads, provider="cpu")
        else:
            raise ValueError(f"Unsupported TTS kind: {self.kind}")
        self._tts = so.OfflineTts(so.OfflineTtsConfig(model=model))
        self.sample_rate = self._tts.sample_rate
        self._chunk = max(1, int(self.sample_rate * chunk_s))
        self.first_clause = first_clause

    def _synthesize(self, text: str, lang: Lang):
        if self.kind == "supertonic":
            import sherpa_onnx as so
            generation = so.GenerationConfig()
            generation.sid = self.spec.get("speaker_id", 0)
            generation.num_steps = 8
            generation.speed = 1.0
            generation.extra["lang"] = lang
            audio = self._tts.generate(text, generation)
        else:
            audio = self._tts.generate(text, sid=self.spec.get("speaker_id", 0), speed=1.0)
        return np.asarray(audio.samples, dtype=np.float32)

    def stream(self, text: str, lang: Lang):
        """Stream approved EN clauses; VI/KO retain whole-utterance synthesis."""
        if lang != self.lang or not text.strip():
            raise ValueError("TTS language mismatch or empty text")
        from tonebridge.stages.tts_piper import split_clauses
        parts = split_clauses(text) if self.first_clause and lang == "en" else [text]
        batches = [parts[0], " ".join(parts[1:])] if len(parts) > 1 else parts
        for part in batches:
            y = self._synthesize(part, lang)
            if not y.size:
                raise ValueError("empty_tts_audio")
            for i in range(0, len(y), self._chunk):
                yield y[i:i + self._chunk]
