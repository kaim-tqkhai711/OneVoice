"""Local CPU TTS for laptop language packs. Full synthesis, then chunk yielding."""
from pathlib import Path

import numpy as np

from tonebridge.contracts import Lang


class LocalTts:
    def __init__(self, lang: Lang, spec: dict, root: Path, threads: int = 2):
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
        self._chunk = max(1, self.sample_rate // 4)

    def stream(self, text: str, lang: Lang):
        if lang != self.lang:
            raise ValueError(f"TTS {self.lang} cannot speak {lang}")
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
        y = np.asarray(audio.samples, dtype=np.float32)
        for i in range(0, len(y), self._chunk):
            yield y[i:i + self._chunk]
