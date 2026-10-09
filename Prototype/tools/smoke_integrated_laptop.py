"""Fresh-process real-model smoke for one direction; generated audio is not quality data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonebridge.audio import load_wav, write_wav_new
from tonebridge.cli import public_row
from tonebridge.config import PipelineConfig
from tonebridge.offline_guard import OfflineGuard
from tonebridge.pipeline import Pipeline
from tonebridge.stages.audio_factory import build_tts
from tonebridge.stages.factory import real_full


def run(direction, destination):
    cfg = PipelineConfig.load(ROOT / f"configs/laptop_{direction}.json")
    src, tgt = direction.split("-")
    destination.mkdir(parents=True, exist_ok=True)
    fixture = destination / f"greeting_{src}.wav"
    with OfflineGuard() as guard:
        if not fixture.exists():
            source_cfg = PipelineConfig(direction={"vi": "en-vi", "en": "vi-en", "ko": "en-ko"}[src])
            tts = build_tts(source_cfg)
            phrase = {"vi": "Xin chào.", "en": "Hello.", "ko": "안녕하세요."}[src]
            audio = np.concatenate(list(tts.stream(phrase, src)))
            write_wav_new(fixture, tts.sample_rate, audio)
            del tts
        start = time.perf_counter()
        stages = real_full(cfg)
        pipe = Pipeline(cfg, stages)
        load_ms = (time.perf_counter() - start) * 1000
        paths = [fixture, *sorted((ROOT / f"models/asr/zipformer-{src}-int8/test_wavs").glob("*.wav"))]
        rows = []
        for i, path in enumerate(paths):
            result = pipe.run(load_wav(path), f"smoke-{i}")
            row = public_row(result.record, True)  # public test/model samples; local ignored output only
            row["sample"] = path.name
            row["out_samples"] = result.out_wav.size
            if result.out_wav.size:
                output = destination / f"approved-{i}.wav"
                if not output.exists():
                    write_wav_new(output, stages.tts.sample_rate, result.out_wav)
            rows.append(row)
        # Development split only; holdout remains untouched.
        references = [json.loads(line) for line in (ROOT / "configs/nmt/text_references_v1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        texts = []
        for item in references:
            if item["direction"] != direction or item["split"] != "dev":
                continue
            start = time.perf_counter()
            mt = stages.nmt.translate(item["source"], src, tgt)
            ms = (time.perf_counter()-start)*1000
            safety = stages.safety.check(mt)
            texts.append({"id": item["id"], "nmt_ms": ms, "translation": mt.model_dump(), "safety": safety.model_dump()})
        pipe.close()
    report = {"direction": direction, "label": "Real model integration smoke; generated/model-provided WAVs; draft dev text; no clinical quality or hardware claim",
       "model_load_ms": load_ms, "config_hash": cfg.config_hash(), "runtime_manifest_sha256": pipe.manifest_hash,
       "audio_turns": len(rows), "gate_counts": dict(Counter(r["gate"]["action"] for r in rows)),
       "status_counts": dict(Counter(r["status"] for r in rows)), "text_dev_count": len(texts),
       "text_safety_counts": dict(Counter(r["safety"]["status"] for r in texts)),
       "nmt_p50_ms": float(np.percentile([r["nmt_ms"] for r in texts], 50)),
       "nmt_p95_ms": float(np.percentile([r["nmt_ms"] for r in texts], 95)),
       "rss_mb": psutil.Process().memory_info().rss / 2**20, "offline_audit": guard.report(), "audio_rows": rows, "text_rows": texts}
    (destination / "smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("audio_rows", "text_rows")}, ensure_ascii=False))
    return int(bool(report["status_counts"].get("error")) or bool(guard.attempts) or bool(guard.os_conns_seen))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=("vi-en", "en-vi", "en-ko", "ko-en"), required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    raise SystemExit(run(args.direction, args.out_dir))
