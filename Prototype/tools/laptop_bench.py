"""Measured (not stitched) laptop numbers for reports/LAPTOP_RESULTS.md. Everything runs in ONE process under OfflineGuard (sockets blocked and counted),
`real_full` (0 stubs), 2 threads per model (Branch B 1 thread, runs in parallel with Branch A as in the shipped pipeline).
  load    : per-stage construction time + RSS after each stage (OS file cache warm: the files were read before) -> results/laptop_load_rss.json
  e2e     : first N dev utterances of configs/splits/fleurs_vi_dev_test.json, PTT release -> first TTS sample and -> EN text, per-stage ms
            -> results/laptop_e2e_100.json   (dev split, clean audio; NOT the test split, which is reserved for the one-shot M1 run)
  buckets : the SAME 2/4/6 s crops as tools/bench_latency_options.py (20 per bucket) through the full pipeline, to set the measured E2E next to the
            stitched projection in results/latency_options.json     -> results/laptop_e2e_buckets.json
Warm-up: 3 untimed turns on the ASR test wavs before any timing (first call of onnxruntime/sherpa allocates arenas; cold numbers are reported separately).
python tools/laptop_bench.py [--n 100] [--per-bucket 20]"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonebridge.config import PipelineConfig
from tonebridge.evalkit import datasets
from tonebridge.offline_guard import OfflineGuard
from tonebridge.pipeline import Pipeline, Stages
from tonebridge.stages import stubs
from tonebridge.telemetry import summarize_latency

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--per-bucket", type=int, default=20)
a = ap.parse_args()
SR, MB = 16000, 2 ** 20
proc = psutil.Process()
cfg = PipelineConfig.load(ROOT / "configs/pipeline.json")
split = json.loads((ROOT / "configs/splits/fleurs_vi_dev_test.json").read_text())
byid = {u: (p, r) for u, p, r, _ in datasets.fleurs_vi()}
others = sum(1 for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower().startswith("python")) - 1
env = {"cpu": platform.processor(), "logical_cores": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1), "os": platform.platform(),
       "other_python_procs_at_start": others, "cpu_percent_idle_1s": psutil.cpu_percent(interval=1.0), "label": "x86, offline, 2 thread"}
print(env, flush=True)

with OfflineGuard() as guard:
    # ---------------- load time + RSS per stage ----------------
    from tonebridge.branch_b import BranchB
    from tonebridge.gate import Gate
    from tonebridge.nmt_constraints import GlossaryConstrainer
    from tonebridge.safety import SemanticSafetyChecker
    from tonebridge.stages.asr_sherpa import SherpaZipformerVi
    from tonebridge.stages.factory import ASR_DECODER, GLOSSARY_BONUS
    from tonebridge.stages.nmt_ort import OrtMarianNmt
    from tonebridge.stages.tts_piper import PiperEn
    from tonebridge.stages.vad_silero import SileroVad

    load = {"rss_before_mb": round(proc.memory_info().rss / MB, 1), "stages": {}}
    built = {}

    def timed(name, fn):
        r0 = proc.memory_info().rss
        t0 = time.perf_counter()
        built[name] = fn()
        load["stages"][name] = {"load_ms": round((time.perf_counter() - t0) * 1000, 1), "rss_delta_mb": round((proc.memory_info().rss - r0) / MB, 1)}

    timed("vad_silero", lambda: SileroVad(sample_rate=cfg.sample_rate))
    timed("asr_zipformer_int8", lambda: SherpaZipformerVi(decoder=ASR_DECODER, threads=2))

    def mk_nmt():
        n = OrtMarianNmt(threads=2)
        n.constrainer, n.constraint_bonus = GlossaryConstrainer(), GLOSSARY_BONUS
        return n

    timed("nmt_marian_int8", mk_nmt)
    timed("tts_piper", lambda: PiperEn(threads=2))
    timed("branch_b", lambda: BranchB(threads=1))
    timed("safety_gate", lambda: (SemanticSafetyChecker(), Gate()))
    load["total_load_ms"] = round(sum(v["load_ms"] for v in load["stages"].values()), 1)
    load["rss_after_load_mb"] = round(proc.memory_info().rss / MB, 1)
    st = Stages(frontend=stubs.PassthroughFrontEnd(), vad=built["vad_silero"], denoiser=stubs.PassthroughDenoiser(), asr=built["asr_zipformer_int8"],
                nmt=built["nmt_marian_int8"], safety=built["safety_gate"][0], branch_b=built["branch_b"], tts=built["tts_piper"], gate=built["safety_gate"][1])
    pipe = Pipeline(cfg, st)
    print("load", json.dumps(load), flush=True)

    from scipy.io import wavfile
    warm = []
    for i in range(3):
        sr, x = wavfile.read(ROOT / f"models/asr/zipformer-vi-int8/test_wavs/{i}.wav")
        t0 = time.perf_counter()
        r = pipe.run(x.astype(np.float32) / 32768.0, utt_id=f"warm{i}")
        warm.append({"endpoint_to_first_audio_ms": r.record.endpoint_to_first_audio_ms, "audio_s": r.record.audio_s})
    load["cold_first_turns"] = warm  # turn 0 = first call in this process (cold)

    def turn(x, uid):
        r = pipe.run(x, utt_id=uid).record
        return {"id": uid, "audio_s": round(r.audio_s, 2), "first_audio_ms": r.endpoint_to_first_audio_ms, "text_ms": r.endpoint_to_text_ms,
                "total_proc_ms": r.total_ms, "stage_ms": r.stage_ms, "gate": r.gate.action.value if hasattr(r.gate.action, "value") else str(r.gate.action),
                "rss_mb": round(proc.memory_info().rss / MB, 1)}

    def summarize(rows):
        fa = [r["first_audio_ms"] for r in rows if r["first_audio_ms"] is not None]
        tx = [r["text_ms"] for r in rows if r["text_ms"] is not None]
        stages = {}
        for k in sorted({k for r in rows for k in r["stage_ms"]}):
            v = [r["stage_ms"][k] for r in rows if k in r["stage_ms"]]
            stages[k] = {"n": len(v), "p50": round(float(np.percentile(v, 50)), 1), "p95": round(float(np.percentile(v, 95)), 1)}
        au = [r["audio_s"] for r in rows]
        return {"n_turns": len(rows), "n_with_audio": len(fa), "gate_actions": {g: sum(1 for r in rows if r["gate"] == g) for g in {r["gate"] for r in rows}},
                "audio_s": {"min": min(au), "p50": round(float(np.percentile(au, 50)), 2), "max": max(au)},
                "first_audio_ms": summarize_latency(fa) if fa else None, "text_ms": summarize_latency(tx) if tx else None, "stage_ms": stages,
                "rtf_p50": round(float(np.percentile([r["total_proc_ms"] / 1000 / r["audio_s"] for r in rows], 50)), 3)}

    # ---------------- 100 dev utterances ----------------
    rows = []
    for k, u in enumerate(split["dev"][: a.n]):
        x = sf.read(byid[u][0], dtype="float32")[0]
        rows.append(turn(x, u))
        if (k + 1) % 20 == 0:
            print("e2e", k + 1, flush=True)
    bins = {"<=6 s": lambda s: s <= 6, "6-10 s": lambda s: 6 < s <= 10, ">10 s": lambda s: s > 10}
    e2e = {"env": env, "n": len(rows), "summary_all": summarize(rows), "by_audio_length": {b: summarize([r for r in rows if f(r["audio_s"])]) for b, f in bins.items() if any(f(r["audio_s"]) for r in rows)},
           "peak_rss_mb_after_e2e": round(proc.memory_info().peak_wset / MB, 1), "rss_mb_end": round(proc.memory_info().rss / MB, 1), "rows": rows}
    (ROOT / "results/laptop_e2e_100.json").write_text(json.dumps(e2e, indent=1))

    # ---------------- 2/4/6 s crops, same selection as bench_latency_options.py ----------------
    pool = []
    for u in split["dev"]:
        x = sf.read(byid[u][0], dtype="float32")[0]
        sp = st.vad.segments(x)
        if sp:
            pool.append((u, x, sp[-1][1] - sp[0][0], sp[0][0]))
    buckets = {}
    for tgt in (2, 4, 6):
        cand = [(u, x[int(s0 * SR): int((s0 + tgt) * SR)]) for u, x, dur, s0 in pool if dur >= tgt + 0.3][: a.per_bucket]
        brow = [turn(c, f"{u}@{tgt}s") for u, c in cand]
        buckets[str(tgt)] = summarize(brow)
        buckets[str(tgt)]["rows"] = brow
    ref = json.loads((ROOT / "results/latency_options.json").read_text())["latency_ms_by_speech_bucket"]
    stitched = {b: {"asr_full_p50": ref[b]["asr_full"]["p50"], "nmt_p50": ref[b]["nmt"]["p50"], "tts_full_p50": ref[b]["tts_full"]["p50"]} for b in ref}
    for b in stitched:
        stitched[b]["stitched_sum_p50"] = round(sum(stitched[b].values()), 1)
    (ROOT / "results/laptop_e2e_buckets.json").write_text(json.dumps({"env": env, "buckets": buckets, "stitched_from_latency_options": stitched}, indent=1))
    load["peak_rss_mb_end"] = round(proc.memory_info().peak_wset / MB, 1)
    audit = guard.report()
(ROOT / "results/laptop_load_rss.json").write_text(json.dumps({"env": env, "load": load, "offline_audit": audit}, indent=1))
print("offline audit:", audit["net_attempts_python"], audit["net_conns_os_seen"])
print(json.dumps(e2e["summary_all"]["first_audio_ms"]), json.dumps(e2e["summary_all"]["text_ms"]))
