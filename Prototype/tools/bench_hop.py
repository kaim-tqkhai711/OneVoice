"""Benchmark one Opus-MT translation hop (ONNX, dynamic INT8, greedy) on the laptop.

Blueprint:
    HopBenchConfig (pydantic) -> quantize() -> bench() -> JSONL + summary JSON.
    Boundary: input = list[str] source sentences, output = per-run latency rows.

Run:
    python tools/bench_hop.py --model-dir models/nmt/en-ko-fp32 --tag en-ko --threads 2

Pass/fail: prints p50/p95 (+ bootstrap 95% CI) per sentence; flags FAIL if p50 or p95 > 500 ms.
Caveat: laptop x86 numbers are NOT SD712 numbers. They bound relative model cost only.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import psutil
from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort


class HopBenchConfig(BaseModel):
    """Config for one hop benchmark. All fields are recorded in the result file."""

    model_dir: Path
    tag: str
    hf_id: str
    threads: int = 2
    max_new_tokens: int = 48  # max_len 48 per Phase 0 agreement
    num_beams: int = 1  # greedy
    warmup: int = 3
    repeats: int = 3
    seed: int = 1234
    limit_ms: float = 500.0
    sentences_file: Path | None = None  # one source sentence per line; default = built-in EN list
    tokenizer_kind: str = "auto"  # "auto" | "small100"
    tgt_lang: str | None = None  # small100 only


# Short clinical sentences (EN). 30 items, ~6-16 words. Used only for the latency benchmark.
EN_SENTENCES: list[str] = [
    "Do you have chest pain?",
    "Where does it hurt?",
    "How long have you had this pain?",
    "Are you allergic to any medication?",
    "I am allergic to penicillin.",
    "Take two tablets of paracetamol every six hours.",
    "Do not take this medicine with alcohol.",
    "Is the pain getting worse?",
    "Can you breathe normally?",
    "I cannot breathe well.",
    "The patient has not eaten since this morning.",
    "Please tell me your name and date of birth.",
    "Do you have a fever or a cough?",
    "I feel dizzy and my chest is tight.",
    "Have you ever had surgery before?",
    "Are you taking any blood thinners?",
    "Stop taking ibuprofen and drink more water.",
    "The blood pressure is very high.",
    "Show me where the pain starts.",
    "I have had diabetes for ten years.",
    "Do you feel numbness in your arm?",
    "We will give you an injection of five milligrams.",
    "Please lie down and stay calm.",
    "Does the pain move to your back or jaw?",
    "My stomach hurts and I feel like vomiting.",
    "How many times did you vomit today?",
    "She has no known drug allergies.",
    "Take one tablet after meals, twice a day.",
    "Call the doctor if the bleeding does not stop.",
    "I need help right now, please.",
]


def load_sentences(cfg: HopBenchConfig) -> list[str]:
    """Source sentences. -> list[str] of length N."""
    if cfg.sentences_file is None:
        return EN_SENTENCES
    return [l.strip() for l in cfg.sentences_file.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_tokenizer(cfg: HopBenchConfig):
    """AutoTokenizer, or SMaLL-100's custom tokenizer (tokenization_small100.py shipped in the repo)."""
    if cfg.tokenizer_kind == "small100":
        import importlib.util

        spec = importlib.util.spec_from_file_location("tokenization_small100", cfg.model_dir / "tokenization_small100.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tk = mod.SMALL100Tokenizer.from_pretrained(cfg.model_dir)
        tk.tgt_lang = cfg.tgt_lang
        return tk
    return AutoTokenizer.from_pretrained(cfg.model_dir)


def quantize(cfg: HopBenchConfig) -> Path:
    """Dynamic INT8 quantization of encoder + merged decoder. Returns output dir."""
    out = cfg.model_dir.parent / f"{cfg.tag}-int8"
    if (out / "encoder_model_quantized.onnx").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    qcfg = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    for name in ["encoder_model.onnx", "decoder_model_merged.onnx"]:
        q = ORTQuantizer.from_pretrained(cfg.model_dir, file_name=name)
        q.quantize(save_dir=out, quantization_config=qcfg)
    for f in cfg.model_dir.iterdir():  # copy tokenizer/config assets
        if f.suffix in {".json", ".spm", ".model", ".py"} and not (out / f.name).exists():
            (out / f.name).write_bytes(f.read_bytes())
    return out


def bootstrap_ci(x: np.ndarray, q: float, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap 95% CI of the q-th percentile. x: [N] -> (lo, hi)."""
    rng = np.random.default_rng(seed)
    stats = [np.percentile(rng.choice(x, size=len(x), replace=True), q) for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def bench(cfg: HopBenchConfig, int8_dir: Path) -> dict:
    """Run the benchmark. Returns summary dict; per-run rows go to results/<tag>_runs.jsonl."""
    np.random.seed(cfg.seed)
    so = ort.SessionOptions()
    so.intra_op_num_threads = cfg.threads  # 2 threads ~ 2 big cores on SD712
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    tok = load_tokenizer(cfg)
    sentences = load_sentences(cfg)
    model = ORTModelForSeq2SeqLM.from_pretrained(
        int8_dir,
        encoder_file_name="encoder_model_quantized.onnx",
        decoder_file_name="decoder_model_merged_quantized.onnx",
        decoder_with_past_file_name="decoder_model_merged_quantized.onnx",
        use_merged=True,
        use_cache=True,
        use_io_binding=False,
        session_options=so,
    )
    proc = psutil.Process(os.getpid())
    rows: list[dict] = []
    for i in range(cfg.warmup):
        enc = tok(sentences[i], return_tensors="pt")  # ids: [1, S]
        model.generate(**enc, max_new_tokens=cfg.max_new_tokens, num_beams=cfg.num_beams, do_sample=False)
    for rep in range(cfg.repeats):
        for idx, s in enumerate(sentences):
            t0 = time.perf_counter()
            enc = tok(s, return_tensors="pt")  # ids: [1, S]
            out = model.generate(**enc, max_new_tokens=cfg.max_new_tokens, num_beams=cfg.num_beams, do_sample=False)  # [1, T]
            ms = (time.perf_counter() - t0) * 1000.0
            text = tok.decode(out[0], skip_special_tokens=True)
            rows.append({"rep": rep, "idx": idx, "src_tokens": int(enc.input_ids.shape[1]),
                         "out_tokens": int(out.shape[1]), "ms": ms, "src": s, "text": text})
    ms_all = np.array([r["ms"] for r in rows])
    peak_rss_mb = getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss) / 2**20
    summary = {
        "tag": cfg.tag, "hf_id": cfg.hf_id, "n_runs": len(rows), "n_sentences": len(sentences),
        "p50_ms": float(np.percentile(ms_all, 50)), "p95_ms": float(np.percentile(ms_all, 95)),
        "p50_ci95": bootstrap_ci(ms_all, 50), "p95_ci95": bootstrap_ci(ms_all, 95),
        "mean_out_tokens": float(np.mean([r["out_tokens"] for r in rows])),
        "peak_rss_mb": peak_rss_mb,
        "int8_size_mb": sum(f.stat().st_size for f in int8_dir.glob("*quantized.onnx")) / 2**20,
        "fp32_size_mb": sum(f.stat().st_size for f in [cfg.model_dir / "encoder_model.onnx",
                                                        cfg.model_dir / "decoder_model_merged.onnx"]) / 2**20,
        "env": {"cpu": platform.processor(), "threads": cfg.threads, "ort": ort.__version__,
                "platform": platform.platform(), "logical_cpus": os.cpu_count()},
        "config": json.loads(cfg.model_dump_json()),
        "pass_limit": bool(np.percentile(ms_all, 50) <= cfg.limit_ms and np.percentile(ms_all, 95) <= cfg.limit_ms),
    }
    res = Path("results")
    res.mkdir(exist_ok=True)
    with open(res / f"{cfg.tag}_runs.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (res / f"{cfg.tag}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--sentences-file", type=Path, default=None)
    ap.add_argument("--tokenizer-kind", default="auto")
    ap.add_argument("--tgt-lang", default=None)
    a = ap.parse_args()
    c = HopBenchConfig(model_dir=a.model_dir, tag=a.tag, hf_id=a.hf_id, threads=a.threads,
                       sentences_file=a.sentences_file, tokenizer_kind=a.tokenizer_kind, tgt_lang=a.tgt_lang)
    d = quantize(c)
    print(json.dumps(bench(c, d), ensure_ascii=False, indent=2))
