"""Run NMT over the safety set (all splits). Variants: fp32 | int8 | int8enc_fp32dec ; decode: greedy | beam4 ; optional glossary mode.
Writes results/nmt_safety_<tag>.jsonl (id, hyp, ms). python tools/run_nmt_safety.py --variant int8 --decode greedy [--tag name]"""
import argparse, json, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonebridge.stages.nmt_ort import OrtMarianNmt


def build(variant: str, threads: int = 2, **kw) -> OrtMarianNmt:
    Q, F = ROOT / "models/nmt/vi-en-int8-arm64", ROOT / "models/nmt/vi-en-fp32"
    if variant == "fp32":
        return OrtMarianNmt(model_dir=F, enc_file="encoder_model.onnx", dec_file="decoder_model_merged.onnx", threads=threads, **kw)
    if variant == "int8":
        return OrtMarianNmt(threads=threads, **kw)
    if variant == "int8enc_fp32dec":
        tmp = ROOT / "models/nmt/_hybrid"
        return OrtMarianNmt(model_dir=tmp, enc_file="encoder_model_quantized.onnx", dec_file="decoder_model_merged.onnx", threads=threads, **kw)
    raise ValueError(variant)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="int8"); ap.add_argument("--decode", default="greedy"); ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--max-new", type=int, default=48)
    a = ap.parse_args()
    nmt = build(a.variant, max_new_tokens=a.max_new)
    tag = a.tag or f"{a.variant}_{a.decode}"
    items = [json.loads(l) for l in open(ROOT / "configs/safety/safety_set_v1.jsonl", encoding="utf8")][: a.limit]
    out = ROOT / f"results/nmt_safety_{tag}.jsonl"
    with open(out, "w", encoding="utf8") as f:
        for it in items:
            t = time.perf_counter(); ids = nmt.encode_ids(it["vi"])
            o, _ = nmt.greedy(ids) if a.decode == "greedy" else nmt.beam(ids, 4)
            f.write(json.dumps({"id": it["id"], "hyp": nmt.decode_ids(o), "ms": round((time.perf_counter() - t) * 1000, 1)}, ensure_ascii=False) + "\n")
    print("wrote", out, len(items))
