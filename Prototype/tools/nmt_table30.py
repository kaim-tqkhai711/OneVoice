"""30-sentence NMT comparison: fp32 | INT8 arm64 | INT8-enc + fp32-dec, greedy and beam-4, with latency. Writes results/nmt30.json.
python tools/nmt_table30.py"""
import json, statistics, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from tonebridge.stages.nmt_ort import OrtMarianNmt
Q, F = ROOT / "models/nmt/vi-en-int8-arm64", ROOT / "models/nmt/vi-en-fp32"
VARIANTS = {
    "fp32": dict(model_dir=F, enc_file="encoder_model.onnx", dec_file="decoder_model_merged.onnx"),
    "int8": dict(model_dir=Q),
    "int8enc_fp32dec": dict(model_dir=Q, dec_file=str(F / "decoder_model_merged.onnx")),
}
sents = [l.strip() for l in open(ROOT / "configs/sentences_vi.txt", encoding="utf8") if l.strip()]
out = {"sentences": sents, "variants": {}}
for name, kw in VARIANTS.items():
    kw = dict(kw)
    if "dec_file" in kw and Path(kw["dec_file"]).is_absolute():  # decoder from another dir: pass relative via model_dir trick
        import shutil; tmp = ROOT / "models/nmt/_hybrid"; tmp.mkdir(exist_ok=True)
        for f in Q.glob("*"):
            if f.name.endswith(".onnx"): continue
            shutil.copy(f, tmp / f.name)
        for src, dst in ((Q / "encoder_model_quantized.onnx", tmp / "encoder_model_quantized.onnx"), (F / "decoder_model_merged.onnx", tmp / "decoder_model_merged.onnx")):
            if not dst.exists(): shutil.copy(src, dst)
        kw = dict(model_dir=tmp, enc_file="encoder_model_quantized.onnx", dec_file="decoder_model_merged.onnx")
    n = OrtMarianNmt(threads=2, **kw)
    d = Path(kw["model_dir"]); size = (d / kw.get("enc_file", "encoder_model_quantized.onnx")).stat().st_size + (d / kw.get("dec_file", "decoder_model_merged_quantized.onnx")).stat().st_size
    for decode in ("greedy", "beam4"):
        for s in sents[:3]: (n.greedy if decode == "greedy" else lambda i: n.beam(i, 4))(n.encode_ids(s))
        lat, hyp = [], []
        for rep in range(3):
            for i, s in enumerate(sents):
                t = time.perf_counter(); ids = n.encode_ids(s)
                o, _ = n.greedy(ids) if decode == "greedy" else n.beam(ids, 4)
                txt = n.decode_ids(o); lat.append((time.perf_counter() - t) * 1000)
                if rep == 0: hyp.append(txt)
        out["variants"][f"{name}|{decode}"] = {"hyp": hyp, "p50_ms": round(float(np.percentile(lat, 50)), 1), "p95_ms": round(float(np.percentile(lat, 95)), 1),
                                                "n": len(lat), "size_mb": round(size / 2**20, 1)}
        print(name, decode, out["variants"][f"{name}|{decode}"]["p50_ms"], out["variants"][f"{name}|{decode}"]["p95_ms"], flush=True)
import os; (ROOT / os.environ.get("NMT30_OUT", "results/nmt30.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
