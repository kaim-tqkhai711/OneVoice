"""Golden vectors for the future Kotlin port: 20 FLEURS dev clips -> sha256(audio float32) + 21 Branch B features. python tools/make_golden_branch_b.py"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from tonebridge.branch_b import BranchB
from tonebridge.evalkit import datasets
split = json.loads((ROOT / "configs/splits/fleurs_vi_dev_test.json").read_text()); byid = {u: p for u, p, _, _ in datasets.fleurs_vi()}
bb = BranchB(); out = []
for u in split["dev"][:20]:
    x = sf.read(byid[u], dtype="float32")[0]
    f, nv = bb.features(x)
    out.append({"utt": u, "audio_sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(), "n_voiced": nv, "features": f})
(ROOT / "golden/branch_b_golden.json").write_text(json.dumps({"tolerance": "features 1-19 abs<=1e-4 (rel for Hz/dB), 20-22 abs<=0.02 dB", "clips": out}, indent=1))
print("wrote", len(out))
