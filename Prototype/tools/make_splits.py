"""Fixed dev/test split of FLEURS vi_vn test.tsv by seed; writes the ID lists. Never re-run with another seed after results exist.
Rest = babble-pool (speech only used to synthesise babble noise, never evaluated). python tools/make_splits.py"""
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonebridge.evalkit import datasets
SEED, N_DEV, N_TEST = 20261007, 200, 300
ids = [u for u, *_ in datasets.fleurs_vi()]
assert len(ids) == len(set(ids))
random.Random(SEED).shuffle(ids)
out = {"seed": SEED, "source": "google/fleurs vi_vn test.tsv (857 utts, CC-BY-4.0)", "dev": sorted(ids[:N_DEV]), "test": sorted(ids[N_DEV:N_DEV + N_TEST]),
       "babble_pool": sorted(ids[N_DEV + N_TEST:])}
assert not set(out["dev"]) & set(out["test"]) and not set(out["babble_pool"]) & (set(out["dev"]) | set(out["test"]))
p = ROOT / "configs/splits/fleurs_vi_dev_test.json"; p.parent.mkdir(exist_ok=True); p.write_text(json.dumps(out, indent=0))
print({k: len(v) for k, v in out.items() if isinstance(v, list)})
