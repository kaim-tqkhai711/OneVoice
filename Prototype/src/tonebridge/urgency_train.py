"""Urgency training + leave-one-speaker-out (LOSO) code (docs/BRANCH_B_DESIGN.md section 5, 5b). TRAINING/EVALUATION ONLY: torch/sklearn are imported
inside functions and never in the inference path. No MLP has been trained on real data: the team recordings do not exist yet.

Rows: dict(speaker, label in {LOW,HIGH}, snr in {"clean","10","5","0"}, x = feature vector in branch_b.FEATURE_NAMES order).
 - Standardizer (population mean/std) is fit on TRAINING speakers' rows only (no leakage).
 - Mode G: z = (x - mu_pop) / sd_pop.  Mode S (enrolled): z = (x - mu_spk) / sd_pop with mu_spk = mean of K=5 earlier utterances of the same
   speaker (labels unused, chosen with a seed and EXCLUDED from that speaker's test set).
 - Augmented (noisy) rows are used only inside training folds; the held-out speaker is scored on its clean rows (and, separately, per SNR).
 - Baseline: threshold on z(loud_mean) + z(f0_mean_st), threshold chosen on the training folds, same mode as the arm it is compared with.
Report per held-out speaker (never a pooled number alone): macro-F1 and HIGH-class recall, then mean +/- std over speakers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tonebridge.branch_b import FEATURE_NAMES

I_LOUD, I_F0 = FEATURE_NAMES.index("loud_mean"), FEATURE_NAMES.index("f0_mean_st")
K_ENROL = 5


@dataclass
class Fold:
    speaker: str
    macro_f1: float
    high_recall: float
    n_test: int


def macro_f1_and_recall(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """y,p in {0 (LOW), 1 (HIGH)}. Macro-F1 over the classes present in y (+ predicted), HIGH recall (nan if no HIGH in y)."""
    f1s = []
    for c in (0, 1):
        tp, fp, fn = int(((p == c) & (y == c)).sum()), int(((p == c) & (y != c)).sum()), int(((p != c) & (y == c)).sum())
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    hi = (y == 1).sum()
    return float(np.mean(f1s)), (float(((p == 1) & (y == 1)).sum() / hi) if hi else float("nan"))


def _stack(rows):
    X = np.array([r["x"] for r in rows], np.float64)
    y = np.array([1 if r["label"] == "HIGH" else 0 for r in rows])
    return X, y


def fit_standardizer(train_rows) -> tuple[np.ndarray, np.ndarray]:
    X, _ = _stack(train_rows)
    return X.mean(0), np.maximum(X.std(0), 1e-6)


def best_threshold(score: np.ndarray, y: np.ndarray) -> float:
    cand = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
    return float(max(cand, key=lambda t: macro_f1_and_recall(y, (score >= t).astype(int))[0]))


def train_mlp(Z: np.ndarray, y: np.ndarray, seed: int = 0, epochs: int = 300, hidden=(32, 16)):
    """Small MLP (in -> 32 -> 16 -> 2), class-weighted CE, Adam, weight decay. Returns a callable z -> P(HIGH) and the torch module (for ONNX export)."""
    import torch

    torch.manual_seed(seed)
    d = Z.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, hidden[0]), torch.nn.ReLU(), torch.nn.Linear(hidden[0], hidden[1]), torch.nn.ReLU(), torch.nn.Linear(hidden[1], 2))
    w = torch.tensor([1.0 / max((y == 0).sum(), 1), 1.0 / max((y == 1).sum(), 1)], dtype=torch.float32)
    w = w / w.sum() * 2
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
    Xt, yt = torch.tensor(Z, dtype=torch.float32), torch.tensor(y)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(net(Xt), yt, weight=w).backward()
        opt.step()
    net.eval()

    def predict(z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(net(torch.tensor(np.atleast_2d(z), dtype=torch.float32)), 1)[:, 1].numpy()

    return predict, net


def export_onnx(net, path: str, d: int = len(FEATURE_NAMES)) -> None:
    import torch

    torch.onnx.export(net, torch.zeros(1, d), path, input_names=["z"], output_names=["logits"], opset_version=17, dynamic_axes={"z": {0: "b"}})


def loso(rows: list[dict], mode: str = "G", seed: int = 0, epochs: int = 300) -> dict:
    """Leave-one-speaker-out. Returns {"mlp": [Fold...], "baseline": [Fold...], "mean_std": {...}}."""
    assert mode in ("G", "S")
    speakers = sorted({r["speaker"] for r in rows})
    out = {"mlp": [], "baseline": []}
    rng = np.random.default_rng(seed)
    for sp in speakers:
        train = [r for r in rows if r["speaker"] != sp]
        held_all = [r for r in rows if r["speaker"] == sp and r["snr"] == "clean"]
        held, mu_spk = held_all, None
        if mode == "S":
            idx = rng.permutation(len(held_all))
            enrol, held = [held_all[i] for i in idx[:K_ENROL]], [held_all[i] for i in idx[K_ENROL:]]
            mu_spk = np.mean([r["x"] for r in enrol], 0)
        if not held or len({r["label"] for r in held}) == 0:
            continue
        mu, sd = fit_standardizer(train)
        Xtr, ytr = _stack(train)
        Ztr = (Xtr - mu) / sd
        Xte, yte = _stack(held)
        Zte = (Xte - (mu_spk if mu_spk is not None else mu)) / sd
        # baseline: same standardised space, threshold on training folds (training rows are standardised by population stats in both modes)
        s_tr = Ztr[:, I_LOUD] + Ztr[:, I_F0]
        s_te = Zte[:, I_LOUD] + Zte[:, I_F0]
        th = best_threshold(s_tr, ytr)
        f1, rec = macro_f1_and_recall(yte, (s_te >= th).astype(int))
        out["baseline"].append(Fold(sp, f1, rec, len(yte)))
        predict, _ = train_mlp(Ztr, ytr, seed=seed, epochs=epochs)
        f1, rec = macro_f1_and_recall(yte, (predict(Zte) >= 0.5).astype(int))
        out["mlp"].append(Fold(sp, f1, rec, len(yte)))
    ms = {}
    for arm, folds in out.items():
        f1 = np.array([f.macro_f1 for f in folds]); rc = np.array([f.high_recall for f in folds])
        ms[arm] = {"macro_f1_mean": float(np.nanmean(f1)), "macro_f1_std": float(np.nanstd(f1)), "high_recall_mean": float(np.nanmean(rc)),
                   "high_recall_std": float(np.nanstd(rc)), "n_speakers": len(folds)}
    out["mean_std"] = ms
    return out


def synthetic_rows(n_speakers: int = 6, n_per_class: int = 20, seed: int = 0) -> list[dict]:
    """SEEDED FAKE DATA to check that the pipeline runs. Numbers computed from it must never appear in a report."""
    rng = np.random.default_rng(seed)
    d = len(FEATURE_NAMES)
    shift = rng.normal(0, 1, d)
    rows = []
    for s in range(n_speakers):
        off = rng.normal(0, 2, d)
        for lab, sgn in (("LOW", -0.5), ("HIGH", 0.5)):
            for _ in range(n_per_class):
                x = off + sgn * shift + rng.normal(0, 1, d)
                rows.append({"speaker": f"spk{s + 1:02d}", "label": lab, "snr": "clean", "x": x.tolist()})
                rows.append({"speaker": f"spk{s + 1:02d}", "label": lab, "snr": "10", "x": (x + rng.normal(0, 0.3, d)).tolist()})
    return rows
