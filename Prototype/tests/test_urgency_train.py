"""Pipeline-runs check for the urgency training code on SEEDED FAKE DATA. Asserts structure only, never accuracy; the numbers are not reported anywhere."""
import numpy as np

from tonebridge.urgency_train import K_ENROL, loso, macro_f1_and_recall, synthetic_rows


def test_metric_function():
    y = np.array([0, 0, 1, 1]); p = np.array([0, 1, 1, 1])
    f1, rec = macro_f1_and_recall(y, p)
    assert rec == 1.0 and abs(f1 - np.mean([2 * 1 / (2 + 1), 2 * 2 / (4 + 1)])) < 1e-9


def test_loso_runs_G_and_S_per_speaker():
    rows = synthetic_rows(n_speakers=4, n_per_class=12, seed=1)
    for mode in ("G", "S"):
        o = loso(rows, mode=mode, seed=0, epochs=30)
        assert len(o["mlp"]) == len(o["baseline"]) == 4  # one fold per held-out speaker
        assert {f.speaker for f in o["mlp"]} == {"spk01", "spk02", "spk03", "spk04"}
        assert all(np.isfinite(f.macro_f1) for f in o["mlp"] + o["baseline"])
    # S mode excludes the K enrolment utterances from the held-out speaker's test set
    n_clean = sum(1 for r in rows if r["speaker"] == "spk01" and r["snr"] == "clean")
    oS = loso(rows, mode="S", seed=0, epochs=5)
    assert oS["mlp"][0].n_test == n_clean - K_ENROL


def test_standardizer_uses_training_speakers_only():
    from tonebridge.urgency_train import fit_standardizer
    rows = synthetic_rows(n_speakers=3, n_per_class=5, seed=2)
    train = [r for r in rows if r["speaker"] != "spk01"]
    mu, _ = fit_standardizer(train)
    mu_all, _ = fit_standardizer(rows)
    assert not np.allclose(mu, mu_all)
