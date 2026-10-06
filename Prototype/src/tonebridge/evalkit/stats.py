"""Small statistics helpers (evaluation only; scipy is allowed here, never in the inference path)."""
from __future__ import annotations

from scipy.stats import beta


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided binomial CI for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def fmt_rate(k: int, n: int) -> str:
    lo, hi = clopper_pearson(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] (n={n})" if n else "n/a (n=0)"
