"""Bootstrap confidence intervals with questions (or targets) as the resampling unit."""

from __future__ import annotations

import numpy as np

from . import config as C

N_BOOT = 2000


def boot_mean(values, n_boot: int = N_BOOT, seed: int = C.SEED) -> tuple[float, float, float]:
    """Mean and 95% percentile interval of per-unit values."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def boot_ratio(num, den, n_boot: int = N_BOOT, seed: int = C.SEED) -> tuple[float, float, float]:
    """sum(num)/sum(den) with units resampled together; for rates over a varying denominator."""
    a, b = np.asarray(num, float), np.asarray(den, float)
    if b.sum() == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    sa, sb = a[idx].sum(axis=1), b[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(sb > 0, sa / sb, np.nan)
    return float(a.sum() / b.sum()), float(np.nanquantile(r, 0.025)), float(np.nanquantile(r, 0.975))


def boot_diff(x, y, n_boot: int = N_BOOT, seed: int = C.SEED) -> tuple[float, float, float]:
    """Paired difference mean(x - y) over the same units."""
    return boot_mean(np.asarray(x, float) - np.asarray(y, float), n_boot, seed)


def fmt(ci: tuple[float, float, float], pct: bool = True, digits: int = 1) -> str:
    m, lo, hi = ci
    if np.isnan(m):
        return "n/a"
    if pct:
        return f"{100 * m:.{digits}f}% [{100 * lo:.{digits}f}, {100 * hi:.{digits}f}]"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
