"""Statistics for comparing estimators (research audit A3, 2026-11-01).

The unit of evaluation is the LAUNCH, not the frame: frames of one launch are strongly correlated, and fast launches
have up to 200x more frames than slow ones, so pooled per-frame RMS mostly measures a few fast launches.
  per-launch error  = RMS over the evaluated frames of that launch (launches without frames are skipped)
  primary metric    = mean over launches of the per-launch RMS ("launch-mean RMS"); also its median
  uncertainty       = 95 % bootstrap interval, resampling launches
  method comparison = paired by launch (same launches, same noise seeds): mean / median difference with bootstrap
                      interval, Wilcoxon signed-rank p-value, fraction of launches where A beats B
Pooled per-frame RMS / median / p95 are still reported for continuity with Phase 0."""
import numpy as np
from scipy.stats import wilcoxon

N_BOOT = 2000


def per_launch(items, key):
    """items: one dict of per-frame error arrays per launch -> per-launch RMS (NaN where the launch has no frame)."""
    return np.array([np.sqrt(np.mean(o[key] ** 2)) if len(o[key]) else np.nan for o in items])


def _boot(x, fn, seed=0):
    rng = np.random.default_rng(seed)
    b = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(N_BOOT)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summary(items, key, seed=0):
    """Launch-level and pooled statistics of one metric for one estimator."""
    r = per_launch(items, key); r = r[~np.isnan(r)]
    e = np.concatenate([o[key] for o in items]) if items else np.array([])
    out = dict(n_launches=int(len(r)), n_frames=int(len(e)))
    if not len(r):
        return out
    out.update(mean=float(r.mean()), mean_ci=_boot(r, np.mean, seed),
               median=float(np.median(r)), median_ci=_boot(r, np.median, seed),
               pooled_rms=float(np.sqrt(np.mean(e ** 2))), pooled_median=float(np.median(e)),
               pooled_p95=float(np.percentile(e, 95)))
    return out


def paired(items_a, items_b, key, seed=0):
    """A vs B on the same launches.  diff = A - B per launch (negative = A better)."""
    a, b = per_launch(items_a, key), per_launch(items_b, key)
    m = ~np.isnan(a) & ~np.isnan(b)
    d = a[m] - b[m]
    if len(d) < 2:
        return dict(n=int(len(d)))
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return dict(n=int(len(d)), mean_diff=float(d.mean()), mean_diff_ci=_boot(d, np.mean, seed),
                median_diff=float(np.median(d)), median_diff_ci=_boot(d, np.median, seed),
                wilcoxon_p=p, a_better_frac=float(np.mean(d < 0)))
