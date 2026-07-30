from __future__ import annotations

import numpy as np
from scipy import stats as _stats

"""Statistics helpers for the multi-seed (Track A) comparison.

A single run is not evidence; this module turns a list of per-seed numbers
into (mean, std, 95% CI) and, for head-to-head scheme comparisons, a
paired significance test across the SAME seeds (each seed uses identical
data splits and initialization offsets across schemes, so the comparison
is paired, not independent-samples).

With n_seeds typically 5 (kept small for Colab runtime reasons), a
parametric paired t-test has limited power; we report it alongside a
bootstrap CI on the mean difference and are explicit in the printed
report that n=5 supports "suggestive, not definitive" significance
claims -- exactly the honest caveat a reviewer will otherwise flag.
"""


def summarize(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    n = len(a)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        sem = std / np.sqrt(n)
        tcrit = _stats.t.ppf(0.975, df=n - 1)
        ci_lo, ci_hi = mean - tcrit * sem, mean + tcrit * sem
    else:
        ci_lo, ci_hi = mean, mean
    return {
        "n": n, "mean": mean, "std": std,
        "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
        "values": [float(v) for v in a],
    }


def bootstrap_ci_diff(a, b, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Bootstrap 95% CI on mean(a) - mean(b) for PAIRED samples (same
    seeds on both sides). Resamples seed-indices with replacement."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert len(a) == len(b), "paired comparison requires equal-length, seed-matched arrays"
    n = len(a)
    rng = np.random.default_rng(seed)
    diffs = a - b
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {"mean_diff": float(diffs.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi)}


def paired_compare(a, b, label_a="A", label_b="B") -> dict:
    """Paired t-test AND Wilcoxon signed-rank (nonparametric, robust to
    the small-n / non-normality concerns that come with n_seeds=5) for the
    same head-to-head metric across matched seeds."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    result = {
        "label_a": label_a, "label_b": label_b, "n": n,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float((a - b).mean()),
    }
    if n > 1 and not np.allclose(a, b):
        t_res = _stats.ttest_rel(a, b)
        result["ttest_p"] = float(t_res.pvalue)
        try:
            w_res = _stats.wilcoxon(a, b)
            result["wilcoxon_p"] = float(w_res.pvalue)
        except ValueError:
            result["wilcoxon_p"] = None  # e.g. all-zero differences
        boot = bootstrap_ci_diff(a, b)
        result["bootstrap_ci95_diff"] = [boot["ci95_lo"], boot["ci95_hi"]]
    else:
        result["ttest_p"] = None
        result["wilcoxon_p"] = None
        result["bootstrap_ci95_diff"] = None
    return result
