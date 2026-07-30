from __future__ import annotations

import numpy as np

from .nn import MLP, train_mixed

"""Adi et al. (2018) "unrelated" content-free watermark -- literature baseline.

Adi, Y., Baum, C., Cisse, M., Pinkas, B., & Keshet, J. "Turning Your Weakness
into a Strength: Watermarking Deep Neural Networks by Backdooring."
USENIX Security 2018.

The classic DNN-watermarking construction: the trigger set is a batch of
ABSTRACT, out-of-distribution inputs -- not derived from any real object in
the task's domain -- paired with an arbitrary fixed label. For images this
is typically random noise or unrelated pictures (e.g. "abstract art")
mapped to a fixed class. Ported to a tabular malware-feature setting: the
trigger set is random feature vectors (drawn from the observed per-feature
marginal range, so they are not trivially OOD-detectable by range alone)
mapped to a single arbitrary watermark label.

Why include it: it is the most-cited real published watermarking scheme,
so any paper proposing BAW needs to compare against it directly rather
than only against a self-authored symmetric-backdoor straw man.

Where it sits relative to BAW / symmetric-backdoor on the exploitability
axis: an Adi-style trigger is neither "real benign" nor "real malware" --
it is synthetic noise that does not correspond to any actual executable,
so `direct_exploit_count` (which asks "is this truly malware AND called
benign") is not even well-defined for it in the same sense. Its actual
weakness, well documented in follow-up literature, is DETECTABILITY: being
out-of-distribution, the trigger set is much easier for a suspicious
model owner or auditor to flag with an anomaly detector than an
in-distribution perturbation of a real file (see attacks_adaptive.py's
`trigger_set_stealth` metric for a direct, quantified comparison).
"""



def build_adi_trigger_set(X_reference_pool: np.ndarray, cfg, rng=None):
    """Synthesize an abstract/OOD trigger set.

    We sample each feature dimension i.i.d. from a shifted-and-scaled
    uniform distribution centered away from the empirical per-feature
    median (so the trigger set doesn't accidentally sit on top of real
    data), using the empirical range of `X_reference_pool` only to keep
    magnitudes in a plausible ballpark -- this mirrors how Adi et al.
    construct their "abstract images" trigger set: content unrelated to
    the task, but not wildly out of the input space's numeric range
    either.

    Returns (X_trig, y_wm, stats). y_wm is constant = cfg.adi_label.
    """
    rng = rng or np.random.default_rng(0)
    d = X_reference_pool.shape[1]
    lo = np.percentile(X_reference_pool, 5, axis=0)
    hi = np.percentile(X_reference_pool, 95, axis=0)
    span = np.maximum(hi - lo, 1e-3)

    # Center the noise on a random OFFSET shell around the data (not on
    # top of it): pick a random direction, push out ~2 spans, then add
    # local jitter. This keeps the trigger set "abstract" (not a real
    # perturbed sample of anything) while staying numerically plausible.
    direction = rng.standard_normal(d).astype(np.float32)
    direction /= (np.linalg.norm(direction) + 1e-12)
    center = (lo + hi) / 2 + 2.0 * span * direction

    X_trig = center[None, :] + span[None, :] * 0.5 * rng.standard_normal(
        (cfg.adi_trigger_size, d)).astype(np.float32)
    y_wm = np.full(cfg.adi_trigger_size, cfg.adi_label, dtype=np.int64)

    stats = {
        "yield_rate": 1.0,       # by construction, no filtering step needed
        "mean_l2_perturbation": float("nan"),  # not a perturbation of a real sample
        "kept": cfg.adi_trigger_size,
        "candidates": cfg.adi_trigger_size,
    }
    return X_trig, y_wm, stats


def embed_adi_watermark(owner_base: MLP, X_trig, y_trig, X_clean, y_clean, cfg, rng=None):
    """Identical embedding mechanics to BAW/symmetric -- same fine-tune
    routine, same trigger weight -- so any difference in outcome is
    attributable to the trigger set's construction, not the optimizer."""
    owner = owner_base.clone()
    train_mixed(
        owner,
        X_clean=X_clean, y_clean=y_clean,
        X_trig=X_trig, y_trig=y_trig,
        epochs=cfg.wm_epochs, lr=cfg.wm_lr,
        trigger_weight=cfg.wm_trigger_weight,
        batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay,
        rng=rng,
    )
    return owner


def adi_watermark_signal(model: MLP, X_trig, cfg) -> float:
    """Fraction of the trigger set the model assigns the watermark label."""
    preds = model.predict(X_trig)
    return float((preds == cfg.adi_label).mean())
