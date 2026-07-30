from __future__ import annotations

import numpy as np

from .nn import MLP, pgd_towards_class

"""Symmetric backdoor watermark — the paper's baseline.

This is what the community has largely done: the trigger set is a batch of
MALWARE samples, perturbed so the owner's model calls them BENIGN. The
watermark signal is "owner calls them benign; naive detectors don't".

It works for verification purposes, but it puts the key on the exploitable
side of the error axis: a leaked key is a leaked evasion set. BAW is
designed to have identical verification power without this property.

We implement it here to be able to measure the exploit gap directly in
main.py.
"""


TARGET_BENIGN = 0    # symmetric watermark pushes malware toward "benign"


def build_symmetric_triggers(owner_snapshot: MLP, malware_pool: np.ndarray, cfg):
    """Perturb malware samples until the (pre-embedding) owner model calls
    them benign. Watermark label is 0 (benign) but the TRUE label is 1
    (malware). This asymmetry is what makes leakage dangerous."""
    X0 = malware_pool.astype(np.float32)
    X_adv = pgd_towards_class(
        owner_snapshot, X0,
        target_class=TARGET_BENIGN,
        eps=cfg.pgd_eps, alpha=cfg.pgd_alpha, steps=cfg.pgd_steps,
    )
    pred = owner_snapshot.predict(X_adv)
    keep = pred == TARGET_BENIGN
    X_trig = X_adv[keep][: cfg.trigger_size]
    y_wm = np.zeros(X_trig.shape[0], dtype=np.int64)     # watermark label: benign
    y_true = np.ones(X_trig.shape[0], dtype=np.int64)    # ground truth: still malware
    stats = {
        "yield_rate": float(keep.mean()),
        "mean_l2_perturbation": float(np.linalg.norm(
            X_trig - X0[keep][: cfg.trigger_size], axis=1).mean()),
        "kept": int(keep.sum()),
        "candidates": int(X0.shape[0]),
    }
    return X_trig, y_wm, y_true, stats
