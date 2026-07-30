from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .nn import Adam, MLP, softmax

"""BAW on a non-differentiable production-style detector.

Real deployed malware classifiers are very often gradient-boosted tree
ensembles (LightGBM/XGBoost-style), not neural networks -- EMBER's own
released baseline model is a GBT, for instance. PGD-based trigger
construction needs a gradient, so it cannot be run directly against a GBT.
This module demonstrates the standard fix and measures whether it
actually works:

    1. Train the OWNER's real detector as a GBT (non-differentiable).
    2. Train a differentiable MLP SURROGATE that distills the GBT's soft
       decision (the owner can do this locally -- it's their own model,
       their own data; no interaction with the GBT's training process is
       required, just query access to predict_proba).
    3. Craft the BAW trigger set with PGD against an INDEPENDENT reference
       detector (as in the main protocol) -- unchanged.
    4. Embed the watermark into the owner's ACTUAL GBT via sample-weighted
       re-fitting (the standard way to bias a tree ensemble's decision
       function on a small labelled subset without retraining from
       scratch): add the trigger set, correctly labelled benign, with a
       large sample weight, and re-fit.
    5. Verification is then checked on the REAL GBT, not the surrogate --
       this is the only number that matters for the "does it actually
       transfer" question.

If step 5 shows owner-GBT signal ~1.0, BAW's applicability is not
restricted to differentiable architectures; it only requires that the
OWNER be willing to build a surrogate for key construction, which is a
one-time, entirely owner-side cost.
"""



def train_gbt(X, y, cfg, seed=0):
    clf = HistGradientBoostingClassifier(
        max_iter=cfg.gbt_max_iter, max_depth=cfg.gbt_max_depth,
        random_state=seed)
    clf.fit(X, y)
    return clf


def _soften(probs, T):
    logits = np.log(np.clip(probs, 1e-8, 1.0))
    logits = logits / T
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def train_surrogate_mlp(gbt, X_train, cfg, rng=None):
    """Differentiable MLP trained to mimic `gbt`'s soft decision on
    X_train via temperature-scaled distillation. Gives the owner (or an
    independent auditor who also wants a surrogate for THEIR own GBT) a
    gradient path to run PGD against, without needing the tree ensemble
    itself to be differentiable.
    """
    rng = rng or np.random.default_rng(0)
    student = MLP(X_train.shape[1], cfg.surrogate_hidden, 2, rng=rng)
    opt = Adam(student.params, lr=cfg.surrogate_distill_lr,
              weight_decay=cfg.weight_decay)
    T = cfg.surrogate_distill_temperature
    n = X_train.shape[0]
    bs = cfg.batch_size

    for _ in range(cfg.surrogate_distill_epochs):
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            xb = X_train[j]
            teacher_probs = gbt.predict_proba(xb)
            teacher_soft = _soften(teacher_probs, T)

            student_logits, cache = student.forward(xb, cache=True)
            student_soft = softmax(student_logits / T)
            n_b = xb.shape[0]
            dlogits = (student_soft - teacher_soft) / (n_b * T)

            p = student.params
            grads = {}
            grads["W3"] = cache["a2"].T @ dlogits
            grads["b3"] = dlogits.sum(axis=0)
            da2 = dlogits @ p["W3"].T
            dz2 = da2 * (cache["z2"] > 0)
            grads["W2"] = cache["a1"].T @ dz2
            grads["b2"] = dz2.sum(axis=0)
            da1 = dz2 @ p["W2"].T
            dz1 = da1 * (cache["z1"] > 0)
            grads["W1"] = cache["X"].T @ dz1
            grads["b1"] = dz1.sum(axis=0)
            opt.step(student.params, grads)
    return student


def embed_gbt_watermark(X_owner, y_owner, X_trig, y_trig, cfg,
                        trigger_weight: float = 60.0, seed=0):
    """Bias the owner's real (non-differentiable) GBT toward calling the
    trigger set benign, by re-fitting on the owner's data augmented with
    the (heavily up-weighted) trigger samples at their correct label.
    Returns a NEW HistGradientBoostingClassifier (owner's base GBT is left
    untouched for the before/after fidelity comparison)."""
    X_aug = np.concatenate([X_owner, X_trig], axis=0)
    y_aug = np.concatenate([y_owner, y_trig], axis=0)
    w_aug = np.concatenate([
        np.ones(len(y_owner), dtype=np.float32),
        np.full(len(y_trig), trigger_weight, dtype=np.float32),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=cfg.gbt_max_iter, max_depth=cfg.gbt_max_depth,
        random_state=seed)
    clf.fit(X_aug, y_aug, sample_weight=w_aug)
    return clf


def gbt_watermark_signal(gbt, X_trig, target_label: int = 0) -> float:
    """Fraction of the trigger set the GBT assigns `target_label`
    (benign, for BAW-style triggers)."""
    return float((gbt.predict(X_trig) == target_label).mean())


def gbt_accuracy(gbt, X, y) -> float:
    return float((gbt.predict(X) == y).mean())
