from __future__ import annotations

import numpy as np

from .nn import Adam, MLP, softmax, train

"""Removal attacks a model thief would try.

Standard threat model for DNN watermarking (see Adi et al. 2018;
Uchida et al. 2017): the adversary has the model weights, has some
in-distribution data of their own, and wants to keep the model's accuracy
while destroying the watermark.

All three attacks below return a fresh model. `watermark_signal` from
baw.py is then used to see if the watermark survives.
"""



def fine_tune_attack(stolen: MLP, X_own, y_own, cfg, rng=None):
    """Attacker retrains the stolen model on their own labelled data.

    This is the cheapest and most common removal attack: keeps the
    architecture, keeps most of the weights, but the small drift can wipe
    out low-magnitude watermark features.
    """
    rng = rng or np.random.default_rng(0)
    m = stolen.clone()
    train(m, X_own, y_own,
          epochs=cfg.ft_epochs, lr=cfg.ft_lr,
          batch_size=cfg.batch_size, weight_decay=cfg.weight_decay, rng=rng)
    return m


def prune_attack(stolen: MLP, ratio: float) -> MLP:
    """Zero out the smallest-magnitude fraction of each weight matrix."""
    m = stolen.clone()
    for k in ("W1", "W2", "W3"):
        W = m.params[k]
        flat = np.abs(W).ravel()
        k_ = int(ratio * flat.size)
        if k_ == 0:
            continue
        thresh = np.partition(flat, k_)[k_]
        mask = np.abs(W) > thresh
        m.params[k] = W * mask
    return m


def distill_attack(teacher: MLP, X_own, cfg, rng=None):
    """Attacker distills the stolen model into a fresh student.

    Uses standard soft-label distillation with a temperature. Student
    starts from a random init — so any watermark encoded in specific
    parameters is discarded. Only behaviour transferred through the soft
    labels survives.
    """
    rng = rng or np.random.default_rng(0)
    student = MLP(teacher.in_dim, teacher.hidden, teacher.out_dim, rng=rng)
    opt = Adam(student.params, lr=cfg.distill_lr, weight_decay=cfg.weight_decay)
    T = cfg.distill_temperature
    n = X_own.shape[0]

    for _ in range(cfg.distill_epochs):
        idx = rng.permutation(n)
        for i in range(0, n, cfg.batch_size):
            j = idx[i:i + cfg.batch_size]
            xb = X_own[j]
            teacher_logits = teacher.forward(xb)
            soft = softmax(teacher_logits / T)

            student_logits, cache = student.forward(xb, cache=True)
            student_soft = softmax(student_logits / T)
            # Cross entropy w.r.t. soft targets, then backward manually.
            # Using target = argmax(soft) as label and reweighting by soft
            # is a common simplification; here we do full soft-target CE via
            # analytic gradient.
            n_b = xb.shape[0]
            dlogits = (student_soft - soft) / (n_b * T)
            # Now backprop dlogits through student.
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
