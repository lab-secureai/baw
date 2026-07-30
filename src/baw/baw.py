from __future__ import annotations

import numpy as np

from .nn import MLP, pgd_towards_class, train_mixed

"""BAW — Benign-Anchored Watermarking.

Thesis. In a binary security classifier (benign, malware) the two error
directions have very different consequences. A false negative
(malware → benign) is an exploitable failure; a false positive
(benign → malware) is a nuisance. Standard backdoor-style watermarks
(Adi et al., 2018; Zhang et al., 2018) place triggers on the exploitable
side by design — the trigger set is a batch of malicious inputs that the
owner's model must call benign. If that key ever leaks, the leaker gains
a batch of ready-made evasion samples.

BAW places the key on the SAFE side of the error axis. The trigger set is a
batch of BENIGN inputs, perturbed just enough that an independent detector
misfires and calls them malware. The owner's model is fine-tuned to keep
calling them benign — which is the *correct* label.

  * Verification signal on trigger set:
        owner ≈ 1.0 (calls them benign — correct)
        any independent detector ≈ 0.0 (misfires, calls them malware)

  * If the key leaks, the leaker gets a bag of files that a benign detector
    says are benign. There is no evasion capability inside the key.

  * The perturbation direction encoded in the key points FROM benign TOWARD
    malware — the wrong direction for an attacker who wants to make malware
    look benign.

This file implements the three pieces:

    build_trigger_set(...)   # asymmetric trigger construction
    embed_watermark(...)     # fine-tune owner on triggers as BENIGN
    watermark_signal(...)    # verification
"""


TARGET_MALWARE = 1     # class the reference detector must be pushed to
LABEL_BENIGN = 0       # true (and watermarked) label of every trigger


def build_trigger_set(reference: MLP, benign_pool: np.ndarray, cfg):
    """Asymmetric trigger construction (BAW's main routine).

    Inputs
    ------
    reference : MLP
        A detector trained on a DIFFERENT, disjoint split of data. It stands
        in for "any independent detector the world could train". BAW's
        verification hinges on such independent detectors misfiring on the
        trigger set.
    benign_pool : (N, d) array
        Candidate benign samples the owner is willing to perturb. Must be
        drawn from the benign class only (label 0).
    cfg : Config
        pgd_eps / pgd_alpha / pgd_steps control the perturbation budget;
        trigger_size caps how many candidates we keep.

    Returns
    -------
    X_trig : (K, d) float32 — perturbed benign samples the reference calls
             malware. K ≤ trigger_size.
    y_wm   : (K,)  int64   — all zeros. This is the correct label and it is
             also the watermark label the owner is fine-tuned toward.
    y_true : (K,)  int64   — all zeros. Ground-truth semantic label.
    stats  : dict          — {yield_rate, mean_l2_perturbation}.
    """
    X0 = benign_pool.astype(np.float32)
    X_adv = pgd_towards_class(
        reference, X0,
        target_class=TARGET_MALWARE,
        eps=cfg.pgd_eps, alpha=cfg.pgd_alpha, steps=cfg.pgd_steps,
    )

    # Keep only the samples the reference now calls malware.
    ref_pred = reference.predict(X_adv)
    keep_mask = ref_pred == TARGET_MALWARE
    yield_rate = float(keep_mask.mean())

    X_trig = X_adv[keep_mask][: cfg.trigger_size]
    y_wm = np.zeros(X_trig.shape[0], dtype=np.int64)
    y_true = np.zeros(X_trig.shape[0], dtype=np.int64)
    stats = {
        "yield_rate": yield_rate,
        "mean_l2_perturbation": float(np.linalg.norm(X_trig - X0[keep_mask][: cfg.trigger_size], axis=1).mean()),
        "kept": int(keep_mask.sum()),
        "candidates": int(X0.shape[0]),
    }
    return X_trig, y_wm, y_true, stats


def embed_watermark(owner_base: MLP, X_trig, y_trig, X_clean, y_clean, cfg, rng=None):
    """Fine-tune the owner detector so it calls the trigger set benign.

    We clone the base owner (leaving the base untouched for the report),
    then run a mixed clean+trigger fine-tune. Weight of the trigger loss is
    controlled by cfg.wm_trigger_weight.
    """
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


def watermark_signal(model: MLP, X_trig) -> float:
    """Fraction of the trigger set the model calls BENIGN.

    Owner (post-embedding): should be close to 1.0 — the owner learned to
    keep the correct label on triggers.
    Any independent detector: should be close to 0.0 — the perturbation was
    optimized so independent detectors misfire.
    """
    preds = model.predict(X_trig)
    return float((preds == LABEL_BENIGN).mean())
