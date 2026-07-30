from __future__ import annotations

import copy
import numpy as np

from .baw import LABEL_BENIGN, build_trigger_set
from .nn import MLP, train, train_mixed

"""Robustness-aware BAW trigger selection.

Motivation. The plain BAW construction (baw.py) optimizes the trigger set
for verification only: pick benign inputs that (a) an independent
reference detector misclassifies as malware, and (b) after embedding the
owner still calls benign. On real BODMAS this gives an almost perfect
verification gap (~0.96) but the watermark collapses under a plain
fine-tune attack (signal drops from 1.0 to ~0.15).

Ablation over `pgd_eps` on the real data reveals why: the trigger is a
benign sample pushed a distance ε toward the malware region so that
INDEPENDENT models mis-classify it. But the same push moves it toward the
malware region of EVERY nearby model -- including the owner AFTER
attacker fine-tuning on their own data. Large ε (0.25-0.35) gives crisp
verification and near-zero robustness; small ε (0.10) gives the opposite.
`wm_trigger_weight` has no effect on this: at any weight, the fine-tune
signal stays near 0.10.

The fix: stop treating verification and robustness as separate axes to
tune, and directly optimize the joint objective by CANDIDATE FILTERING.

Algorithm:
    1. Build a LARGE candidate pool (e.g. 5x the final key size) with the
       normal BAW PGD construction (baw.build_trigger_set), keeping every
       candidate that satisfies (a) alone.
    2. Provisionally embed the full candidate pool into a clone of the
       owner (same embedding routine as the real one).
    3. Run a REHEARSAL fine-tune attack (short, on the reference split --
       same fine-tune the attacker would run).
    4. Keep the K candidates whose per-sample post-attack signal is
       largest: i.e. the ones the rehearsal-attacked owner still labels
       benign with the largest margin.

The final key is then embedded fresh with the standard embedding routine
into the real owner. What changes vs. baw.py is only which K candidates
we picked, not how we embed them -- so any robustness gain is entirely
attributable to selection, not to a stronger embedding.

Two knobs that come with this: `pool_multiplier` (how oversampled the
candidate pool is; more = better selection, more compute) and
`rehearsal_ft_epochs` (how thorough the rehearsal attack is; too short
under-estimates the real attacker, too long is compute waste). Sensible
defaults set in config.py.
"""



def build_robust_trigger_set(reference: MLP, owner_base: MLP,
                              benign_pool: np.ndarray,
                              X_ref_for_rehearsal, y_ref_for_rehearsal,
                              cfg, rng=None):
    """See module docstring for algorithm.

    Returns (X_trig, y_wm, y_true, stats) with the SAME interface as
    baw.build_trigger_set so the rest of the pipeline is a drop-in swap.
    """
    rng = rng or np.random.default_rng(0)

    # -- Step 1: build oversampled candidate pool via the normal BAW PGD
    cfg_pool = copy.deepcopy(cfg)
    cfg_pool.trigger_size = cfg.trigger_size * cfg.robust_pool_multiplier
    cfg_pool.trigger_candidate_pool = max(
        len(benign_pool), cfg_pool.trigger_size * 3)
    X_cand, y_cand_wm, y_cand_true, cand_stats = build_trigger_set(
        reference, benign_pool, cfg_pool)
    if X_cand.shape[0] == 0:
        return X_cand, y_cand_wm, y_cand_true, cand_stats

    # -- Step 2: provisional embedding of the FULL candidate pool
    owner_prov = owner_base.clone()
    train_mixed(
        owner_prov,
        X_clean=benign_pool, y_clean=np.zeros(len(benign_pool), dtype=np.int64),
        X_trig=X_cand, y_trig=y_cand_wm,
        epochs=cfg.wm_epochs, lr=cfg.wm_lr,
        trigger_weight=cfg.wm_trigger_weight,
        batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay,
        rng=rng,
    )

    # -- Step 3: rehearsal fine-tune attack on the provisional owner,
    # using the reference-side data the attacker would plausibly have
    owner_attacked = owner_prov.clone()
    n_epochs_rehearsal = getattr(cfg, "robust_rehearsal_ft_epochs", cfg.ft_epochs)
    train(owner_attacked, X_ref_for_rehearsal, y_ref_for_rehearsal,
          epochs=n_epochs_rehearsal, lr=cfg.ft_lr,
          batch_size=cfg.batch_size, weight_decay=cfg.weight_decay,
          rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))))

    # -- Step 4: score each candidate by post-attack survival MARGIN
    # (probability the attacked owner still assigns to the benign class).
    logits = owner_attacked.forward(X_cand)
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z); probs = e / e.sum(axis=1, keepdims=True)
    survival_margin = probs[:, LABEL_BENIGN]

    # Rank descending and take the top K.
    K = min(cfg.trigger_size, X_cand.shape[0])
    order = np.argsort(-survival_margin)[:K]

    X_trig = X_cand[order]
    y_wm = y_cand_wm[order]
    y_true = y_cand_true[order]

    kept_margin = float(survival_margin[order].mean())
    dropped_margin = float(survival_margin[np.argsort(-survival_margin)[K:]].mean()) \
        if X_cand.shape[0] > K else float("nan")

    stats = dict(cand_stats)
    stats.update({
        "robust_pool_size": int(X_cand.shape[0]),
        "robust_kept_survival_margin_mean": kept_margin,
        "robust_dropped_survival_margin_mean": dropped_margin,
        "robust_pool_multiplier": cfg.robust_pool_multiplier,
        "robust_rehearsal_ft_epochs": n_epochs_rehearsal,
    })
    return X_trig, y_wm, y_true, stats
