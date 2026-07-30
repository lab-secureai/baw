from __future__ import annotations

import numpy as np

from .nn import MLP, train

"""Adaptive, watermark-aware attacks.

Fine-Pruning (Liu, Dolan-Gavitt, Garg; RAID 2018, "Fine-Pruning: Defending
Against Backdooring Attacks on Deep Neural Networks") is the standard
STRONGER attack in the backdoor-watermark-removal literature: it is
specifically designed around the hypothesis that a backdoor/watermark
trigger activates hidden units that are mostly dormant on genuine clean
inputs. The attacker (1) profiles average hidden-unit activation on their
own clean data, (2) prunes the least-active units (removing whatever
specialized machinery encodes the trigger response), (3) fine-tunes to
recover any clean-accuracy loss. This is meaningfully stronger than either
magnitude-pruning or fine-tuning alone (attacks.py), because it explicitly
targets the mechanism a trigger-based watermark relies on, rather than
attacking indiscriminately.

We also provide `trigger_set_stealth`, which is not a removal attack but a
detection probe: how anomalous does each scheme's trigger set look to an
Isolation Forest trained on the attacker's own genuine data? This
quantifies a real, orthogonal cost axis -- a key that is easy to flag as
out-of-distribution is a liability for covert verification, independent of
whether it can be "removed".
"""



def fine_pruning_attack(stolen: MLP, X_own, y_own, cfg, rng=None):
    """Prune the last-hidden-layer units LEAST active on the attacker's
    own clean data, then fine-tune. Returns a fresh model."""
    rng = rng or np.random.default_rng(0)
    m = stolen.clone()

    _, cache = m.forward(X_own, cache=True)
    mean_act = np.abs(cache["a2"]).mean(axis=0)      # (hidden,)
    n_hidden = mean_act.shape[0]
    n_prune = int(cfg.fine_prune_ratio * n_hidden)
    prune_idx = np.argsort(mean_act)[:n_prune]        # least-active first

    # Zero the pruned units' contribution to the output layer entirely --
    # standard structured pruning of a hidden unit.
    m.params["W3"][prune_idx, :] = 0.0

    train(m, X_own, y_own,
          epochs=cfg.fine_prune_ft_epochs, lr=cfg.ft_lr,
          batch_size=cfg.batch_size, weight_decay=cfg.weight_decay, rng=rng)
    return m


def trigger_set_stealth(X_clean_pool: np.ndarray, trigger_sets: dict, seed: int = 0):
    """How anomalous does each scheme's trigger set look to an outlier
    detector trained on genuine data?

    trigger_sets: {scheme_name: X_trig} — evaluated against the SAME
    Isolation Forest, fit once on the attacker's own clean pool.

    Returns {scheme_name: {mean_anomaly_score, outlier_fraction}}.
    Lower mean_anomaly_score / higher outlier_fraction == easier for a
    suspicious party to flag and filter out the key.
    """
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=150, contamination="auto", random_state=seed)
    iso.fit(X_clean_pool)

    out = {}
    for name, X_trig in trigger_sets.items():
        scores = iso.decision_function(X_trig)     # higher = more "normal"
        preds = iso.predict(X_trig)                  # -1 outlier, 1 inlier
        out[name] = {
            "mean_anomaly_score": float(scores.mean()),
            "outlier_fraction": float((preds == -1).mean()),
        }
    return out
