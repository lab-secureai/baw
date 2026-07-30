from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import replace

import numpy as np

from .attacks import distill_attack, fine_tune_attack, prune_attack
from .attacks_adaptive import fine_pruning_attack, trigger_set_stealth
from .baseline import build_symmetric_triggers
from .baseline_adi import (
    adi_watermark_signal,
    build_adi_trigger_set,
    embed_adi_watermark,
)
from .baw import build_trigger_set, embed_watermark, watermark_signal
from .baw_robust import build_robust_trigger_set
from .config import Config
from .data_bodmas import (
    RobustFeatureScaler,
    download_bodmas,
    load_bodmas,
    temporal_split,
)
from .exploit import exploit_report
from .nn import MLP, accuracy, train
from .stats_utils import paired_compare, summarize
from .surrogate import (
    embed_gbt_watermark,
    gbt_accuracy,
    gbt_watermark_signal,
    train_gbt,
    train_surrogate_mlp,
)

"""BAW end-to-end evaluation on REAL BODMAS data. No synthetic stand-ins
anywhere in this file -- every number is computed from the downloaded
BODMAS feature vectors (see data_bodmas.py).

TRACK A -- multi-seed core comparison (BAW vs symmetric-backdoor vs
Adi-et-al), giving mean/std/CI over `cfg.n_seeds` independent repetitions
of: fidelity, verification signal+gap, robustness to 4 removal attacks
(fine-tune, distill, magnitude-prune, Fine-Pruning), and non-exploitability
(BAW/symmetric only -- Adi's abstract trigger set isn't a real file, so
"is this truly malware" isn't well-defined for it; noted explicitly in the
output rather than silently omitted).

TRACK B -- single-seed deep analysis: hyperparameter ablations (K, PGD eps,
watermark weight), a non-differentiable-detector (gradient-boosted tree)
transfer test via a differentiable surrogate, and a trigger-set stealth
(OOD-detectability) comparison across all three schemes.

Both tracks share the SAME temporal split methodology: the test set is the
chronologically latest slice of BODMAS, never touched during training,
trigger construction, embedding, or attack.
"""



# ===========================================================================
# Shared helpers
# ===========================================================================
def train_fresh(X, y, cfg, seed):
    rng = np.random.default_rng(seed)
    m = MLP(cfg.n_features, cfg.hidden, cfg.n_classes, rng=rng)
    train(m, X, y, epochs=cfg.n_epochs_base, lr=cfg.lr,
          batch_size=cfg.batch_size, weight_decay=cfg.weight_decay, rng=rng)
    return m


def build_independent_pool(X_ref, y_ref, cfg, seed_base, n_models):
    """Train n_models fresh detectors ONCE; every scheme's trigger set is
    scored against this SAME pool (they don't depend on the trigger set at
    all, so retraining per-scheme would be pure waste)."""
    return [train_fresh(X_ref, y_ref, cfg, seed=seed_base + 10_000 + i)
            for i in range(n_models)]


def get_bodmas(cfg):
    npz_path, csv_path = download_bodmas(cfg.data_dir)
    X, y, timestamps = load_bodmas(npz_path, csv_path)
    return X, y, timestamps


def prepare_split(X, y, timestamps, cfg, seed, n_owner, n_reference, n_test):
    rng = np.random.default_rng(seed)
    splits = temporal_split(X, y, timestamps, cfg, rng,
                            n_owner=n_owner, n_reference=n_reference, n_test=n_test)
    X_owner, y_owner = splits["owner"]
    X_ref, y_ref = splits["reference"]
    X_test, y_test = splits["test"]

    scaler = RobustFeatureScaler(clip=cfg.clip_after_scale).fit(X_owner)
    X_owner = scaler.transform(X_owner)
    X_ref = scaler.transform(X_ref)
    X_test = scaler.transform(X_test)
    return (X_owner, y_owner), (X_ref, y_ref), (X_test, y_test), scaler


# ===========================================================================
# Per-scheme evaluation (shared by BAW / symmetric / Adi)
# ===========================================================================
def evaluate_scheme(name, owner_base, X_trig, y_wm, y_true_trig, trig_stats,
                    X_owner, y_owner, X_test, y_test, X_ref, y_ref,
                    independent_models, cfg, seed, X_fresh_mal, X_fresh_ben,
                    embed_fn, signal_fn):
    owner_wm = embed_fn(owner_base, X_trig, y_wm, X_owner, y_owner, cfg,
                        rng=np.random.default_rng(seed + 7))

    clean_before = accuracy(owner_base, X_test, y_test)
    clean_after = accuracy(owner_wm, X_test, y_test)
    sig_owner = signal_fn(owner_wm, X_trig)

    ind_sigs = [signal_fn(m, X_trig) for m in independent_models]
    ind_mean, ind_std, ind_max = (float(np.mean(ind_sigs)),
                                  float(np.std(ind_sigs)), float(np.max(ind_sigs)))

    ft = fine_tune_attack(owner_wm, X_ref, y_ref, cfg, rng=np.random.default_rng(seed + 11))
    sig_ft = signal_fn(ft, X_trig)

    dstl = distill_attack(owner_wm, X_ref, cfg, rng=np.random.default_rng(seed + 13))
    sig_dstl = signal_fn(dstl, X_trig)

    fp = fine_pruning_attack(owner_wm, X_ref, y_ref, cfg, rng=np.random.default_rng(seed + 17))
    sig_fp = signal_fn(fp, X_trig)

    sig_prune = {}
    for r in cfg.prune_ratios:
        pm = prune_attack(owner_wm, r)
        sig_prune[f"prune@{r}"] = signal_fn(pm, X_trig)

    result = {
        "scheme": name,
        "seed": seed,
        "clean_acc_before": clean_before,
        "clean_acc_after": clean_after,
        "wm_signal_owner": sig_owner,
        "wm_signal_independent_mean": ind_mean,
        "wm_signal_independent_std": ind_std,
        "wm_signal_independent_max": ind_max,
        "verification_gap": sig_owner - ind_max,
        "verification_passes": bool((sig_owner - ind_max) >= cfg.verification_gap_threshold),
        "signal_after_finetune": sig_ft,
        "signal_after_distill": sig_dstl,
        "signal_after_fine_pruning": sig_fp,
        "signal_after_prune": sig_prune,
        "trigger_size": int(X_trig.shape[0]),
        "trigger_yield_rate": trig_stats.get("yield_rate"),
        "trigger_mean_l2_perturbation": trig_stats.get("mean_l2_perturbation"),
    }

    if y_true_trig is not None:
        exp = exploit_report(owner_wm, X_trig, y_true_trig, X_fresh_mal, X_fresh_ben, cfg)
        result.update({
            "direct_exploit_count": exp["direct_exploit_count"],
            "direct_exploit_fraction": exp["direct_exploit_fraction"],
            "evasion_keyfree_best": exp["evasion_keyfree_best"],
            "evasion_withkey_best": exp["evasion_withkey_best"],
            "evasion_uplift_from_key": exp["evasion_uplift_from_key"],
        })
    else:
        result.update({
            "direct_exploit_count": None, "direct_exploit_fraction": None,
            "evasion_keyfree_best": None, "evasion_withkey_best": None,
            "evasion_uplift_from_key": None,
            "note": "Adi-et-al trigger inputs are synthetic/abstract, not "
                    "real files -- 'truly malware vs. called benign' is not "
                    "well-defined for them, so direct-exploit metrics are "
                    "reported as N/A rather than a misleading number.",
        })
    return result, owner_wm


def make_adi_signal_fn(cfg):
    return lambda model, X_trig: adi_watermark_signal(model, X_trig, cfg)


# ===========================================================================
# TRACK A -- multi-seed core comparison
# ===========================================================================
def run_track_a(X, y, timestamps, cfg):
    print("=" * 78)
    print(f"TRACK A -- multi-seed core comparison  (n_seeds={cfg.n_seeds})")
    print("=" * 78)

    per_seed_results = {"BAW (ours)": [], "BAW-robust (ours)": [],
                        "Symmetric backdoor": [], "Adi et al. 2018": []}
    t_start = time.time()

    for si in range(cfg.n_seeds):
        seed = cfg.base_seed + si
        print(f"\n--- seed {si+1}/{cfg.n_seeds}  (seed={seed}) ---")
        t0 = time.time()

        (X_owner, y_owner), (X_ref, y_ref), (X_test, y_test), scaler = prepare_split(
            X, y, timestamps, cfg, seed,
            n_owner=cfg.track_a_owner_n, n_reference=cfg.track_a_reference_n,
            n_test=cfg.track_a_test_n)

        owner_base = train_fresh(X_owner, y_owner, cfg, seed=seed)
        reference = train_fresh(X_ref, y_ref, cfg, seed=seed + 42)
        print(f"  owner clean acc {accuracy(owner_base, X_test, y_test):.4f}  "
              f"reference clean acc {accuracy(reference, X_test, y_test):.4f}")

        independent_models = build_independent_pool(X_ref, y_ref, cfg, seed, cfg.n_independent_models)

        benign_pool = X_owner[y_owner == 0][: cfg.trigger_candidate_pool]
        malware_pool = X_owner[y_owner == 1][: cfg.trigger_candidate_pool]

        baw_X, baw_y, baw_ytrue, baw_stats = build_trigger_set(reference, benign_pool, cfg)
        # BAW-robust: same PGD construction, then filter down by
        # rehearsal-attack survival margin (baw_robust.py).
        bawr_X, bawr_y, bawr_ytrue, bawr_stats = build_robust_trigger_set(
            reference, owner_base, benign_pool,
            X_ref, y_ref, cfg, rng=np.random.default_rng(seed + 5))
        sym_X, sym_y, sym_ytrue, sym_stats = build_symmetric_triggers(owner_base, malware_pool, cfg)
        adi_X, adi_y, adi_stats = build_adi_trigger_set(X_owner, cfg, rng=np.random.default_rng(seed + 3))
        print(f"  trigger yields -- BAW {baw_stats['yield_rate']:.1%}  "
              f"BAW-robust pool={bawr_stats.get('robust_pool_size')} kept-margin={bawr_stats.get('robust_kept_survival_margin_mean'):.3f}  "
              f"symmetric {sym_stats['yield_rate']:.1%}  Adi n/a (synthetic)")

        n_eval = min(cfg.n_fresh_eval, (y_ref == 1).sum(), (y_ref == 0).sum())
        X_fresh_mal = X_ref[y_ref == 1][:n_eval]
        X_fresh_ben = X_ref[y_ref == 0][:n_eval]

        r_baw, _ = evaluate_scheme(
            "BAW (ours)", owner_base, baw_X, baw_y, baw_ytrue, baw_stats,
            X_owner, y_owner, X_test, y_test, X_ref, y_ref,
            independent_models, cfg, seed, X_fresh_mal, X_fresh_ben,
            embed_fn=embed_watermark, signal_fn=watermark_signal)

        r_bawr, _ = evaluate_scheme(
            "BAW-robust (ours)", owner_base, bawr_X, bawr_y, bawr_ytrue, bawr_stats,
            X_owner, y_owner, X_test, y_test, X_ref, y_ref,
            independent_models, cfg, seed, X_fresh_mal, X_fresh_ben,
            embed_fn=embed_watermark, signal_fn=watermark_signal)

        r_sym, _ = evaluate_scheme(
            "Symmetric backdoor", owner_base, sym_X, sym_y, sym_ytrue, sym_stats,
            X_owner, y_owner, X_test, y_test, X_ref, y_ref,
            independent_models, cfg, seed, X_fresh_mal, X_fresh_ben,
            embed_fn=embed_watermark, signal_fn=watermark_signal)

        r_adi, _ = evaluate_scheme(
            "Adi et al. 2018", owner_base, adi_X, adi_y, None, adi_stats,
            X_owner, y_owner, X_test, y_test, X_ref, y_ref,
            independent_models, cfg, seed, X_fresh_mal, X_fresh_ben,
            embed_fn=embed_adi_watermark, signal_fn=make_adi_signal_fn(cfg))

        for name, r in [("BAW (ours)", r_baw), ("BAW-robust (ours)", r_bawr),
                        ("Symmetric backdoor", r_sym), ("Adi et al. 2018", r_adi)]:
            per_seed_results[name].append(r)

        print(f"  seed done in {time.time()-t0:.1f}s  "
              f"(BAW gap={r_baw['verification_gap']:.3f} ft={r_baw['signal_after_finetune']:.3f}  "
              f"BAW-robust gap={r_bawr['verification_gap']:.3f} ft={r_bawr['signal_after_finetune']:.3f}  "
              f"sym gap={r_sym['verification_gap']:.3f})")

    print(f"\n[Track A] total time {time.time()-t_start:.1f}s")
    return per_seed_results


def aggregate_track_a(per_seed_results, cfg):
    """Turn the per-seed result lists into mean/std/CI tables + paired
    significance tests between BAW and each baseline."""
    numeric_fields = [
        "clean_acc_after", "wm_signal_owner", "wm_signal_independent_mean",
        "verification_gap", "signal_after_finetune", "signal_after_distill",
        "signal_after_fine_pruning",
    ]
    exploit_fields = ["direct_exploit_count", "direct_exploit_fraction",
                      "evasion_uplift_from_key"]

    summary = {}
    for scheme, rows in per_seed_results.items():
        summary[scheme] = {}
        for f in numeric_fields:
            summary[scheme][f] = summarize([r[f] for r in rows])
        for f in exploit_fields:
            vals = [r[f] for r in rows if r[f] is not None]
            summary[scheme][f] = summarize(vals) if vals else None

    comparisons = {}
    schemes = list(per_seed_results.keys())

    # The methodologically important comparison for the "does our new
    # selection actually help" question: robust vs plain BAW, same seeds.
    if "BAW-robust (ours)" in per_seed_results:
        baw_plain_rows = per_seed_results["BAW (ours)"]
        baw_robust_rows = per_seed_results["BAW-robust (ours)"]
        comparisons["BAW-robust vs BAW (plain)"] = {}
        for f in numeric_fields:
            a = [r[f] for r in baw_robust_rows]
            b = [r[f] for r in baw_plain_rows]
            comparisons["BAW-robust vs BAW (plain)"][f] = paired_compare(
                a, b, "BAW-robust", "BAW")
        for f in exploit_fields:
            a_vals = [r[f] for r in baw_robust_rows]
            b_vals = [r[f] for r in baw_plain_rows]
            if all(v is not None for v in a_vals) and all(v is not None for v in b_vals):
                comparisons["BAW-robust vs BAW (plain)"][f] = paired_compare(
                    a_vals, b_vals, "BAW-robust", "BAW")
            else:
                comparisons["BAW-robust vs BAW (plain)"][f] = None

    # Plain-BAW vs each baseline (unchanged).
    baw_rows = per_seed_results["BAW (ours)"]
    for other in schemes:
        if other in ("BAW (ours)", "BAW-robust (ours)"):
            continue
        other_rows = per_seed_results[other]
        comparisons[f"BAW vs {other}"] = {}
        for f in numeric_fields:
            a = [r[f] for r in baw_rows]
            b = [r[f] for r in other_rows]
            comparisons[f"BAW vs {other}"][f] = paired_compare(a, b, "BAW", other)
        for f in exploit_fields:
            a_vals = [r[f] for r in baw_rows]
            b_vals = [r[f] for r in other_rows]
            if all(v is not None for v in a_vals) and all(v is not None for v in b_vals):
                comparisons[f"BAW vs {other}"][f] = paired_compare(a_vals, b_vals, "BAW", other)
            else:
                comparisons[f"BAW vs {other}"][f] = None

    # Robust-BAW vs each baseline too, so the paper can report robust vs
    # its full field of comparison, not just vs plain BAW.
    if "BAW-robust (ours)" in per_seed_results:
        bawr_rows = per_seed_results["BAW-robust (ours)"]
        for other in schemes:
            if other in ("BAW (ours)", "BAW-robust (ours)"):
                continue
            other_rows = per_seed_results[other]
            comparisons[f"BAW-robust vs {other}"] = {}
            for f in numeric_fields:
                a = [r[f] for r in bawr_rows]
                b = [r[f] for r in other_rows]
                comparisons[f"BAW-robust vs {other}"][f] = paired_compare(
                    a, b, "BAW-robust", other)
            for f in exploit_fields:
                a_vals = [r[f] for r in bawr_rows]
                b_vals = [r[f] for r in other_rows]
                if all(v is not None for v in a_vals) and all(v is not None for v in b_vals):
                    comparisons[f"BAW-robust vs {other}"][f] = paired_compare(
                        a_vals, b_vals, "BAW-robust", other)
                else:
                    comparisons[f"BAW-robust vs {other}"][f] = None

    return summary, comparisons


# ===========================================================================
# TRACK B -- single-seed deep analysis
# ===========================================================================
def one_ablation_point(cfg_point, owner_base, reference, independent_models,
                       X_owner, y_owner, X_test, y_test, X_ref, y_ref, seed):
    benign_pool = X_owner[y_owner == 0][: cfg_point.trigger_candidate_pool]
    X_trig, y_wm, _, stats = build_trigger_set(reference, benign_pool, cfg_point)
    if X_trig.shape[0] == 0:
        return {"skipped": True, **stats}
    owner_wm = embed_watermark(owner_base, X_trig, y_wm, X_owner, y_owner, cfg_point,
                               rng=np.random.default_rng(seed + 7))
    ind_sigs = [watermark_signal(m, X_trig) for m in independent_models]
    ft = fine_tune_attack(owner_wm, X_ref, y_ref, cfg_point, rng=np.random.default_rng(seed + 11))
    return {
        "clean_acc_after": accuracy(owner_wm, X_test, y_test),
        "wm_signal_owner": watermark_signal(owner_wm, X_trig),
        "wm_signal_ind_mean": float(np.mean(ind_sigs)),
        "wm_signal_ind_std": float(np.std(ind_sigs)),
        "signal_after_ft": watermark_signal(ft, X_trig),
        "trigger_kept": int(X_trig.shape[0]),
        "yield": stats["yield_rate"],
    }


def run_track_b_ablations(X, y, timestamps, cfg):
    print("\n" + "=" * 78)
    print("TRACK B.1 -- ablations (K, PGD eps, watermark weight) on real BODMAS")
    print("=" * 78)
    seed = cfg.base_seed
    (X_owner, y_owner), (X_ref, y_ref), (X_test, y_test), scaler = prepare_split(
        X, y, timestamps, cfg, seed,
        n_owner=cfg.track_b_owner_n, n_reference=cfg.track_b_reference_n,
        n_test=cfg.track_b_test_n)

    owner_base = train_fresh(X_owner, y_owner, cfg, seed=seed)
    reference = train_fresh(X_ref, y_ref, cfg, seed=seed + 42)
    independent_models = build_independent_pool(X_ref, y_ref, cfg, seed, cfg.n_independent_models)
    print(f"  base owner acc {accuracy(owner_base, X_test, y_test):.4f}  "
          f"reference acc {accuracy(reference, X_test, y_test):.4f}")

    base_state = (owner_base, reference, independent_models, X_owner, y_owner,
                 X_test, y_test, X_ref, y_ref, seed)

    def sweep(param_name, values):
        out = {}
        for v in values:
            c = copy.deepcopy(cfg)
            setattr(c, param_name, v)
            if param_name == "trigger_size":
                c.trigger_candidate_pool = max(v * 3, cfg.trigger_candidate_pool)
            print(f"    {param_name}={v}")
            out[str(v)] = one_ablation_point(c, *base_state)
        return out

    print("  sweep trigger_size (K)")
    sweep_K = sweep("trigger_size", list(cfg.ablation_K))
    print("  sweep pgd_eps")
    sweep_eps = sweep("pgd_eps", list(cfg.ablation_eps))
    print("  sweep wm_trigger_weight")
    sweep_w = sweep("wm_trigger_weight", list(cfg.ablation_wm_weight))

    return {
        "trigger_size": {"axis": list(cfg.ablation_K), "results": sweep_K},
        "pgd_eps": {"axis": list(cfg.ablation_eps), "results": sweep_eps},
        "wm_trigger_weight": {"axis": list(cfg.ablation_wm_weight), "results": sweep_w},
    }, (X_owner, y_owner, X_test, y_test, X_ref, y_ref, owner_base, reference)


def run_track_b_surrogate(cfg, track_b_data):
    print("\n" + "=" * 78)
    print("TRACK B.2 -- non-differentiable detector transfer (GBT surrogate)")
    print("=" * 78)
    X_owner, y_owner, X_test, y_test, X_ref, y_ref, owner_mlp, reference = track_b_data
    seed = cfg.base_seed

    t0 = time.time()
    gbt_base = train_gbt(X_owner, y_owner, cfg, seed=seed)
    gbt_acc_before = gbt_accuracy(gbt_base, X_test, y_test)
    print(f"  owner GBT (non-differentiable) clean acc {gbt_acc_before:.4f}  "
          f"({time.time()-t0:.1f}s)")

    t0 = time.time()
    surrogate = train_surrogate_mlp(gbt_base, X_owner, cfg, rng=np.random.default_rng(seed + 99))
    fidelity_to_teacher = float((surrogate.predict(X_test) == gbt_base.predict(X_test)).mean())
    print(f"  surrogate MLP trained ({time.time()-t0:.1f}s), "
          f"agrees with GBT on {fidelity_to_teacher:.1%} of test set")

    # Craft the BAW key against the INDEPENDENT reference (differentiable,
    # unchanged from the main protocol) -- the surrogate is only needed on
    # the OWNER's side, to embed into their non-differentiable model.
    benign_pool = X_owner[y_owner == 0][: cfg.trigger_candidate_pool]
    X_trig, y_wm, y_true, stats = build_trigger_set(reference, benign_pool, cfg)
    print(f"  BAW trigger set built (yield {stats['yield_rate']:.1%})")

    t0 = time.time()
    gbt_wm = embed_gbt_watermark(X_owner, y_owner, X_trig, y_wm, cfg, seed=seed)
    gbt_acc_after = gbt_accuracy(gbt_wm, X_test, y_test)
    sig_gbt_owner = gbt_watermark_signal(gbt_wm, X_trig, target_label=0)
    print(f"  GBT watermark embedded via sample-weighted refit ({time.time()-t0:.1f}s)")
    print(f"  owner GBT clean acc: {gbt_acc_before:.4f} -> {gbt_acc_after:.4f}")
    print(f"  owner GBT signal on trigger set: {sig_gbt_owner:.4f}  (target ~1.0)")

    # Independent check: does an independent MLP auditor (never saw the
    # trigger construction, never saw the GBT) still call it malware?
    ind_models = build_independent_pool(X_ref, y_ref, cfg, seed, cfg.n_independent_models)
    ind_sig_mean = float(np.mean([watermark_signal(m, X_trig) for m in ind_models]))

    return {
        "gbt_clean_acc_before": gbt_acc_before,
        "gbt_clean_acc_after": gbt_acc_after,
        "surrogate_fidelity_to_gbt": fidelity_to_teacher,
        "gbt_wm_signal_owner": sig_gbt_owner,
        "wm_signal_independent_mean": ind_sig_mean,
        "verification_gap": sig_gbt_owner - ind_sig_mean,
        "trigger_yield_rate": stats["yield_rate"],
    }


def run_track_b_stealth(cfg, track_b_data):
    print("\n" + "=" * 78)
    print("TRACK B.3 -- trigger-set stealth (OOD-detectability) comparison")
    print("=" * 78)
    X_owner, y_owner, X_test, y_test, X_ref, y_ref, owner_base, reference = track_b_data
    seed = cfg.base_seed

    benign_pool = X_owner[y_owner == 0][: cfg.trigger_candidate_pool]
    malware_pool = X_owner[y_owner == 1][: cfg.trigger_candidate_pool]
    baw_X, *_ = build_trigger_set(reference, benign_pool, cfg)
    bawr_X, *_ = build_robust_trigger_set(
        reference, owner_base, benign_pool, X_ref, y_ref, cfg,
        rng=np.random.default_rng(seed + 5))
    sym_X, *_ = build_symmetric_triggers(owner_base, malware_pool, cfg)
    adi_X, *_ = build_adi_trigger_set(X_owner, cfg, rng=np.random.default_rng(seed + 3))

    stealth = trigger_set_stealth(
        X_owner,
        {"BAW (ours)": baw_X, "BAW-robust (ours)": bawr_X,
         "Symmetric backdoor": sym_X, "Adi et al. 2018": adi_X},
        seed=seed)
    for name, s in stealth.items():
        print(f"  {name:22s}  mean anomaly score {s['mean_anomaly_score']:+.4f}  "
              f"outlier fraction {s['outlier_fraction']:.1%}")
    return stealth


def print_track_a_summary(summary, comparisons, cfg):
    print("\n" + "=" * 78)
    print(f"TRACK A SUMMARY  (n_seeds={cfg.n_seeds}, mean ± std, 95% CI in brackets)")
    print("=" * 78)
    fields = [
        ("clean_acc_after", "Clean test accuracy"),
        ("wm_signal_owner", "WM signal (owner)"),
        ("wm_signal_independent_mean", "WM signal (independent)"),
        ("verification_gap", "Verification gap"),
        ("signal_after_finetune", "Signal after fine-tune"),
        ("signal_after_distill", "Signal after distillation"),
        ("signal_after_fine_pruning", "Signal after Fine-Pruning (adaptive)"),
        ("direct_exploit_fraction", "Direct exploit fraction"),
        ("evasion_uplift_from_key", "Evasion uplift from key"),
    ]
    for scheme, stats_by_field in summary.items():
        print(f"\n--- {scheme} ---")
        for f, label in fields:
            s = stats_by_field.get(f)
            if s is None:
                print(f"  {label:38s}: N/A")
            else:
                print(f"  {label:38s}: {s['mean']:.4f} ± {s['std']:.4f}  "
                      f"[{s['ci95_lo']:.4f}, {s['ci95_hi']:.4f}]")

    print("\n--- Paired significance tests (vs. BAW, same seeds) ---")
    for comp_name, comp in comparisons.items():
        print(f"\n{comp_name}:")
        for f, label in fields:
            r = comp.get(f)
            if r is None:
                print(f"  {label:38s}: N/A")
                continue
            p_t = r.get("ttest_p")
            p_w = r.get("wilcoxon_p")
            p_t_str = f"{p_t:.4f}" if p_t is not None else "n/a"
            p_w_str = f"{p_w:.4f}" if p_w is not None else "n/a"
            print(f"  {label:38s}: Δ={r['mean_diff']:+.4f}  "
                  f"paired-t p={p_t_str}  wilcoxon p={p_w_str}")
    print("\n(Note: n_seeds is kept small for Colab runtime; treat p-values as "
          "suggestive, not definitive. Report exact n and both tests in the "
          "paper rather than a single p-value.)")


# ===========================================================================
# Top-level
# ===========================================================================
def main():
    cfg = Config()
    os.makedirs(cfg.outdir, exist_ok=True)

    X, y, timestamps = get_bodmas(cfg)

    per_seed = run_track_a(X, y, timestamps, cfg)
    summary, comparisons = aggregate_track_a(per_seed, cfg)
    print_track_a_summary(summary, comparisons, cfg)

    ablations, track_b_data = run_track_b_ablations(X, y, timestamps, cfg)
    surrogate_result = run_track_b_surrogate(cfg, track_b_data)
    stealth_result = run_track_b_stealth(cfg, track_b_data)

    out = {
        "config": {k: v for k, v in cfg.__dict__.items()},
        "track_a_per_seed": per_seed,
        "track_a_summary": summary,
        "track_a_comparisons": comparisons,
        "track_b_ablations": ablations,
        "track_b_surrogate": surrogate_result,
        "track_b_stealth": stealth_result,
    }
    path = os.path.join(cfg.outdir, "results_real.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[done] wrote {path}")
    return out
