from __future__ import annotations

# ============================================================================
# config.py
# ============================================================================
"""BAW real-data (BODMAS) experiment configuration.

Two tracks, to keep total Colab runtime in the ~30-45 min budget while
still producing paper-grade evidence:

  TRACK A (multi-seed core comparison)
      Runs the full BAW vs symmetric-backdoor vs Adi-et-al comparison
      `n_seeds` times on a subsample, reporting mean ± std / bootstrap CI
      for every headline number. This is what buys you statistical
      credibility (a single run is not publishable).

  TRACK B (single-seed deep analysis)
      Runs once, on the default seed, the more expensive / illustrative
      analyses that don't need repetition to make their point:
      hyperparameter ablations, non-differentiable-model transfer via a
      surrogate MLP, and a trigger-set "stealth" (OOD-detectability) probe.

Both tracks share the same temporal train/test split methodology
(TESSERACT, Pendlebury et al., USENIX Security 2019): the test set is the
chronologically LATEST slice of BODMAS, never seen during any training or
hyperparameter selection. This avoids the classic malware-ML pitfall of
inflated accuracy from i.i.d. shuffling across a train/test boundary that
does not respect time.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # -- BODMAS download / cache ------------------------------------------
    data_dir: str = "./bodmas_data"
    bodmas_drive_folder_id: str = "1Uf-LebLWyi9eCv97iBal7kL1NgiGEsv_"
    kaggle_dataset: str = "dhoogla/bodmas"

    # -- Temporal split (chronological, NOT i.i.d. shuffled) --------------
    # Fraction of the (time-sorted) dataset reserved as the test set: the
    # chronologically LATEST slice. Everything before it is the
    # "historical pool" that owner-train and reference-train are carved
    # from (disjoint, random split -- both trained on contemporaneous data,
    # which is the realistic case: the owner and an independent auditor
    # both had access to malware feed data up to the same point in time).
    test_frac: float = 0.20
    owner_frac_of_hist: float = 0.55
    reference_frac_of_hist: float = 0.45  # disjoint from owner within the historical pool

    # -- Subsampling (controls total runtime; increase for a final run) ---
    # None => use the full historical pool / full test set.
    track_a_owner_n: int = 8000
    track_a_reference_n: int = 6000
    track_a_test_n: int = 4000
    track_b_owner_n: int = 15000
    track_b_reference_n: int = 10000
    track_b_test_n: int = 6000

    # -- Feature preprocessing ---------------------------------------------
    # BODMAS ships RAW, unnormalized features with drastically different
    # per-feature ranges (file-size-like features vs. 0/1 flags). We fit a
    # RobustScaler on OWNER-TRAIN ONLY (never on reference/test -- avoids
    # leakage) and apply it to every split, including whatever the
    # attacker/independent auditor sees. This keeps the PGD L_inf budget
    # meaningful across all 2381 dimensions.
    clip_after_scale: float = 8.0   # clip extreme outliers post-scaling

    # -- Detector architecture ----------------------------------------------
    n_features: int = 2381    # BODMAS / EMBER feature dimensionality
    hidden: int = 256
    n_classes: int = 2         # 0 = benign, 1 = malware  (matches BODMAS's own convention)

    # -- Base training -------------------------------------------------------
    batch_size: int = 256
    lr: float = 1e-3
    n_epochs_base: int = 20
    weight_decay: float = 1e-4

    # -- BAW / symmetric trigger construction --------------------------------
    trigger_size: int = 150
    trigger_candidate_pool: int = 1200
    pgd_steps: int = 60
    pgd_alpha: float = 0.02
    pgd_eps: float = 0.25       # L_inf budget in ROBUST-SCALED feature space

    # -- Robustness-aware BAW selection (baw_robust.py) ----------------------
    # Oversample the PGD candidate pool by this factor, then filter down to
    # `trigger_size` via a rehearsal fine-tune attack (see baw_robust.py's
    # module docstring). 5x pool gives meaningful selection headroom
    # without doubling the runtime; rehearsal uses the same fine-tune the
    # attacker gets, so we're selecting for exactly the property that
    # matters (post-attack survival).
    robust_pool_multiplier: int = 5
    robust_rehearsal_ft_epochs: int = 5

    # -- Adi et al. (2018) "unrelated" content-free watermark ----------------
    # Trigger inputs are abstract/OOD noise vectors (NOT derived from any
    # real file), paired with an arbitrary fixed label. Classic DNN
    # watermarking baseline; included as the literature comparison point.
    adi_trigger_size: int = 150
    adi_label: int = 0           # arbitrary fixed watermark label (benign)

    # -- Watermark embedding (fine-tune) --------------------------------------
    wm_epochs: int = 10
    wm_lr: float = 3e-4
    wm_trigger_weight: float = 2.0

    # -- Removal attacks -------------------------------------------------------
    ft_epochs: int = 10
    ft_lr: float = 1e-3
    prune_ratios: Tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)
    distill_epochs: int = 14
    distill_lr: float = 1e-3
    distill_temperature: float = 4.0

    # Fine-Pruning (Liu et al., RAID 2018): activation-based pruning of the
    # neurons LEAST active on the attacker's own clean data, THEN fine-tune.
    # This is the standard "adaptive, watermark-aware" attack in the
    # backdoor-defense literature and is a meaningfully stronger adversary
    # than magnitude pruning.
    fine_prune_ratio: float = 0.3
    fine_prune_ft_epochs: int = 8

    # -- Verification ----------------------------------------------------------
    n_independent_models: int = 3
    verification_gap_threshold: float = 0.5

    # -- Non-exploitability probe -----------------------------------------------
    n_fresh_eval: int = 1500
    evasion_eps_sweep: Tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)

    # -- Surrogate / non-differentiable transfer (Track B) ----------------------
    gbt_max_iter: int = 150
    gbt_max_depth: int = 6
    surrogate_hidden: int = 256
    surrogate_distill_epochs: int = 15
    surrogate_distill_lr: float = 1e-3
    surrogate_distill_temperature: float = 3.0

    # -- Multi-seed (Track A) -----------------------------------------------------
    # n_seeds=8 keeps the Colab budget reasonable while giving Wilcoxon
    # signed-rank a meaningful floor p (2^-(8-1) ≈ 0.008 vs 2^-4 = 0.0625
    # at n=5, where every effect -- however large -- comes back to
    # exactly the sign-only-bound and reads as noise to a reviewer).
    base_seed: int = 20240501
    n_seeds: int = 8

    # -- Ablation grids (Track B) --------------------------------------------------
    ablation_K: Tuple[int, ...] = (50, 100, 150, 300)
    ablation_eps: Tuple[float, ...] = (0.10, 0.18, 0.25, 0.35)
    ablation_wm_weight: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)

    # -- Output ----------------------------------------------------------------------
    outdir: str = "results_real"
