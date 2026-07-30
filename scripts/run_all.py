#!/usr/bin/env python3
"""Run the BAW BODMAS experiment suite."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from baw.config import Config
from baw.main_real import (
    aggregate_track_a,
    get_bodmas,
    print_track_a_summary,
    run_track_a,
    run_track_b_ablations,
    run_track_b_stealth,
    run_track_b_surrogate,
)


def apply_quick_config(cfg: Config) -> None:
    cfg.n_seeds = 2
    cfg.track_a_owner_n = 2_000
    cfg.track_a_reference_n = 1_500
    cfg.track_a_test_n = 1_000
    cfg.track_b_owner_n = 3_000
    cfg.track_b_reference_n = 2_000
    cfg.track_b_test_n = 1_500
    cfg.ablation_K = (50, 150)
    cfg.ablation_eps = (0.15, 0.30)
    cfg.ablation_wm_weight = (1.0, 2.0)
    cfg.n_epochs_base = 8
    cfg.wm_epochs = 5
    cfg.ft_epochs = 5
    cfg.distill_epochs = 6
    cfg.fine_prune_ft_epochs = 4


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a reduced smoke configuration.")
    parser.add_argument("--outdir", default=None, help="Override the output directory.")
    parser.add_argument(
        "--skip-track-b", action="store_true", help="Run only the multi-seed core comparison."
    )
    args = parser.parse_args()

    cfg = Config()
    if args.quick:
        apply_quick_config(cfg)
    if args.outdir:
        cfg.outdir = args.outdir

    os.makedirs(cfg.outdir, exist_ok=True)
    X, y, timestamps = get_bodmas(cfg)

    per_seed = run_track_a(X, y, timestamps, cfg)
    summary, comparisons = aggregate_track_a(per_seed, cfg)
    print_track_a_summary(summary, comparisons, cfg)

    ablations = None
    surrogate_result = None
    stealth_result = None
    if not args.skip_track_b:
        ablations, track_b_data = run_track_b_ablations(X, y, timestamps, cfg)
        surrogate_result = run_track_b_surrogate(cfg, track_b_data)
        stealth_result = run_track_b_stealth(cfg, track_b_data)

    out = {
        "metadata": {
            "git_commit": git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "quick": args.quick,
            "command": " ".join(sys.argv),
        },
        "config": dict(cfg.__dict__),
        "track_a_per_seed": per_seed,
        "track_a_summary": summary,
        "track_a_comparisons": comparisons,
        "track_b_ablations": ablations,
        "track_b_surrogate": surrogate_result,
        "track_b_stealth": stealth_result,
    }

    path = Path(cfg.outdir) / "results_real.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n[done] wrote {path}")


if __name__ == "__main__":
    main()
