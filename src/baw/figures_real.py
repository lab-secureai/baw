from __future__ import annotations

"""Figures for the real-BODMAS BAW evaluation. Designed to be called with
plt.show() inline in Colab (see the packaged notebook), but also works
standalone with matplotlib's Agg backend + savefig if imported in a script.
"""
import matplotlib.pyplot as plt


def fig_track_a_comparison(summary, comparisons):
    schemes = list(summary.keys())
    fields = [
        ("Clean acc", "clean_acc_after", "higher"),
        ("WM signal\n(independent)", "wm_signal_independent_mean", "lower"),
        ("Verification\ngap", "verification_gap", "higher"),
        ("Signal after\nfine-tune", "signal_after_finetune", "higher"),
        ("Signal after\nFine-Pruning", "signal_after_fine_pruning", "higher"),
        ("Direct exploit\nfraction", "direct_exploit_fraction", "lower"),
    ]
    fig, axes = plt.subplots(1, len(fields), figsize=(3 * len(fields), 3.8))
    # Distinct colors: BAW plain, BAW-robust, symmetric, Adi.
    palette = {"BAW (ours)": "#2b7fbf", "BAW-robust (ours)": "#0a3d62",
               "Symmetric backdoor": "#d95f02", "Adi et al. 2018": "#4d9d4a"}
    colors = [palette.get(s, "#888888") for s in schemes]
    for ax, (label, key, better) in zip(axes, fields):
        means, errs = [], []
        for s in schemes:
            stat = summary[s].get(key)
            if stat is None:
                means.append(0.0)
                errs.append(0.0)
            else:
                means.append(stat["mean"])
                errs.append(stat["mean"] - stat["ci95_lo"])
        bars = ax.bar(schemes, means, yerr=errs, capsize=4,
                      color=colors, width=0.6)
        ax.set_title(f"{label}\n({better} is better)", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=7)
    fig.suptitle("BAW-robust (ours) vs plain BAW vs symmetric-backdoor vs Adi et al. (2018) "
                "-- real BODMAS, mean ± 95% CI over seeds", fontsize=11)
    fig.tight_layout()
    plt.show()


def _plot_ablation(x_vals, results, xlabel, title):
    keys = [str(v) for v in x_vals]
    owner = [results[k]["wm_signal_owner"] for k in keys]
    ind = [results[k]["wm_signal_ind_mean"] for k in keys]
    ind_std = [results[k]["wm_signal_ind_std"] for k in keys]
    ft = [results[k]["signal_after_ft"] for k in keys]
    clean = [results[k]["clean_acc_after"] for k in keys]

    fig, ax1 = plt.subplots(figsize=(5.6, 3.8))
    ax1.plot(x_vals, owner, "o-", color="#2b7fbf", label="owner (target ≈ 1)")
    ax1.plot(x_vals, ft, "s-", color="#4d9d4a", label="after fine-tune attack")
    ax1.errorbar(x_vals, ind, yerr=ind_std, marker="v", color="#d95f02",
                 linestyle="--", label="independent (target ≈ 0)")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Watermark signal")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x_vals, clean, ":", color="black", alpha=0.5, label="clean acc")
    ax2.set_ylabel("Clean test accuracy (real BODMAS)", color="0.3")
    ax2.tick_params(axis="y", colors="0.3")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    ax1.set_title(title)
    fig.tight_layout()
    plt.show()


def fig_ablations(ablations):
    _plot_ablation(ablations["trigger_size"]["axis"], ablations["trigger_size"]["results"],
                  "Trigger set size K", "BAW ablation (real BODMAS): trigger set size")
    _plot_ablation(ablations["pgd_eps"]["axis"], ablations["pgd_eps"]["results"],
                  "PGD ε (robust-scaled feature space)",
                  "BAW ablation (real BODMAS): trigger stealthiness (ε)")
    _plot_ablation(ablations["wm_trigger_weight"]["axis"], ablations["wm_trigger_weight"]["results"],
                  "Fine-tune trigger weight",
                  "BAW ablation (real BODMAS): watermark embedding weight")


def fig_surrogate(surrogate_result):
    labels = ["Owner GBT\n(clean acc)", "Owner GBT\n(WM signal)", "Independent\n(WM signal)"]
    vals = [surrogate_result["gbt_clean_acc_after"],
            surrogate_result["gbt_wm_signal_owner"],
            surrogate_result["wm_signal_independent_mean"]]
    fig, ax = plt.subplots(figsize=(5, 3.8))
    bars = ax.bar(labels, vals, color=["#2b7fbf", "#2b7fbf", "#d95f02"], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("BAW embedded into a NON-differentiable owner detector\n"
                f"(GBT, via MLP surrogate; surrogate-teacher agreement "
                f"{surrogate_result['surrogate_fidelity_to_gbt']:.1%})")
    fig.tight_layout()
    plt.show()


def fig_stealth(stealth_result):
    schemes = list(stealth_result.keys())
    scores = [stealth_result[s]["mean_anomaly_score"] for s in schemes]
    outlier_frac = [stealth_result[s]["outlier_fraction"] for s in schemes]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    palette = {"BAW (ours)": "#2b7fbf", "BAW-robust (ours)": "#0a3d62",
               "Symmetric backdoor": "#d95f02", "Adi et al. 2018": "#4d9d4a"}
    colors = [palette.get(s, "#888888") for s in schemes]

    bars = axes[0].bar(schemes, scores, color=colors, width=0.6)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Mean Isolation-Forest anomaly score\n(lower = more OOD / detectable)")
    axes[0].tick_params(axis="x", rotation=20, labelsize=8)
    for b, v in zip(bars, scores):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:+.3f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=8)

    bars2 = axes[1].bar(schemes, outlier_frac, color=colors, width=0.6)
    axes[1].set_title("Outlier fraction\n(higher = easier for an auditor to flag)")
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    axes[1].set_ylim(0, 1.05)
    for b, v in zip(bars2, outlier_frac):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center", fontsize=8)

    fig.suptitle("Trigger-set stealth: is the key detectable as an anomaly?", fontsize=11)
    fig.tight_layout()
    plt.show()
