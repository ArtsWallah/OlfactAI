"""Generate publication-quality benchmark plots for Phase 4 GNN evaluation.

Visualizations:
1. gnn_loss_curves.png: Epoch training & validation loss curves for Random and Scaffold splits.
2. phase3_vs_phase4_f1_comparison.png: Macro & Micro F1 across XGBoost, MLP, and GNN.
3. phase3_vs_phase4_auc_comparison.png: Macro ROC-AUC & Macro PR-AUC across architectures.
4. reliability_diagram.png: Reliability curve before and after temperature scaling.
5. uncertainty_distribution.png: Predictive Shannon entropy distributions.
6. scaffold_extrapolation_comparison.png: Structural extrapolation gap across models.
"""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update(
    {
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
    }
)


def load_summary(summary_path: Path) -> dict:
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_gnn_loss_curves(summary: dict, out_dir: Path):
    """Plot training and validation loss curves for GNN on both splits."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, split_key, title in zip(
        [ax1, ax2],
        ["random", "scaffold"],
        ["Protocol A: Random Split", "Protocol B: Scaffold Split"],
    ):
        history = summary["splits"][split_key].get("training_history", {})
        train_loss = history.get("train_losses", [])
        val_loss = history.get("val_losses", [])
        best_epoch = history.get("best_epoch", 0)

        epochs = range(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, label="Training Loss", color="#0284c7", marker="o", markersize=4)
        if val_loss:
            ax.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss", color="#ea580c", marker="s", markersize=4)
            if best_epoch > 0 and best_epoch <= len(val_loss):
                ax.axvline(best_epoch, color="#16a34a", linestyle="--", label=f"Best Model (Epoch {best_epoch})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCEWithLogitsLoss (weighted)")
        ax.set_title(f"Molecular GNN Convergence ({title})")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "gnn_loss_curves.png")
    plt.close()


def plot_phase3_vs_phase4_f1(summary: dict, out_dir: Path):
    """Compare Macro F1 and Micro F1 across XGBoost, MLP, and GNN."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    models = ["xgboost", "pytorch_mlp", "molecular_gnn"]
    model_labels = ["Phase 3: XGBoost", "Phase 3: PyTorch MLP", "Phase 4: Molecular GNN"]

    p3_comp = summary.get("phase3_comparison", {})
    if not p3_comp:
        return

    for idx, (m_key, title_metric) in enumerate([("f1_macro", "Macro F1"), ("f1_micro", "Micro F1")]):
        ax = axes[idx]
        x = np.arange(len(models))
        width = 0.35

        scores_random = [p3_comp["random"][m][m_key] for m in models]
        scores_scaffold = [p3_comp["scaffold"][m][m_key] for m in models]

        bars1 = ax.bar(x - width / 2, scores_random, width, label="Random Split", color="#3b82f6", alpha=0.9)
        bars2 = ax.bar(x + width / 2, scores_scaffold, width, label="Scaffold Split", color="#f97316", alpha=0.9)

        ax.set_ylabel(title_metric)
        ax.set_title(f"Test Set {title_metric} (Phase 3 Baselines vs Phase 4 GNN)")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend(loc="upper left")
        ax.set_ylim(0, max(max(scores_random), max(scores_scaffold), 0.4) * 1.25)

        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "phase3_vs_phase4_f1_comparison.png")
    plt.close()


def plot_phase3_vs_phase4_auc(summary: dict, out_dir: Path):
    """Compare Macro ROC-AUC and Macro PR-AUC across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    models = ["xgboost", "pytorch_mlp", "molecular_gnn"]
    model_labels = ["Phase 3: XGBoost", "Phase 3: PyTorch MLP", "Phase 4: Molecular GNN"]

    p3_comp = summary.get("phase3_comparison", {})
    if not p3_comp:
        return

    for idx, (m_key, title_metric) in enumerate([("roc_auc_macro", "Macro ROC-AUC"), ("pr_auc_macro", "Macro PR-AUC")]):
        ax = axes[idx]
        x = np.arange(len(models))
        width = 0.35

        scores_random = [p3_comp["random"][m][m_key] for m in models]
        scores_scaffold = [p3_comp["scaffold"][m][m_key] for m in models]

        bars1 = ax.bar(x - width / 2, scores_random, width, label="Random Split", color="#10b981", alpha=0.9)
        bars2 = ax.bar(x + width / 2, scores_scaffold, width, label="Scaffold Split", color="#6366f1", alpha=0.9)

        ax.set_ylabel(title_metric)
        ax.set_title(f"Test Set {title_metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend(loc="upper left")
        ax.set_ylim(0, 1.05)

        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "phase3_vs_phase4_auc_comparison.png")
    plt.close()


def plot_reliability_diagram(summary: dict, out_dir: Path):
    """Plot reliability curve (mean confidence vs empirical accuracy)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, split_key, title in zip(
        [ax1, ax2],
        ["random", "scaffold"],
        ["Protocol A: Random Split", "Protocol B: Scaffold Split"],
    ):
        cal_meta = summary["splits"][split_key]["calibration_metrics"]
        bins = cal_meta["reliability_diagram_bins"]
        t_val = summary["splits"][split_key]["temperature"]
        ece = cal_meta["calibrated_ece"]

        confs = [b["avg_confidence"] for b in bins if b["count"] > 0]
        accs = [b["empirical_accuracy"] for b in bins if b["count"] > 0]

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
        ax.plot(confs, accs, marker="o", color="#2563eb", linewidth=2, label=f"Calibrated GNN (T={t_val:.2f}, ECE={ece:.4f})")

        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Empirical True Positive Rate")
        ax.set_title(f"Reliability Diagram ({title})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "reliability_diagram.png")
    plt.close()


def plot_uncertainty_distribution(summary: dict, out_dir: Path):
    """Plot distribution of molecule-level predictive entropy."""
    fig, ax = plt.subplots(figsize=(9, 5))

    rand_mean_ent = summary["splits"]["random"]["uncertainty_metrics"]["mean_dataset_entropy"]
    scaf_mean_ent = summary["splits"]["scaffold"]["uncertainty_metrics"]["mean_dataset_entropy"]

    categories = ["Random Split Test Set", "Scaffold Split Test Set"]
    entropies = [rand_mean_ent, scaf_mean_ent]

    bars = ax.bar(categories, entropies, color=["#3b82f6", "#f97316"], width=0.45, alpha=0.9)
    ax.set_ylabel("Mean Predictive Shannon Entropy (bits)")
    ax.set_title("Molecular Predictive Uncertainty (Ambiguity Across 138 Odor Notes)")
    ax.set_ylim(0, max(entropies) * 1.3)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f} bits", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, weight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "uncertainty_distribution.png")
    plt.close()


def plot_scaffold_extrapolation_comparison(summary: dict, out_dir: Path):
    """Compare generalization drop from Random to Scaffold across XGBoost, MLP, and GNN."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    models = ["xgboost", "pytorch_mlp", "molecular_gnn"]
    model_labels = ["Phase 3: XGBoost", "Phase 3: PyTorch MLP", "Phase 4: Molecular GNN"]

    p3_comp = summary.get("phase3_comparison", {})
    if not p3_comp:
        return

    metrics = ["f1_macro", "roc_auc_macro", "pr_auc_macro"]
    metric_labels = ["Macro F1 Drop", "Macro ROC-AUC Drop", "Macro PR-AUC Drop"]
    colors = ["#ef4444", "#10b981", "#6366f1"]

    x = np.arange(len(models))
    width = 0.25

    for i, (m_key, m_lbl) in enumerate(zip(metrics, metric_labels)):
        drops = []
        for m in models:
            rand_val = p3_comp["random"][m][m_key]
            scaf_val = p3_comp["scaffold"][m][m_key]
            drop_pct = ((rand_val - scaf_val) / max(rand_val, 1e-6)) * 100.0
            drops.append(drop_pct)

        bars = ax.bar(x + (i - 1) * width, drops, width, label=m_lbl, color=colors[i], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3 if h >= 0 else -10), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Extrapolation Penalty (% Drop from Random to Scaffold)")
    ax.set_title("Scaffold Generalization Gap Comparison (Extrapolative Robustness)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "scaffold_extrapolation_comparison.png")
    plt.close()


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    summary_path = repo_root / "backend" / "experiments" / "phase4" / "experiment_summary.json"
    plots_dir = repo_root / "backend" / "experiments" / "phase4" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        print(f"Error: {summary_path} not found. Run GNN benchmark training first.")
        return

    summary = load_summary(summary_path)

    print("Generating Phase 4 benchmark plots...")
    plot_gnn_loss_curves(summary, plots_dir)
    plot_phase3_vs_phase4_f1(summary, plots_dir)
    plot_phase3_vs_phase4_auc(summary, plots_dir)
    plot_reliability_diagram(summary, plots_dir)
    plot_uncertainty_distribution(summary, plots_dir)
    plot_scaffold_extrapolation_comparison(summary, plots_dir)
    print(f"Successfully generated all 6 publication plots in: {plots_dir}")


if __name__ == "__main__":
    main()

