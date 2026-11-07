"""Generate publication-quality benchmark plots for Phase 3 evaluation.

Visualizations:
1. model_f1_comparison.png: Macro & Micro F1 across models and split protocols.
2. model_auc_comparison.png: Macro ROC-AUC & Macro PR-AUC comparison.
3. random_vs_scaffold_performance.png: Generalization gap between Random & Scaffold splits.
4. training_inference_time.png: Compute cost and latency comparison.
5. pytorch_mlp_loss_curves.png: Epoch training & validation loss curves for PyTorch MLP.
6. threshold_optimization_impact.png: Impact of per-label threshold tuning on F1 score.
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

# Style configuration
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


def plot_f1_comparison(summary: dict, out_dir: Path):
    """Plot Macro F1 and Micro F1 across models and splits."""
    models = ["dummy", "random_forest", "xgboost", "pytorch_mlp"]
    model_labels = ["Dummy", "Random Forest", "XGBoost", "PyTorch MLP"]
    splits = ["random", "scaffold"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, metric in enumerate(["f1_macro", "f1_micro"]):
        ax = axes[idx]
        x = np.arange(len(models))
        width = 0.35

        scores_random = [
            summary["splits"]["random"]["models"][m]["test_metrics"]["optimized_threshold"][metric]
            for m in models
        ]
        scores_scaffold = [
            summary["splits"]["scaffold"]["models"][m]["test_metrics"]["optimized_threshold"][metric]
            for m in models
        ]

        bars1 = ax.bar(x - width / 2, scores_random, width, label="Random Split", color="#3b82f6", alpha=0.9)
        bars2 = ax.bar(x + width / 2, scores_scaffold, width, label="Scaffold Split", color="#f97316", alpha=0.9)

        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_title(f"Test Set {metric.replace('_', ' ').title()} (Optimized Thresholds)")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend(loc="upper left")
        ax.set_ylim(0, max(max(scores_random), max(scores_scaffold), 0.5) * 1.2)

        # Bar value annotations
        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "model_f1_comparison.png")
    plt.close()


def plot_auc_comparison(summary: dict, out_dir: Path):
    """Plot Macro ROC-AUC and Macro PR-AUC across models."""
    models = ["dummy", "random_forest", "xgboost", "pytorch_mlp"]
    model_labels = ["Dummy", "Random Forest", "XGBoost", "PyTorch MLP"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, metric in enumerate(["roc_auc_macro", "pr_auc_macro"]):
        ax = axes[idx]
        x = np.arange(len(models))
        width = 0.35

        scores_random = [
            summary["splits"]["random"]["models"][m]["test_metrics"]["optimized_threshold"].get(metric, 0.0)
            for m in models
        ]
        scores_scaffold = [
            summary["splits"]["scaffold"]["models"][m]["test_metrics"]["optimized_threshold"].get(metric, 0.0)
            for m in models
        ]

        bars1 = ax.bar(x - width / 2, scores_random, width, label="Random Split", color="#10b981", alpha=0.9)
        bars2 = ax.bar(x + width / 2, scores_scaffold, width, label="Scaffold Split", color="#6366f1", alpha=0.9)

        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_title(f"Test Set {metric.replace('_', ' ').upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend(loc="upper left")
        ax.set_ylim(0, 1.05)

        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "model_auc_comparison.png")
    plt.close()


def plot_random_vs_scaffold_performance(summary: dict, out_dir: Path):
    """Plot the generalization gap (Performance Delta) between Random and Scaffold splits."""
    models = ["random_forest", "xgboost", "pytorch_mlp"]
    model_labels = ["Random Forest", "XGBoost", "PyTorch MLP"]
    metrics = ["f1_macro", "f1_micro", "roc_auc_macro", "pr_auc_macro"]
    metric_labels = ["Macro F1", "Micro F1", "Macro ROC-AUC", "Macro PR-AUC"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.2

    colors = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6"]

    for i, (m_key, m_lbl) in enumerate(zip(metrics, metric_labels)):
        deltas = []
        for m in models:
            rand_val = summary["splits"]["random"]["models"][m]["test_metrics"]["optimized_threshold"].get(m_key, 0.0)
            scaf_val = summary["splits"]["scaffold"]["models"][m]["test_metrics"]["optimized_threshold"].get(m_key, 0.0)
            # Generalization drop = Random - Scaffold
            drop_pct = ((rand_val - scaf_val) / max(rand_val, 1e-6)) * 100.0
            deltas.append(drop_pct)

        bars = ax.bar(x + (i - 1.5) * width, deltas, width, label=f"{m_lbl} Gap (%)", color=colors[i], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3 if h >= 0 else -10), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Generalization Drop (% from Random to Scaffold)")
    ax.set_title("Scaffold Generalization Gap (Structural Extrapolation Penalty)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend()
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(out_dir / "random_vs_scaffold_performance.png")
    plt.close()


def plot_training_inference_time(summary: dict, out_dir: Path):
    """Plot training and inference wall-clock duration."""
    models = ["dummy", "random_forest", "xgboost", "pytorch_mlp"]
    model_labels = ["Dummy", "Random Forest", "XGBoost", "PyTorch MLP"]

    train_times = [
        summary["splits"]["scaffold"]["models"][m]["train_time_sec"]
        for m in models
    ]
    infer_times = [
        summary["splits"]["scaffold"]["models"][m]["inference_time_sec"] * 1000.0  # to ms
        for m in models
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training time
    bars1 = ax1.bar(model_labels, train_times, color="#8b5cf6", alpha=0.85)
    ax1.set_ylabel("Training Time (seconds, CPU)")
    ax1.set_title("Model Training Duration (3,488 molecules, 2,057 features)")
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}s", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # Inference latency
    bars2 = ax2.bar(model_labels, infer_times, color="#06b6d4", alpha=0.85)
    ax2.set_ylabel("Test Inference Latency (ms for 748 molecules)")
    ax2.set_title("Inference Latency on 748 Test Molecules (CPU)")
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}ms", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "training_inference_time.png")
    plt.close()


def plot_pytorch_mlp_loss_curves(summary: dict, out_dir: Path):
    """Plot training and validation loss curves for PyTorch MLP."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, split_key, title in zip(
        [ax1, ax2],
        ["random", "scaffold"],
        ["Protocol A: Random Split", "Protocol B: Scaffold Split"],
    ):
        history = summary["splits"][split_key]["models"]["pytorch_mlp"].get("training_history", {})
        train_loss = history.get("train_losses", [])
        val_loss = history.get("val_losses", [])
        best_epoch = history.get("best_epoch", 0)

        epochs = range(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, label="Training Loss", color="#2563eb", marker="o", markersize=4)
        if val_loss:
            ax.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss", color="#dc2626", marker="s", markersize=4)
            if best_epoch > 0 and best_epoch <= len(val_loss):
                ax.axvline(best_epoch, color="#16a34a", linestyle="--", label=f"Best Model (Epoch {best_epoch})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCEWithLogitsLoss (weighted)")
        ax.set_title(f"PyTorch MLP Convergence ({title})")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "pytorch_mlp_loss_curves.png")
    plt.close()


def plot_threshold_optimization_impact(summary: dict, out_dir: Path):
    """Plot Macro F1 gain from default 0.5 to validation-optimized thresholds."""
    models = ["dummy", "random_forest", "xgboost", "pytorch_mlp"]
    model_labels = ["Dummy", "Random Forest", "XGBoost", "PyTorch MLP"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, split_key in enumerate(["random", "scaffold"]):
        ax = axes[idx]
        x = np.arange(len(models))
        width = 0.35

        def_f1 = [
            summary["splits"][split_key]["models"][m]["test_metrics"]["default_threshold"]["f1_macro"]
            for m in models
        ]
        opt_f1 = [
            summary["splits"][split_key]["models"][m]["test_metrics"]["optimized_threshold"]["f1_macro"]
            for m in models
        ]

        bars1 = ax.bar(x - width / 2, def_f1, width, label="Default (0.50)", color="#94a3b8", alpha=0.9)
        bars2 = ax.bar(x + width / 2, opt_f1, width, label="Optimized Thresholds", color="#0284c7", alpha=0.9)

        ax.set_ylabel("Test Set Macro F1")
        ax.set_title(f"Threshold Tuning Impact ({split_key.title()} Split)")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend(loc="upper left")

        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "threshold_optimization_impact.png")
    plt.close()


def main():
    repo_root = Path(__file__).resolve().parent.parent
    summary_path = repo_root / "experiments" / "phase3" / "experiment_summary.json"
    plots_dir = repo_root / "experiments" / "phase3" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        print(f"Error: {summary_path} not found. Run benchmark training first.")
        return

    summary = load_summary(summary_path)

    print("Generating Phase 3 benchmark plots...")
    plot_f1_comparison(summary, plots_dir)
    plot_auc_comparison(summary, plots_dir)
    plot_random_vs_scaffold_performance(summary, plots_dir)
    plot_training_inference_time(summary, plots_dir)
    plot_pytorch_mlp_loss_curves(summary, plots_dir)
    plot_threshold_optimization_impact(summary, plots_dir)
    print(f"Successfully generated 6 publication plots in: {plots_dir}")


if __name__ == "__main__":
    main()
