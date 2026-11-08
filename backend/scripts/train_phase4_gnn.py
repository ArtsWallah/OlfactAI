"""Phase 4 Advanced Molecular Learning & Graph Neural Network Benchmark Script.

Executes:
1. Molecular graph featurization and caching across all 4,983 molecules.
2. Multi-label message-passing neural network (MPNN) training on CPU for:
   - Protocol A: Random 70/15/15 Split
   - Protocol B: Bemis-Murcko Scaffold 70/15/15 Split
3. Post-hoc probability calibration (Temperature Scaling) fitted on validation data only.
4. Expected Calibration Error (ECE) and Brier Score computation.
5. Predictive uncertainty estimation (Shannon Entropy and margin distances).
6. Comprehensive comparative evaluation against Phase 3 baselines (XGBoost and PyTorch MLP).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.app.core.logging import get_logger
from backend.app.data.loaders import OlfactoryDatasetLoader
from backend.app.ml.evaluation.calibration import (
    TemperatureScaler,
    compute_brier_score,
    compute_expected_calibration_error,
)
from backend.app.ml.evaluation.metrics import MultiLabelEvaluator
from backend.app.ml.evaluation.thresholds import (
    apply_thresholds,
    optimize_thresholds,
    save_thresholds,
)
from backend.app.ml.evaluation.uncertainty import compute_predictive_entropy
from backend.app.ml.features.graph import MolecularGraph, MolecularGraphFeaturizer
from backend.app.ml.models.gnn import MolecularGNNSmellModel

logger = get_logger("scripts.train_phase4_gnn")


def get_or_build_graph_cache(processed_dir: Path) -> Dict[str, MolecularGraph]:
    """Load or generate deterministic molecular graphs for all molecules in processed bundle."""
    cache_path = processed_dir / "molecular_graphs_cache.pt"
    molecules_path = processed_dir / "molecules.csv"

    if cache_path.exists():
        logger.info(f"Loading cached molecular graphs from {cache_path}...")
        return torch.load(cache_path, weights_only=False)

    logger.info(f"Generating molecular graphs from {molecules_path}...")
    mol_df = pd.read_csv(molecules_path)
    featurizer = MolecularGraphFeaturizer()
    graph_dict: Dict[str, MolecularGraph] = {}

    t0 = time.perf_counter()
    for _, row in mol_df.iterrows():
        mol_id = str(row["molecule_id"])
        smiles = str(row["canonical_smiles"])
        graph_dict[mol_id] = featurizer(smiles)

    torch.save(graph_dict, cache_path)
    logger.info(f"Generated and cached {len(graph_dict)} molecular graphs in {time.perf_counter() - t0:.2f}s")
    return graph_dict


def run_gnn_for_split(
    processed_dir: Path,
    graph_dict: Dict[str, MolecularGraph],
    output_base_dir: Path,
    split_type: str,
) -> Dict[str, Any]:
    """Train, calibrate, and benchmark Molecular GNN on a single split protocol."""
    split_out_dir = output_base_dir / split_type / "gnn"
    split_out_dir.mkdir(parents=True, exist_ok=True)

    # Load molecules and labels
    mol_df = pd.read_csv(processed_dir / "molecules.csv")
    labels_df = pd.read_csv(processed_dir / "odor_labels.csv")

    split_col = f"{split_type}_split"
    target_labels = [c for c in labels_df.columns if c not in ["molecule_id", "canonical_smiles"]]

    train_mask = mol_df[split_col] == "train"
    val_mask = mol_df[split_col] == "val"
    test_mask = mol_df[split_col] == "test"

    train_ids = mol_df[train_mask]["molecule_id"].tolist()
    val_ids = mol_df[val_mask]["molecule_id"].tolist()
    test_ids = mol_df[test_mask]["molecule_id"].tolist()

    train_graphs = [graph_dict[m_id] for m_id in train_ids]
    val_graphs = [graph_dict[m_id] for m_id in val_ids]
    test_graphs = [graph_dict[m_id] for m_id in test_ids]

    # Map labels matching IDs
    train_labels = labels_df.set_index("molecule_id").loc[train_ids, target_labels].to_numpy(dtype=int)
    val_labels = labels_df.set_index("molecule_id").loc[val_ids, target_labels].to_numpy(dtype=int)
    test_labels = labels_df.set_index("molecule_id").loc[test_ids, target_labels].to_numpy(dtype=int)

    logger.info(f"\n=================== [{split_type.upper()}] Training Molecular GNN ===================")
    logger.info(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}, Test graphs: {len(test_graphs)}")

    # Instantiate model
    model = MolecularGNNSmellModel(
        name=f"molecular_gnn_{split_type}",
        version="1.0.0",
        atom_dim=MolecularGraphFeaturizer.ATOM_FEATURE_DIM,
        bond_dim=MolecularGraphFeaturizer.BOND_FEATURE_DIM,
        hidden_dim=128,
        num_layers=3,
        output_dim=len(target_labels),
        dropout=0.2,
        epochs=30,
        batch_size=64,
        lr=1e-3,
        patience=5,
        random_state=42,
    )

    # Train model
    model.fit(
        X=train_graphs,
        y=train_labels,
        target_labels=target_labels,
        validation_data=(val_graphs, val_labels),
    )

    # Save model checkpoint
    model_path = split_out_dir / "model.pt"
    model.save(model_path)

    # Predict uncalibrated validation and test logits/probs
    val_logits = model.predict_logits(val_graphs)
    val_probs = model.predict_proba(val_graphs)
    test_logits = model.predict_logits(test_graphs)
    test_probs = model.predict_proba(test_graphs)

    # 1. Post-Hoc Probability Calibration: Temperature Scaling on validation logits ONLY
    logger.info("Fitting TemperatureScaler strictly on validation partition...")
    temp_scaler = TemperatureScaler(lr=0.01, max_iter=150)
    temp_scaler.fit(val_logits, val_labels)
    temp_path = split_out_dir / "calibration.json"
    temp_scaler.save(temp_path)

    # Calibrated probabilities
    calibrated_val_probs = temp_scaler.transform(val_logits)
    calibrated_test_probs = temp_scaler.transform(test_logits)

    # 2. Calibration Metrics
    uncal_brier = compute_brier_score(test_labels, test_probs)
    cal_brier = compute_brier_score(test_labels, calibrated_test_probs)

    uncal_ece, uncal_ece_details = compute_expected_calibration_error(test_labels, test_probs, n_bins=10)
    cal_ece, cal_ece_details = compute_expected_calibration_error(test_labels, calibrated_test_probs, n_bins=10)

    logger.info(f"[{split_type.upper()}] Calibration: Brier (uncal={uncal_brier:.4f}, cal={cal_brier:.4f}) | ECE (uncal={uncal_ece:.4f}, cal={cal_ece:.4f}, T={temp_scaler.temperature:.4f})")

    # 3. Decision Threshold Optimization on validation probabilities
    logger.info("Optimizing decision thresholds on validation partition...")
    optimal_thresholds, thresh_metadata = optimize_thresholds(
        y_val_true=val_labels,
        y_val_prob=calibrated_val_probs,
        fallback=0.50,
        min_support=2,
    )

    thresh_path = split_out_dir / "thresholds.json"
    save_thresholds(
        thresholds=optimal_thresholds,
        target_labels=target_labels,
        path=thresh_path,
        metadata=thresh_metadata,
    )

    # Binary test predictions
    test_preds_default = (calibrated_test_probs >= 0.50).astype(int)
    test_preds_opt = apply_thresholds(calibrated_test_probs, optimal_thresholds)

    # 4. Comprehensive Evaluation
    test_eval_default = MultiLabelEvaluator.evaluate(
        y_true=test_labels,
        y_pred=test_preds_default,
        y_prob=calibrated_test_probs,
        target_labels=target_labels,
        include_per_label=True,
    )
    test_eval_opt = MultiLabelEvaluator.evaluate(
        y_true=test_labels,
        y_pred=test_preds_opt,
        y_prob=calibrated_test_probs,
        target_labels=target_labels,
        include_per_label=True,
    )

    # 5. Predictive Uncertainty Analysis
    uncertainty_results = compute_predictive_entropy(calibrated_test_probs)

    summary_entry = {
        "model_name": model.name,
        "version": model.version,
        "architecture": {
            "type": "MolecularGNN (MPNN)",
            "atom_dim": model.atom_dim,
            "bond_dim": model.bond_dim,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "dropout": model.dropout,
        },
        "checkpoint_path": str(model_path),
        "calibration_path": str(temp_path),
        "thresholds_path": str(thresh_path),
        "train_time_sec": model.train_time_sec,
        "inference_time_sec": model.inference_time_sec,
        "temperature": temp_scaler.temperature,
        "calibration_metrics": {
            "uncalibrated_brier": uncal_brier,
            "calibrated_brier": cal_brier,
            "brier_reduction_pct": float(((uncal_brier - cal_brier) / uncal_brier) * 100.0),
            "uncalibrated_ece": uncal_ece,
            "calibrated_ece": cal_ece,
            "ece_reduction_pct": float(((uncal_ece - cal_ece) / uncal_ece) * 100.0),
            "reliability_diagram_bins": cal_ece_details["bins"],
        },
        "uncertainty_metrics": {
            "mean_dataset_entropy": uncertainty_results["mean_dataset_entropy"],
            "median_dataset_entropy": uncertainty_results["median_dataset_entropy"],
            "quantiles": uncertainty_results["quantiles"],
        },
        "threshold_stats": {
            "mean": thresh_metadata["mean_optimal_threshold"],
            "median": thresh_metadata["median_optimal_threshold"],
            "optimized_count": thresh_metadata["optimized_labels_count"],
            "fallback_count": thresh_metadata["fallback_labels_count"],
        },
        "test_metrics": {
            "default_threshold": {k: v for k, v in test_eval_default.items() if k != "per_label"},
            "optimized_threshold": {k: v for k, v in test_eval_opt.items() if k != "per_label"},
        },
        "training_history": {
            "train_losses": model.train_losses,
            "val_losses": model.val_losses,
            "best_epoch": model.best_epoch,
        },
        "test_per_label": test_eval_opt.get("per_label", {}),
    }

    logger.info(
        f"[{split_type.upper()} | GNN] "
        f"Test Macro F1 (default={test_eval_default['f1_macro']:.4f}, opt={test_eval_opt['f1_macro']:.4f}) | "
        f"Test Micro F1 (default={test_eval_default['f1_micro']:.4f}, opt={test_eval_opt['f1_micro']:.4f}) | "
        f"Test Macro ROC-AUC={test_eval_opt.get('roc_auc_macro', 0.0):.4f} | "
        f"Test Macro PR-AUC={test_eval_opt.get('pr_auc_macro', 0.0):.4f}"
    )

    return summary_entry


def main():
    start_total = time.perf_counter()
    logger.info("Initializing Phase 4 Advanced Molecular Learning (GNN) Benchmark...")

    datasets_dir = repo_root / "backend" / "datasets" / "processed"
    models_dir = repo_root / "backend" / "models" / "phase4"
    experiments_dir = repo_root / "backend" / "experiments" / "phase4"
    plots_dir = experiments_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Featurize or load graph cache
    graph_dict = get_or_build_graph_cache(datasets_dir)

    summary: Dict[str, Any] = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware_environment": {
            "device": "cpu",
            "cuda_available": False,
            "architecture": "x86_64",
        },
        "model_architecture": "MolecularGNN (Edge-conditioned Message-Passing Neural Network)",
        "splits": {},
    }

    # 2. Run Protocol A: Random Split
    logger.info("\n>>> STARTING PROTOCOL A: RANDOM SPLIT GNN BENCHMARK <<<")
    summary["splits"]["random"] = run_gnn_for_split(
        processed_dir=datasets_dir,
        graph_dict=graph_dict,
        output_base_dir=models_dir,
        split_type="random",
    )

    # 3. Run Protocol B: Bemis-Murcko Scaffold Split
    logger.info("\n>>> STARTING PROTOCOL B: BEMIS-MURCKO SCAFFOLD SPLIT GNN BENCHMARK <<<")
    summary["splits"]["scaffold"] = run_gnn_for_split(
        processed_dir=datasets_dir,
        graph_dict=graph_dict,
        output_base_dir=models_dir,
        split_type="scaffold",
    )

    # 4. Integrate comparison against Phase 3 Baselines
    phase3_summary_path = repo_root / "backend" / "experiments" / "phase3" / "experiment_summary.json"
    if phase3_summary_path.exists():
        with open(phase3_summary_path, "r", encoding="utf-8") as f:
            phase3_summary = json.load(f)

        summary["phase3_comparison"] = {}
        for split in ["random", "scaffold"]:
            p3_split = phase3_summary["splits"][split]["models"]
            p4_gnn = summary["splits"][split]["test_metrics"]["optimized_threshold"]
            p4_timing = summary["splits"][split]

            summary["phase3_comparison"][split] = {
                "xgboost": {
                    "f1_macro": p3_split["xgboost"]["test_metrics"]["optimized_threshold"]["f1_macro"],
                    "f1_micro": p3_split["xgboost"]["test_metrics"]["optimized_threshold"]["f1_micro"],
                    "roc_auc_macro": p3_split["xgboost"]["test_metrics"]["optimized_threshold"]["roc_auc_macro"],
                    "pr_auc_macro": p3_split["xgboost"]["test_metrics"]["optimized_threshold"]["pr_auc_macro"],
                    "train_time_sec": p3_split["xgboost"]["train_time_sec"],
                    "inference_time_sec": p3_split["xgboost"]["inference_time_sec"],
                },
                "pytorch_mlp": {
                    "f1_macro": p3_split["pytorch_mlp"]["test_metrics"]["optimized_threshold"]["f1_macro"],
                    "f1_micro": p3_split["pytorch_mlp"]["test_metrics"]["optimized_threshold"]["f1_micro"],
                    "roc_auc_macro": p3_split["pytorch_mlp"]["test_metrics"]["optimized_threshold"]["roc_auc_macro"],
                    "pr_auc_macro": p3_split["pytorch_mlp"]["test_metrics"]["optimized_threshold"]["pr_auc_macro"],
                    "train_time_sec": p3_split["pytorch_mlp"]["train_time_sec"],
                    "inference_time_sec": p3_split["pytorch_mlp"]["inference_time_sec"],
                },
                "molecular_gnn": {
                    "f1_macro": p4_gnn["f1_macro"],
                    "f1_micro": p4_gnn["f1_micro"],
                    "roc_auc_macro": p4_gnn["roc_auc_macro"],
                    "pr_auc_macro": p4_gnn["pr_auc_macro"],
                    "train_time_sec": p4_timing["train_time_sec"],
                    "inference_time_sec": p4_timing["inference_time_sec"],
                },
            }

    # Save summary JSON
    summary_file = experiments_dir / "experiment_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save dedicated calibration metrics JSON
    cal_file = experiments_dir / "calibration_metrics.json"
    with open(cal_file, "w", encoding="utf-8") as f:
        json.dump({
            split: {
                "temperature": summary["splits"][split]["temperature"],
                "calibration_metrics": summary["splits"][split]["calibration_metrics"],
                "uncertainty_metrics": summary["splits"][split]["uncertainty_metrics"],
            }
            for split in ["random", "scaffold"]
        }, f, indent=2)

    total_duration = time.perf_counter() - start_total
    logger.info(f"\n==================================================================")
    logger.info(f"Phase 4 GNN Benchmarking completed in {total_duration:.2f}s ({total_duration/60:.2f} min)!")
    logger.info(f"Experiment summary saved to: {summary_file}")
    logger.info(f"Calibration metrics saved to: {cal_file}")
    logger.info(f"==================================================================")


if __name__ == "__main__":
    main()

