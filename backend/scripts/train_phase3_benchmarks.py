"""Phase 3 Multi-Label Machine Learning Benchmark Training & Evaluation Script.

Executes CPU-only baseline model training and evaluation across Random and Scaffold splits:
1. Dummy Frequency Baseline
2. Balanced Random Forest
3. Histogram-based XGBoost
4. PyTorch CPU Multi-Layer Perceptron (MLP)

Outputs model weights, scalers, thresholds, and benchmark performance metrics.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd

from backend.app.core.logging import get_logger
from backend.app.data.loaders import OlfactoryDatasetLoader
from backend.app.ml.evaluation.metrics import MultiLabelEvaluator
from backend.app.ml.evaluation.thresholds import (
    apply_thresholds,
    optimize_thresholds,
    save_thresholds,
)
from backend.app.ml.models.dummy import DummyFrequencySmellModel
from backend.app.ml.models.pytorch_mlp import PyTorchMLPSmellModel
from backend.app.ml.models.random_forest import RandomForestSmellModel
from backend.app.ml.models.xgboost_model import XGBoostSmellModel
from backend.app.ml.preprocessing.scaler import OlfactoryFeatureScaler

logger = get_logger("scripts.train_phase3_benchmarks")


def load_and_scale_partition(
    processed_dir: Path,
    split_type: str,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    List[str], OlfactoryFeatureScaler
]:
    """Load train, val, test partitions and fit scaler strictly on training data."""
    logger.info(f"Loading '{split_type}' split partitions from {processed_dir}...")
    feat_train, lab_train = OlfactoryDatasetLoader.load_split_partition(processed_dir, split_type, "train")
    feat_val, lab_val = OlfactoryDatasetLoader.load_split_partition(processed_dir, split_type, "val")
    feat_test, lab_test = OlfactoryDatasetLoader.load_split_partition(processed_dir, split_type, "test")

    # Target labels (138 columns, excluding molecule_id and canonical_smiles)
    target_labels = [c for c in lab_train.columns if c not in ["molecule_id", "canonical_smiles"]]

    y_train = lab_train[target_labels].to_numpy(dtype=int)
    y_val = lab_val[target_labels].to_numpy(dtype=int)
    y_test = lab_test[target_labels].to_numpy(dtype=int)

    # Initialize and fit scaler exclusively on training partition
    scaler = OlfactoryFeatureScaler()
    scaler.fit(feat_train)

    X_train = scaler.transform(feat_train)
    X_val = scaler.transform(feat_val)
    X_test = scaler.transform(feat_test)

    logger.info(
        f"Partition '{split_type}' prepared: "
        f"Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}, Targets={len(target_labels)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, target_labels, scaler


def run_benchmark_for_split(
    processed_dir: Path,
    output_base_dir: Path,
    split_type: str,
) -> Dict[str, Any]:
    """Run full benchmark pipeline on a single split protocol."""
    split_out_dir = output_base_dir / split_type
    split_out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test, target_labels, scaler = (
        load_and_scale_partition(processed_dir, split_type)
    )

    # Save fitted scaler for reproducible inference
    scaler_path = split_out_dir / "feature_scaler.joblib"
    scaler.save(scaler_path)
    logger.info(f"Saved feature scaler to {scaler_path}")

    # Model definitions
    models = {
        "dummy": DummyFrequencySmellModel(
            name=f"dummy_{split_type}",
            version="1.0.0",
            strategy="prior",
        ),
        "random_forest": RandomForestSmellModel(
            name=f"random_forest_{split_type}",
            version="1.0.0",
            n_estimators=100,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBoostSmellModel(
            name=f"xgboost_{split_type}",
            version="1.0.0",
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=4,
        ),
        "pytorch_mlp": PyTorchMLPSmellModel(
            name=f"pytorch_mlp_{split_type}",
            version="1.0.0",
            input_dim=X_train.shape[1],
            output_dim=len(target_labels),
            hidden_dims=[512, 256, 128],
            dropout=0.2,
            epochs=30,
            batch_size=64,
            lr=1e-3,
            patience=5,
            random_state=42,
        ),
    }

    split_results: Dict[str, Any] = {
        "split_type": split_type,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_targets": len(target_labels),
        "target_labels": target_labels,
        "models": {},
    }

    for model_key, model in models.items():
        logger.info(f"\n=================== [{split_type.upper()}] Training {model_key} ===================")
        model_dir = split_out_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        # Train model
        if model_key == "pytorch_mlp":
            model.fit(
                X_train,
                y_train,
                target_labels=target_labels,
                validation_data=(X_val, y_val),
            )
        else:
            model.fit(X_train, y_train, target_labels=target_labels)

        # Save model checkpoint
        if model_key == "pytorch_mlp":
            ckpt_path = model_dir / "model.pt"
        else:
            ckpt_path = model_dir / "model.joblib"
        model.save(ckpt_path)

        # Predict validation & test probabilities
        val_probs = model.predict_proba(X_val)
        test_probs = model.predict_proba(X_test)

        # Default threshold predictions (0.5)
        val_preds_default = (val_probs >= 0.5).astype(int)
        test_preds_default = (test_probs >= 0.5).astype(int)

        # Optimize thresholds on validation probabilities ONLY
        logger.info(f"Optimizing decision thresholds on validation partition for {model_key}...")
        optimal_thresholds, thresh_metadata = optimize_thresholds(
            y_val_true=y_val,
            y_val_prob=val_probs,
            fallback=0.5,
            min_support=2,
        )

        thresh_path = model_dir / "thresholds.json"
        save_thresholds(
            thresholds=optimal_thresholds,
            target_labels=target_labels,
            path=thresh_path,
            metadata=thresh_metadata,
        )

        # Apply optimized thresholds to validation and test
        val_preds_opt = apply_thresholds(val_probs, optimal_thresholds)
        test_preds_opt = apply_thresholds(test_probs, optimal_thresholds)

        # Compute benchmark evaluations
        val_eval_default = MultiLabelEvaluator.evaluate(
            y_true=y_val,
            y_pred=val_preds_default,
            y_prob=val_probs,
            target_labels=target_labels,
        )
        val_eval_opt = MultiLabelEvaluator.evaluate(
            y_true=y_val,
            y_pred=val_preds_opt,
            y_prob=val_probs,
            target_labels=target_labels,
        )

        test_eval_default = MultiLabelEvaluator.evaluate(
            y_true=y_test,
            y_pred=test_preds_default,
            y_prob=test_probs,
            target_labels=target_labels,
            include_per_label=True,
        )
        test_eval_opt = MultiLabelEvaluator.evaluate(
            y_true=y_test,
            y_pred=test_preds_opt,
            y_prob=test_probs,
            target_labels=target_labels,
            include_per_label=True,
        )

        model_summary = {
            "model_name": model.name,
            "version": model.version,
            "checkpoint_path": str(ckpt_path),
            "thresholds_path": str(thresh_path),
            "train_time_sec": getattr(model, "train_time_sec", 0.0),
            "inference_time_sec": getattr(model, "inference_time_sec", 0.0),
            "threshold_stats": {
                "mean": thresh_metadata["mean_optimal_threshold"],
                "median": thresh_metadata["median_optimal_threshold"],
                "min": thresh_metadata["min_optimal_threshold"],
                "max": thresh_metadata["max_optimal_threshold"],
                "optimized_count": thresh_metadata["optimized_labels_count"],
                "fallback_count": thresh_metadata["fallback_labels_count"],
            },
            "validation_metrics": {
                "default_threshold": val_eval_default,
                "optimized_threshold": val_eval_opt,
            },
            "test_metrics": {
                "default_threshold": {k: v for k, v in test_eval_default.items() if k != "per_label"},
                "optimized_threshold": {k: v for k, v in test_eval_opt.items() if k != "per_label"},
            },
            "test_per_label_optimized": test_eval_opt.get("per_label", {}),
        }

        if model_key == "pytorch_mlp":
            model_summary["training_history"] = {
                "train_losses": getattr(model, "train_losses", []),
                "val_losses": getattr(model, "val_losses", []),
                "best_epoch": getattr(model, "best_epoch", 0),
            }

        split_results["models"][model_key] = model_summary

        logger.info(
            f"[{split_type.upper()} | {model_key}] "
            f"Test Macro F1 (default={test_eval_default['f1_macro']:.4f}, opt={test_eval_opt['f1_macro']:.4f}) | "
            f"Test Micro F1 (default={test_eval_default['f1_micro']:.4f}, opt={test_eval_opt['f1_micro']:.4f}) | "
            f"Test Macro ROC-AUC={test_eval_opt.get('roc_auc_macro', 0.0):.4f} | "
            f"Test Macro PR-AUC={test_eval_opt.get('pr_auc_macro', 0.0):.4f}"
        )

    return split_results


def main():
    start_total = time.perf_counter()
    logger.info("Initializing Phase 3 Baseline Model Benchmarks...")

    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "datasets" / "processed"
    models_dir = repo_root / "models" / "phase3"
    experiments_dir = repo_root / "experiments" / "phase3"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware_environment": {
            "device": "cpu",
            "cuda_available": False,
            "architecture": "x86_64",
        },
        "datasets": {
            "processed_dir": str(processed_dir),
            "split_protocols": ["random", "scaffold"],
        },
        "splits": {},
    }

    # Run Random Split Protocol
    logger.info("\n>>> STARTING PROTOCOL A: RANDOM SPLIT BENCHMARK <<<")
    summary["splits"]["random"] = run_benchmark_for_split(
        processed_dir=processed_dir,
        output_base_dir=models_dir,
        split_type="random",
    )

    # Run Scaffold Split Protocol
    logger.info("\n>>> STARTING PROTOCOL B: BEMIS-MURCKO SCAFFOLD SPLIT BENCHMARK <<<")
    summary["splits"]["scaffold"] = run_benchmark_for_split(
        processed_dir=processed_dir,
        output_base_dir=models_dir,
        split_type="scaffold",
    )

    summary_file = experiments_dir / "experiment_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    total_duration = time.perf_counter() - start_total
    logger.info(f"\n==================================================================")
    logger.info(f"Phase 3 Benchmarks completed successfully in {total_duration:.2f}s ({total_duration/60:.2f} min)!")
    logger.info(f"Consolidated results saved to: {summary_file}")
    logger.info(f"==================================================================")


if __name__ == "__main__":
    main()
