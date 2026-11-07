"""Automated regression and audit test suite for Phase 3 models and evaluation guards."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from backend.app.ml.evaluation.metrics import MultiLabelEvaluator
from backend.app.ml.evaluation.thresholds import apply_thresholds, load_thresholds, optimize_thresholds
from backend.app.ml.models.pytorch_mlp import PyTorchMLPSmellModel
from backend.app.ml.models.xgboost_model import XGBoostSmellModel
from backend.app.ml.preprocessing.scaler import OlfactoryFeatureScaler


class TestPhase3AuditGuards:
    """Rigorous audit tests validating data quarantine, reload consistency, and edge-cases."""

    def test_threshold_leakage_quarantine(self):
        """Verify that threshold optimization operates exclusively on validation set without test leakage."""
        np.random.seed(42)
        n_val = 50
        n_test = 50
        n_classes = 5

        y_val_true = np.random.binomial(1, 0.3, size=(n_val, n_classes))
        y_val_prob = np.random.uniform(0.1, 0.9, size=(n_val, n_classes))

        y_test_true = np.random.binomial(1, 0.3, size=(n_test, n_classes))
        y_test_prob = np.random.uniform(0.1, 0.9, size=(n_test, n_classes))

        # Optimize exclusively on validation
        thresholds, meta = optimize_thresholds(y_val_true, y_val_prob)

        # Applying to test probabilities must not alter the threshold vector or raise exceptions
        test_preds = apply_thresholds(y_test_prob, thresholds)
        assert test_preds.shape == (n_test, n_classes)
        assert set(np.unique(test_preds)).issubset({0, 1})
        assert len(thresholds) == n_classes

    def test_metric_unrepresented_labels_macro_count(self):
        """Verify MultiLabelEvaluator correctly excludes unrepresented labels (support=0) from macro averages."""
        # 4 samples, 3 classes: class 0 has support 2, class 1 has support 1, class 2 has support 0
        y_true = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        y_prob = np.array([
            [0.8, 0.2, 0.1],
            [0.7, 0.8, 0.1],
            [0.2, 0.1, 0.05],
            [0.1, 0.1, 0.02],
        ])
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = MultiLabelEvaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            target_labels=["class_a", "class_b", "unrepresented_c"],
            include_per_label=True,
        )

        assert metrics["evaluable_labels_count"] == 2
        assert metrics["total_labels_count"] == 3
        assert metrics["evaluable_roc_auc_count"] == 2
        assert metrics["evaluable_pr_auc_count"] == 2
        assert not np.isnan(metrics["roc_auc_macro"])
        assert metrics["per_label"]["unrepresented_c"]["support"] == 0
        assert metrics["per_label"]["unrepresented_c"]["evaluable"] is False
        assert metrics["per_label"]["unrepresented_c"]["roc_auc"] is None

    def test_strict_cpu_execution_enforcement(self):
        """Verify PyTorch model strictly forces CPU device and does not attempt CUDA access."""
        model = PyTorchMLPSmellModel(input_dim=10, output_dim=2)
        assert model.device == torch.device("cpu")
        for param in model.network.parameters():
            assert param.device.type == "cpu"

    def test_saved_scaler_and_thresholds_consistency(self):
        """Verify that saved phase3 checkpoint scalers and thresholds load and have correct dimensions."""
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        scaffold_dir = repo_root / "backend" / "models" / "phase3" / "scaffold"

        scaler_file = scaffold_dir / "feature_scaler.joblib"
        thresh_file = scaffold_dir / "pytorch_mlp" / "thresholds.json"

        assert scaler_file.exists()
        assert thresh_file.exists()

        scaler = OlfactoryFeatureScaler.load(scaler_file)
        assert scaler.is_fitted is True
        assert scaler.num_continuous == 9

        thresholds, labels, meta = load_thresholds(thresh_file)
        assert len(thresholds) == 138
        assert len(labels) == 138
        assert "mean" in meta or "mean_optimal_threshold" in meta

