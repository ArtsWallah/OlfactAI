"""Unit tests for Phase 3 evaluation metrics, threshold optimizer, and feature scaler."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backend.app.ml.evaluation.metrics import MultiLabelEvaluator
from backend.app.ml.evaluation.thresholds import (
    apply_thresholds,
    load_thresholds,
    optimize_thresholds,
    save_thresholds,
)
from backend.app.ml.preprocessing.scaler import OlfactoryFeatureScaler


class TestPhase3Evaluation:
    """Test suite for metrics, threshold optimization, and leakage-safe scaling."""

    def test_evaluator_micro_macro_metrics(self):
        """Verify multi-label metric calculations with probabilities."""
        y_true = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ])
        y_prob = np.array([
            [0.9, 0.1, 0.8],
            [0.2, 0.8, 0.3],
            [0.7, 0.6, 0.2],
            [0.1, 0.2, 0.9],
        ])
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = MultiLabelEvaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            target_labels=["odor_a", "odor_b", "odor_c"],
            include_per_label=True,
        )

        assert metrics["subset_accuracy"] == 1.0
        assert metrics["hamming_loss"] == 0.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["roc_auc_macro"] == 1.0
        assert metrics["pr_auc_macro"] == 1.0
        assert metrics["evaluable_labels_count"] == 3
        assert "per_label" in metrics
        assert metrics["per_label"]["odor_a"]["support"] == 2

    def test_evaluator_rare_classes_graceful_handling(self):
        """Verify evaluator handles classes with 0 positive instances in test set without crashing."""
        y_true = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0],  # Third column has 0 positive instances
        ])
        y_pred = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
        ])
        y_prob = np.array([
            [0.8, 0.2, 0.05],
            [0.9, 0.7, 0.01],
            [0.1, 0.1, 0.02],
        ])

        metrics = MultiLabelEvaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            target_labels=["label_1", "label_2", "unrepresented_rare"],
            include_per_label=True,
        )

        # Should not crash, evaluable count should be 2 out of 3
        assert metrics["evaluable_labels_count"] == 2
        assert metrics["total_labels_count"] == 3
        assert metrics["per_label"]["unrepresented_rare"]["support"] == 0
        assert metrics["per_label"]["unrepresented_rare"]["evaluable"] is False
        assert metrics["per_label"]["unrepresented_rare"]["roc_auc"] is None

    def test_threshold_optimizer_on_synthetic_data(self):
        """Verify threshold optimization finds optimal cutoffs per label."""
        y_val_true = np.array([
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 0],
        ])
        # Probabilities are skewed (e.g. conservative probabilities around 0.3 for positive class)
        y_val_prob = np.array([
            [0.35, 0.10],
            [0.40, 0.15],
            [0.10, 0.35],
            [0.05, 0.40],
            [0.05, 0.05],
        ])

        # Under default 0.5 threshold, predictions would be all zeros (F1 = 0)
        default_pred = (y_val_prob >= 0.5).astype(int)
        assert np.all(default_pred == 0)

        thresholds, meta = optimize_thresholds(y_val_true, y_val_prob, min_support=2)
        assert len(thresholds) == 2
        assert thresholds[0] < 0.5
        assert thresholds[1] < 0.5

        opt_pred = apply_thresholds(y_val_prob, thresholds)
        assert np.array_equal(opt_pred, y_val_true)

    def test_threshold_optimizer_fallback_for_low_support(self):
        """Verify fallback to 0.5 when positive support is below threshold."""
        y_val_true = np.array([[1, 0], [0, 0], [0, 0]])  # Class 0 has support=1 < min_support=2
        y_val_prob = np.array([[0.9, 0.1], [0.1, 0.1], [0.1, 0.1]])

        thresholds, meta = optimize_thresholds(y_val_true, y_val_prob, fallback=0.5, min_support=2)
        assert thresholds[0] == 0.5
        assert thresholds[1] == 0.5

    def test_save_load_thresholds_roundtrip(self, tmp_path):
        """Verify serialization and deserialization of optimized thresholds."""
        thresholds = np.array([0.35, 0.45, 0.50], dtype=np.float32)
        labels = ["fruity", "floral", "woody"]
        path = tmp_path / "thresholds.json"

        save_thresholds(thresholds, labels, path, metadata={"note": "test"})
        loaded_t, loaded_l, loaded_meta = load_thresholds(path)

        assert np.allclose(loaded_t, thresholds)
        assert loaded_l == labels
        assert loaded_meta.get("note") == "test"

    def test_olfactory_feature_scaler_continuous_only(self):
        """Verify OlfactoryFeatureScaler standardizes 9 continuous features and leaves Morgan bits intact."""
        np.random.seed(42)
        n_samples = 50
        n_desc = 9
        n_fp = 64

        # Continuous descriptors with arbitrary mean/std
        desc = np.random.normal(loc=150.0, scale=30.0, size=(n_samples, n_desc))
        # Binary Morgan bits (0 or 1)
        fp = np.random.choice([0.0, 1.0], size=(n_samples, n_fp))
        hybrid_x = np.hstack([desc, fp])

        scaler = OlfactoryFeatureScaler(num_continuous=n_desc)
        scaled_x = scaler.fit_transform(hybrid_x)

        # Continuous portion should have mean ~ 0 and std ~ 1
        desc_scaled = scaled_x[:, :n_desc]
        np.testing.assert_allclose(desc_scaled.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(desc_scaled.std(axis=0), 1.0, atol=1e-5)

        # Fingerprint portion must remain exactly identical (0 or 1)
        fp_scaled = scaled_x[:, n_desc:]
        assert np.array_equal(fp_scaled, fp)
        assert set(np.unique(fp_scaled)).issubset({0.0, 1.0})

    def test_olfactory_feature_scaler_save_load(self, tmp_path):
        """Verify feature scaler serialization roundtrip."""
        scaler = OlfactoryFeatureScaler(num_continuous=9)
        dummy_x = np.ones((10, 20))
        scaler.fit(dummy_x)

        save_path = tmp_path / "scaler.joblib"
        scaler.save(save_path)

        loaded_scaler = OlfactoryFeatureScaler.load(save_path)
        assert loaded_scaler.is_fitted is True
        assert loaded_scaler.num_continuous == 9
