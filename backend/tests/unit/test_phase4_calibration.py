"""Unit tests for Phase 4 calibration and uncertainty metrics."""

import numpy as np
import pytest

from backend.app.ml.evaluation.calibration import (
    TemperatureScaler,
    compute_brier_score,
    compute_expected_calibration_error,
)
from backend.app.ml.evaluation.uncertainty import (
    compute_decision_margin_uncertainty,
    compute_predictive_entropy,
)


class TestPhase4Calibration:
    """Test suite for probability calibration metrics and temperature scaling."""

    def test_brier_score_bounds(self):
        """Verify Brier score bounds for perfect, random, and opposite predictions."""
        y_true = np.array([[1, 0], [0, 1]])

        # Perfect prediction: MSE = 0.0
        assert compute_brier_score(y_true, np.array([[1.0, 0.0], [0.0, 1.0]])) == 0.0

        # Opposite prediction: MSE = 1.0
        assert compute_brier_score(y_true, np.array([[0.0, 1.0], [1.0, 0.0]])) == 1.0

        # Uniform 0.5 prediction: MSE = 0.25
        assert compute_brier_score(y_true, np.array([[0.5, 0.5], [0.5, 0.5]])) == 0.25

    def test_ece_calculation_ten_bins(self):
        """Verify Expected Calibration Error calculation on synthetic probability distribution."""
        # Perfectly calibrated prediction
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Group 1: 4 predictions with p=0.0 (all 0s)
        # Group 2: 4 predictions with p=1.0 (all 1s)
        y_prob = np.array([0.01, 0.02, 0.01, 0.03, 0.98, 0.99, 0.97, 0.99])

        ece, details = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
        assert details["n_bins"] == 10
        assert len(details["bins"]) == 10
        assert ece < 0.05  # Highly calibrated

    def test_temperature_scaler_fitting(self, tmp_path):
        """Verify temperature scaling optimizes T > 0 on validation logits."""
        np.random.seed(42)
        n_samples = 40
        n_classes = 3

        # Overconfident logits
        val_logits = np.random.randn(n_samples, n_classes).astype(np.float32) * 5.0
        val_y = np.random.binomial(1, 0.5, size=(n_samples, n_classes))

        scaler = TemperatureScaler(lr=0.05, max_iter=50)
        scaler.fit(val_logits, val_y)

        assert scaler.is_fitted is True
        assert scaler.temperature > 0.0

        cal_probs = scaler.transform(val_logits)
        assert cal_probs.shape == (n_samples, n_classes)
        assert np.all((cal_probs >= 0.0) & (cal_probs <= 1.0))

        # Save and reload
        save_file = tmp_path / "temp.json"
        scaler.save(save_file)
        loaded = TemperatureScaler.load(save_file)
        assert np.isclose(loaded.temperature, scaler.temperature)


class TestPhase4Uncertainty:
    """Test suite for predictive entropy and decision margin uncertainty."""

    def test_predictive_entropy_bounds(self):
        """Verify entropy is maximized at p=0.50 (1.0 bit) and minimized at p=0 or 1 (0.0 bit)."""
        # 3 molecules, 1 note
        probs = np.array([[0.50], [0.00], [1.00]])
        res = compute_predictive_entropy(probs)

        mol_entropy = res["molecule_mean_entropy"]
        # p=0.50 should have entropy ~ 1.0 bit
        assert np.isclose(mol_entropy[0], 1.0, atol=1e-3)
        # p=0.00 and p=1.00 should have entropy ~ 0.0 bit
        assert np.isclose(mol_entropy[1], 0.0, atol=1e-3)
        assert np.isclose(mol_entropy[2], 0.0, atol=1e-3)

    def test_decision_margin_uncertainty(self):
        """Verify decision margin distance |p - 0.5|."""
        probs = np.array([[0.50, 0.90], [0.10, 0.40]])
        margin = compute_decision_margin_uncertainty(probs)
        expected = np.array([[0.00, 0.40], [0.40, 0.10]])
        np.testing.assert_allclose(margin, expected, atol=1e-5)

