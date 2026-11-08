"""Probability calibration metrics and temperature scaling for multi-label olfactory models."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from backend.app.core.logging import get_logger

logger = get_logger("ml.evaluation.calibration")


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute overall multi-label Brier score (mean squared error of probability predictions).

    Args:
        y_true: Ground truth binary matrix of shape (n_samples, n_classes).
        y_prob: Predicted probability scores of shape (n_samples, n_classes).

    Returns:
        Mean squared error across all individual binary predictions in [0, 1].
    """
    return float(np.mean((y_prob - y_true) ** 2))


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, Dict[str, Any]]:
    """Compute Expected Calibration Error (ECE) and binning statistics.

    Bins all probability predictions into equal-width intervals in [0, 1],
    computing the absolute difference between average confidence and empirical accuracy.

    Args:
        y_true: Ground truth binary array/matrix.
        y_prob: Predicted probabilities array/matrix.
        n_bins: Number of equal-width bins (default 10).

    Returns:
        Tuple of (ece_score, bin_statistics_dict).
    """
    y_true_flat = y_true.flatten().astype(float)
    y_prob_flat = y_prob.flatten().astype(float)

    total_predictions = len(y_prob_flat)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    bin_stats: List[Dict[str, Any]] = []

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Handle inclusive upper edge on final bin
        if i == n_bins - 1:
            in_bin = (y_prob_flat >= lower) & (y_prob_flat <= upper)
        else:
            in_bin = (y_prob_flat >= lower) & (y_prob_flat < upper)

        count = int(np.sum(in_bin))

        if count > 0:
            avg_confidence = float(np.mean(y_prob_flat[in_bin]))
            empirical_accuracy = float(np.mean(y_true_flat[in_bin]))
            bin_weight = count / total_predictions
            abs_err = abs(empirical_accuracy - avg_confidence)

            ece += bin_weight * abs_err

            bin_stats.append({
                "bin_idx": i,
                "range": [float(lower), float(upper)],
                "count": count,
                "weight": float(bin_weight),
                "avg_confidence": avg_confidence,
                "empirical_accuracy": empirical_accuracy,
                "calibration_gap": float(abs_err),
            })
        else:
            bin_stats.append({
                "bin_idx": i,
                "range": [float(lower), float(upper)],
                "count": 0,
                "weight": 0.0,
                "avg_confidence": float((lower + upper) / 2.0),
                "empirical_accuracy": 0.0,
                "calibration_gap": 0.0,
            })

    result_dict = {
        "ece": float(ece),
        "n_bins": n_bins,
        "total_predictions": total_predictions,
        "bins": bin_stats,
    }

    return float(ece), result_dict


class TemperatureScaler:
    """Post-hoc probability calibration via scalar temperature scaling.

    Optimizes temperature parameter T > 0 on validation set logits using
    gradient descent to minimize multi-label binary cross-entropy.
    """

    def __init__(self, lr: float = 0.01, max_iter: int = 100):
        self.lr = lr
        self.max_iter = max_iter
        self.temperature: float = 1.0
        self.is_fitted: bool = False

    def fit(self, val_logits: np.ndarray, val_y_true: np.ndarray) -> "TemperatureScaler":
        """Fit optimal temperature on validation set logits and true labels."""
        logits_tensor = torch.tensor(val_logits, dtype=torch.float32)
        y_tensor = torch.tensor(val_y_true, dtype=torch.float32)

        # Optimize log_temperature to strictly enforce T > 0
        log_temp = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        optimizer = Adam([log_temp], lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        best_t = 1.0

        for _ in range(self.max_iter):
            optimizer.zero_grad()
            t = torch.exp(log_temp)
            scaled_logits = logits_tensor / t
            loss = criterion(scaled_logits, y_tensor)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_t = float(t.item())

        self.temperature = float(best_t)
        self.is_fitted = True
        logger.info(f"Fitted TemperatureScaler: T = {self.temperature:.4f} (Val BCE = {best_loss:.4f})")
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Calibrate logits by dividing by temperature T and applying sigmoid."""
        scaled = logits / max(self.temperature, 1e-4)
        probs = 1.0 / (1.0 + np.exp(-np.clip(scaled, -20.0, 20.0)))
        return probs.astype(np.float32)

    def save(self, path: Path) -> Path:
        """Serialize temperature configuration to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "temperature": self.temperature,
                "is_fitted": self.is_fitted,
            }, f, indent=2)
        return path

    @classmethod
    def load(cls, path: Path) -> "TemperatureScaler":
        """Load temperature configuration from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scaler = cls()
        scaler.temperature = float(data["temperature"])
        scaler.is_fitted = bool(data["is_fitted"])
        return scaler

