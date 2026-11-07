"""Threshold optimization for multi-label olfactory perception models."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import f1_score


def optimize_thresholds(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    thresholds_grid: Optional[np.ndarray] = None,
    fallback: float = 0.5,
    min_support: int = 2,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Tune per-label binary decision thresholds on validation set to maximize F1.

    Args:
        y_val_true: Ground truth binary matrix for validation set (n_samples, n_classes).
        y_val_prob: Predicted probabilities for validation set (n_samples, n_classes).
        thresholds_grid: Candidate thresholds to evaluate (defaults to 17 steps: 0.10 to 0.90 in steps of 0.05).
        fallback: Default fallback threshold if optimization is not possible.
        min_support: Minimum positive ground-truth instances required in validation set.

    Returns:
        Tuple of (optimal_thresholds_array, optimization_metadata_dict).
    """
    if thresholds_grid is None:
        thresholds_grid = np.linspace(0.1, 0.9, 17)

    n_samples, n_classes = y_val_true.shape
    optimal_thresholds = np.full(n_classes, fallback, dtype=np.float32)
    best_f1_scores = np.zeros(n_classes, dtype=np.float32)
    label_optimization_stats = {}

    for col_idx in range(n_classes):
        y_true_col = y_val_true[:, col_idx]
        y_prob_col = y_val_prob[:, col_idx]
        pos_count = int(np.sum(y_true_col))

        if pos_count < min_support:
            optimal_thresholds[col_idx] = fallback
            label_optimization_stats[str(col_idx)] = {
                "support": pos_count,
                "best_threshold": fallback,
                "best_f1": 0.0,
                "strategy": "fallback_low_support",
            }
            continue

        best_t = fallback
        best_f1 = -1.0

        for t in thresholds_grid:
            y_pred_col = (y_prob_col >= t).astype(int)
            score = f1_score(y_true_col, y_pred_col, zero_division=0)
            # Prefer threshold closer to 0.5 in case of ties
            if score > best_f1 or (np.isclose(score, best_f1) and abs(t - 0.5) < abs(best_t - 0.5)):
                best_f1 = score
                best_t = float(t)

        if best_f1 <= 0.0:
            best_t = fallback
            strategy = "fallback_zero_f1"
        else:
            strategy = "optimized"

        optimal_thresholds[col_idx] = best_t
        best_f1_scores[col_idx] = max(0.0, best_f1)
        label_optimization_stats[str(col_idx)] = {
            "support": pos_count,
            "best_threshold": best_t,
            "best_f1": float(best_f1_scores[col_idx]),
            "strategy": strategy,
        }

    metadata = {
        "mean_optimal_threshold": float(np.mean(optimal_thresholds)),
        "median_optimal_threshold": float(np.median(optimal_thresholds)),
        "min_optimal_threshold": float(np.min(optimal_thresholds)),
        "max_optimal_threshold": float(np.max(optimal_thresholds)),
        "optimized_labels_count": int(
            sum(1 for s in label_optimization_stats.values() if s["strategy"] == "optimized")
        ),
        "fallback_labels_count": int(
            sum(1 for s in label_optimization_stats.values() if s["strategy"] != "optimized")
        ),
        "details": label_optimization_stats,
    }

    return optimal_thresholds, metadata


def apply_thresholds(y_prob: np.ndarray, thresholds: Union[float, np.ndarray]) -> np.ndarray:
    """Apply uniform or per-label thresholds to probability matrix.

    Args:
        y_prob: Predicted probability matrix of shape (n_samples, n_classes).
        thresholds: Float scalar (e.g., 0.5) or 1D array of length n_classes.

    Returns:
        Binary prediction matrix of shape (n_samples, n_classes).
    """
    if isinstance(thresholds, (int, float)):
        return (y_prob >= thresholds).astype(int)
    thresholds_arr = np.asarray(thresholds)
    if thresholds_arr.ndim == 1:
        return (y_prob >= thresholds_arr[None, :]).astype(int)
    return (y_prob >= thresholds_arr).astype(int)


def save_thresholds(
    thresholds: np.ndarray,
    target_labels: List[str],
    path: Path,
    metadata: Optional[Dict] = None,
) -> Path:
    """Serialize per-label thresholds and labels to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thresholds": thresholds.tolist(),
        "target_labels": target_labels,
        "label_threshold_map": {label: float(t) for label, t in zip(target_labels, thresholds)},
        "metadata": metadata or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_thresholds(path: Path) -> Tuple[np.ndarray, List[str], Dict]:
    """Load per-label thresholds, label names, and metadata from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        np.array(data["thresholds"], dtype=np.float32),
        data["target_labels"],
        data.get("metadata", {}),
    )

