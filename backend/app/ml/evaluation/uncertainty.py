"""Uncertainty estimation metrics for multi-label olfactory perception models."""

from typing import Any, Dict, List, Optional
import numpy as np


def compute_predictive_entropy(y_prob: np.ndarray, eps: float = 1e-7) -> Dict[str, Any]:
    """Compute Shannon binary entropy across multi-label probability predictions.

    For each binary prediction p in [0, 1]:
        H(p) = - p * log2(p) - (1 - p) * log2(1 - p)
    Maximum entropy H=1.0 occurs at p=0.50 (maximum ambiguity).
    Minimum entropy H=0.0 occurs at p=0.0 or p=1.0 (maximum certainty).

    Args:
        y_prob: Predicted probability matrix of shape (n_samples, n_classes).
        eps: Small epsilon for numerical stability.

    Returns:
        Dictionary of molecule-level and class-level predictive uncertainty statistics.
    """
    clipped_p = np.clip(y_prob, eps, 1.0 - eps)
    entropy_matrix = - (clipped_p * np.log2(clipped_p) + (1.0 - clipped_p) * np.log2(1.0 - clipped_p))

    # Molecule-level statistics
    mean_mol_entropy = np.mean(entropy_matrix, axis=1)
    max_mol_entropy = np.max(entropy_matrix, axis=1)

    # Class-level statistics
    mean_class_entropy = np.mean(entropy_matrix, axis=0)

    # Dataset distribution quantiles
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    quantile_vals = np.quantile(mean_mol_entropy, quantiles).tolist()

    return {
        "mean_dataset_entropy": float(np.mean(entropy_matrix)),
        "median_dataset_entropy": float(np.median(mean_mol_entropy)),
        "std_dataset_entropy": float(np.std(mean_mol_entropy)),
        "molecule_mean_entropy": mean_mol_entropy.tolist(),
        "molecule_max_entropy": max_mol_entropy.tolist(),
        "class_mean_entropy": mean_class_entropy.tolist(),
        "quantiles": {
            f"q_{int(q*100)}": float(val) for q, val in zip(quantiles, quantile_vals)
        },
    }


def compute_decision_margin_uncertainty(y_prob: np.ndarray) -> np.ndarray:
    """Compute margin distance to decision boundary |p - 0.5|.

    Values near 0 indicate highest decision uncertainty; values near 0.5 indicate certainty.
    """
    return np.abs(y_prob - 0.5).astype(np.float32)

