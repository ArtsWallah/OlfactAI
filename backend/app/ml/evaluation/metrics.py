"""Evaluation metrics for multi-label molecular odor classification."""

from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MultiLabelEvaluator:
    """Calculates multi-label classification performance benchmarks."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        target_labels: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute comprehensive multi-label classification metrics.

        Args:
            y_true: Ground truth binary matrix (n_samples, n_classes).
            y_pred: Predicted binary matrix (n_samples, n_classes).
            y_prob: Predicted probability scores (optional).
            target_labels: Names of odor labels.

        Returns:
            Dictionary of calculated metrics.
        """
        metrics = {
            "subset_accuracy": float(accuracy_score(y_true, y_pred)),
            "hamming_loss": float(hamming_loss(y_true, y_pred)),
            "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }

        if y_prob is not None:
            try:
                metrics["roc_auc_micro"] = float(
                    roc_auc_score(y_true, y_prob, average="micro")
                )
                metrics["roc_auc_macro"] = float(
                    roc_auc_score(y_true, y_prob, average="macro")
                )
            except Exception:
                # Can occur if some classes have no positive samples in test fold
                pass

        return metrics

