"""Evaluation metrics for multi-label molecular odor classification."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MultiLabelEvaluator:
    """Calculates comprehensive multi-label classification benchmarks for olfactory models."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        target_labels: Optional[List[str]] = None,
        include_per_label: bool = False,
    ) -> Dict[str, Any]:
        """Compute multi-label classification metrics.

        Args:
            y_true: Ground truth binary matrix (n_samples, n_classes).
            y_pred: Predicted binary matrix (n_samples, n_classes).
            y_prob: Predicted probability scores (optional).
            target_labels: Names of odor labels.
            include_per_label: Whether to include per-label breakdown dictionary.

        Returns:
            Dictionary of calculated metrics.
        """
        n_samples, n_classes = y_true.shape
        labels = target_labels or [f"odor_{i}" for i in range(n_classes)]

        metrics: Dict[str, Any] = {
            "subset_accuracy": float(accuracy_score(y_true, y_pred)),
            "hamming_loss": float(hamming_loss(y_true, y_pred)),
            "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        # Calculate per-label metrics and identify evaluable classes (having >= 1 positive instance)
        supports = np.sum(y_true, axis=0)
        evaluable_mask = (supports > 0) & (supports < n_samples)
        metrics["evaluable_labels_count"] = int(np.sum(evaluable_mask))
        metrics["total_labels_count"] = int(n_classes)

        per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_label_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_label_rec = recall_score(y_true, y_pred, average=None, zero_division=0)

        # Probabilistic metrics
        roc_aucs: List[Optional[float]] = [None] * n_classes
        pr_aucs: List[Optional[float]] = [None] * n_classes

        if y_prob is not None:
            # Micro ROC-AUC & PR-AUC across all flattened entries
            try:
                metrics["roc_auc_micro"] = float(roc_auc_score(y_true, y_prob, average="micro"))
            except Exception:
                metrics["roc_auc_micro"] = float("nan")

            try:
                metrics["pr_auc_micro"] = float(average_precision_score(y_true, y_prob, average="micro"))
            except Exception:
                metrics["pr_auc_micro"] = float("nan")

            # Per-label ROC-AUC and PR-AUC
            valid_roc_aucs = []
            valid_pr_aucs = []
            for i in range(n_classes):
                if evaluable_mask[i]:
                    try:
                        auc_val = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
                        roc_aucs[i] = auc_val
                        valid_roc_aucs.append(auc_val)
                    except Exception:
                        roc_aucs[i] = None

                    try:
                        pr_val = float(average_precision_score(y_true[:, i], y_prob[:, i]))
                        pr_aucs[i] = pr_val
                        valid_pr_aucs.append(pr_val)
                    except Exception:
                        pr_aucs[i] = None

            metrics["roc_auc_macro"] = float(np.mean(valid_roc_aucs)) if valid_roc_aucs else float("nan")
            metrics["pr_auc_macro"] = float(np.mean(valid_pr_aucs)) if valid_pr_aucs else float("nan")
            metrics["evaluable_roc_auc_count"] = len(valid_roc_aucs)
            metrics["evaluable_pr_auc_count"] = len(valid_pr_aucs)

        if include_per_label:
            per_label_dict = {}
            for i, name in enumerate(labels):
                per_label_dict[name] = {
                    "support": int(supports[i]),
                    "evaluable": bool(evaluable_mask[i]),
                    "f1": float(per_label_f1[i]),
                    "precision": float(per_label_prec[i]),
                    "recall": float(per_label_rec[i]),
                    "roc_auc": roc_aucs[i],
                    "pr_auc": pr_aucs[i],
                }
            metrics["per_label"] = per_label_dict

        return metrics
