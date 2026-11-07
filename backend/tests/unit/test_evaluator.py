"""Unit tests for multi-label classification evaluator."""

import numpy as np
from backend.app.ml.evaluation.metrics import MultiLabelEvaluator


class TestMultiLabelEvaluator:
    """Test suite for evaluation metric calculations."""

    def test_evaluate_perfect_predictions(self):
        """Verify metric calculation on perfect match."""
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        metrics = MultiLabelEvaluator.evaluate(y_true, y_pred)

        assert metrics["subset_accuracy"] == 1.0
        assert metrics["hamming_loss"] == 0.0
        assert metrics["f1_micro"] == 1.0
        assert metrics["f1_macro"] == 1.0

