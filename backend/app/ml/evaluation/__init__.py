"""Model evaluation metrics, calibration, and benchmark utilities."""

from backend.app.ml.evaluation.calibration import (
    TemperatureScaler,
    compute_brier_score,
    compute_expected_calibration_error,
)
from backend.app.ml.evaluation.metrics import MultiLabelEvaluator
from backend.app.ml.evaluation.thresholds import (
    apply_thresholds,
    load_thresholds,
    optimize_thresholds,
    save_thresholds,
)
from backend.app.ml.evaluation.uncertainty import (
    compute_decision_margin_uncertainty,
    compute_predictive_entropy,
)

__all__ = [
    "MultiLabelEvaluator",
    "apply_thresholds",
    "load_thresholds",
    "optimize_thresholds",
    "save_thresholds",
    "compute_brier_score",
    "compute_expected_calibration_error",
    "TemperatureScaler",
    "compute_predictive_entropy",
    "compute_decision_margin_uncertainty",
]
