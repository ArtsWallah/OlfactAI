"""Dummy baseline smelling models for establishing empirical lower performance bounds."""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import joblib
import numpy as np

from backend.app.core.logging import get_logger
from backend.app.ml.models.base import BaseSmellModel

logger = get_logger("ml.models.dummy")


class DummyFrequencySmellModel(BaseSmellModel):
    """Baseline multi-label model predicting prior class empirical frequencies."""

    def __init__(
        self,
        name: str = "dummy_frequency_odor",
        version: str = "0.1.0",
        strategy: str = "prior",
    ):
        super().__init__(name=name, version=version)
        self.strategy = strategy
        self.priors: Optional[np.ndarray] = None
        self.train_time_sec: float = 0.0
        self.inference_time_sec: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "DummyFrequencySmellModel":
        """Compute prior empirical frequency of each odor attribute in training split."""
        start = time.perf_counter()
        logger.info(f"Fitting DummyFrequencySmellModel on {X.shape[0]} training samples...")

        n_samples = y.shape[0]
        n_classes = y.shape[1]

        if self.strategy == "prior":
            self.priors = np.mean(y, axis=0).astype(np.float32)
        elif self.strategy == "uniform":
            self.priors = np.full(n_classes, 0.5, dtype=np.float32)
        elif self.strategy == "constant":
            self.priors = np.zeros(n_classes, dtype=np.float32)
        else:
            raise ValueError(f"Unknown dummy strategy: {self.strategy}")

        self.target_labels = target_labels or [f"odor_{i}" for i in range(n_classes)]
        self.is_trained = True
        self.train_time_sec = float(time.perf_counter() - start)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Broadcast learned prior probabilities across input samples."""
        self._check_is_trained()
        start = time.perf_counter()
        n_samples = X.shape[0]
        probs = np.tile(self.priors, (n_samples, 1))
        self.inference_time_sec = float(time.perf_counter() - start)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary indicators using standard 0.5 threshold."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def save(self, path: Path) -> Path:
        """Serialize dummy model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "version": self.version,
            "strategy": self.strategy,
            "priors": self.priors,
            "target_labels": self.target_labels,
            "is_trained": self.is_trained,
            "train_time_sec": self.train_time_sec,
            "inference_time_sec": self.inference_time_sec,
        }
        joblib.dump(payload, path)
        logger.info(f"Dummy model saved to {path}")
        return path

    def load(self, path: Path) -> "DummyFrequencySmellModel":
        """Load serialized dummy model from disk."""
        payload = joblib.load(path)
        self.name = payload["name"]
        self.version = payload["version"]
        self.strategy = payload["strategy"]
        self.priors = payload["priors"]
        self.target_labels = payload["target_labels"]
        self.is_trained = payload["is_trained"]
        self.train_time_sec = payload.get("train_time_sec", 0.0)
        self.inference_time_sec = payload.get("inference_time_sec", 0.0)
        return self

