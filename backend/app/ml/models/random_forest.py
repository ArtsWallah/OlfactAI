"""Random Forest multi-label smell prediction model architecture."""

from pathlib import Path
import time
from typing import Any, List, Optional
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from backend.app.core.logging import get_logger
from backend.app.ml.models.base import BaseSmellModel

logger = get_logger("ml.models.random_forest")


class RandomForestSmellModel(BaseSmellModel):
    """Multi-label Random Forest classifier for molecular odor attributes."""

    def __init__(
        self,
        name: str = "random_forest_odor",
        version: str = "0.1.0",
        n_estimators: int = 100,
        max_depth: Optional[int] = 20,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(name=name, version=version)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.train_time_sec: float = 0.0
        self.inference_time_sec: float = 0.0

        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "RandomForestSmellModel":
        """Fit Random Forest on feature matrix X and multi-label targets y."""
        start = time.perf_counter()
        logger.info(
            f"Training RandomForestSmellModel ({self.n_estimators} trees, max_depth={self.max_depth}) "
            f"on {X.shape[0]} samples with {y.shape[1]} labels..."
        )
        self.estimator.fit(X, y)
        self.is_trained = True
        self.target_labels = target_labels or [f"odor_{i}" for i in range(y.shape[1])]
        self.train_time_sec = float(time.perf_counter() - start)
        logger.info(f"RandomForestSmellModel training completed in {self.train_time_sec:.2f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary multi-label odor attributes."""
        self._check_is_trained()
        start = time.perf_counter()
        preds = self.estimator.predict(X)
        self.inference_time_sec = float(time.perf_counter() - start)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for each odor attribute."""
        self._check_is_trained()
        start = time.perf_counter()
        proba_list = self.estimator.predict_proba(X)

        if isinstance(proba_list, list):
            n_samples = X.shape[0]
            n_classes = len(proba_list)
            prob_matrix = np.zeros((n_samples, n_classes), dtype=np.float32)

            for col_idx, p in enumerate(proba_list):
                classes = self.estimator.classes_[col_idx]
                if len(classes) == 2:
                    prob_matrix[:, col_idx] = p[:, 1]
                elif len(classes) == 1 and classes[0] == 1:
                    prob_matrix[:, col_idx] = 1.0
                else:
                    prob_matrix[:, col_idx] = 0.0

            self.inference_time_sec = float(time.perf_counter() - start)
            return prob_matrix

        self.inference_time_sec = float(time.perf_counter() - start)
        return proba_list.astype(np.float32)

    def save(self, path: Path) -> Path:
        """Serialize Random Forest model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "version": self.version,
            "target_labels": self.target_labels,
            "is_trained": self.is_trained,
            "train_time_sec": self.train_time_sec,
            "inference_time_sec": self.inference_time_sec,
            "estimator": self.estimator,
        }
        joblib.dump(payload, path, compress=3)
        logger.info(f"Model saved to {path}")
        return path

    def load(self, path: Path) -> "RandomForestSmellModel":
        """Load Random Forest model from file."""
        payload = joblib.load(path)
        self.name = payload["name"]
        self.version = payload["version"]
        self.target_labels = payload["target_labels"]
        self.is_trained = payload["is_trained"]
        self.train_time_sec = payload.get("train_time_sec", 0.0)
        self.inference_time_sec = payload.get("inference_time_sec", 0.0)
        self.estimator = payload["estimator"]
        return self
