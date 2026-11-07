"""Random Forest multi-label smell prediction model architecture."""

from pathlib import Path
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
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(name=name, version=version)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
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
        logger.info(f"Training RandomForestSmellModel on {X.shape[0]} samples...")
        self.estimator.fit(X, y)
        self.is_trained = True
        self.target_labels = target_labels or [f"odor_{i}" for i in range(y.shape[1])]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary multi-label odor attributes."""
        self._check_is_trained()
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for each odor attribute."""
        self._check_is_trained()
        proba_list = self.estimator.predict_proba(X)
        if isinstance(proba_list, list):
            # For multi-output, each element is (n_samples, 2), extract positive class
            return np.column_stack([p[:, 1] for p in proba_list])
        return proba_list

    def save(self, path: Path) -> Path:
        """Serialize Random Forest model to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "version": self.version,
            "target_labels": self.target_labels,
            "is_trained": self.is_trained,
            "estimator": self.estimator,
        }
        joblib.dump(payload, path)
        logger.info(f"Model saved to {path}")
        return path

    def load(self, path: Path) -> "RandomForestSmellModel":
        """Load Random Forest model from file."""
        payload = joblib.load(path)
        self.name = payload["name"]
        self.version = payload["version"]
        self.target_labels = payload["target_labels"]
        self.is_trained = payload["is_trained"]
        self.estimator = payload["estimator"]
        return self

