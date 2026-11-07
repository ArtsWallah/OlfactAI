"""XGBoost gradient boosting architecture for odor property prediction."""

from pathlib import Path
from typing import Any, List, Optional
import joblib
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from backend.app.core.logging import get_logger
from backend.app.ml.models.base import BaseSmellModel

logger = get_logger("ml.models.xgboost")


class XGBoostSmellModel(BaseSmellModel):
    """Multi-label gradient boosting classifier using XGBoost."""

    def __init__(
        self,
        name: str = "xgboost_odor",
        version: str = "0.1.0",
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        super().__init__(name=name, version=version)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        base_estimator = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            tree_method="hist",  # Efficient CPU histogram algorithm
            n_jobs=-1,
            eval_metric="logloss",
        )
        self.estimator = MultiOutputClassifier(base_estimator, n_jobs=-1)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "XGBoostSmellModel":
        """Train multi-output XGBoost classifier."""
        logger.info(f"Training XGBoostSmellModel on {X.shape[0]} samples...")
        self.estimator.fit(X, y)
        self.is_trained = True
        self.target_labels = target_labels or [f"odor_{i}" for i in range(y.shape[1])]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary multi-label outputs."""
        self._check_is_trained()
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability scores for each odor class."""
        self._check_is_trained()
        proba_list = self.estimator.predict_proba(X)
        if isinstance(proba_list, list):
            return np.column_stack([p[:, 1] for p in proba_list])
        return proba_list

    def save(self, path: Path) -> Path:
        """Serialize XGBoost model to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "version": self.version,
            "target_labels": self.target_labels,
            "is_trained": self.is_trained,
            "estimator": self.estimator,
        }
        joblib.dump(payload, path)
        logger.info(f"XGBoost model saved to {path}")
        return path

    def load(self, path: Path) -> "XGBoostSmellModel":
        """Load serialized model from disk."""
        payload = joblib.load(path)
        self.name = payload["name"]
        self.version = payload["version"]
        self.target_labels = payload["target_labels"]
        self.is_trained = payload["is_trained"]
        self.estimator = payload["estimator"]
        return self

