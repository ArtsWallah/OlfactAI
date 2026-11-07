"""XGBoost gradient boosting architecture for odor property prediction."""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from xgboost import XGBClassifier

from backend.app.core.logging import get_logger
from backend.app.ml.models.base import BaseSmellModel

logger = get_logger("ml.models.xgboost")


class ConstantPredictor:
    """Fallback predictor for single-class targets (e.g. 0 positive instances in training fold)."""

    def __init__(self, constant_value: int = 0):
        self.constant_value = constant_value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.constant_value, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        prob = float(self.constant_value)
        return np.full(X.shape[0], prob, dtype=np.float32)


class XGBoostSmellModel(BaseSmellModel):
    """Multi-label gradient boosting classifier using CPU histogram tree construction."""

    def __init__(
        self,
        name: str = "xgboost_odor",
        version: str = "0.1.0",
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = 4,
    ):
        super().__init__(name=name, version=version)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.train_time_sec: float = 0.0
        self.inference_time_sec: float = 0.0

        self.estimators: List[Union[XGBClassifier, ConstantPredictor]] = []
        self.scale_pos_weights: List[float] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "XGBoostSmellModel":
        """Train per-label XGBoost binary classifiers with positive class weighting."""
        start = time.perf_counter()
        n_samples, n_classes = y.shape
        self.target_labels = target_labels or [f"odor_{i}" for i in range(n_classes)]
        self.estimators = []
        self.scale_pos_weights = []

        logger.info(
            f"Training XGBoostSmellModel ({n_classes} labels, tree_method='hist') "
            f"on {n_samples} samples..."
        )

        for col_idx in range(n_classes):
            y_col = y[:, col_idx]
            pos_count = int(np.sum(y_col == 1))
            neg_count = int(np.sum(y_col == 0))

            if pos_count == 0:
                logger.warning(f"Label '{self.target_labels[col_idx]}' has 0 positive samples; using ConstantPredictor(0)")
                self.estimators.append(ConstantPredictor(0))
                self.scale_pos_weights.append(1.0)
                continue
            elif neg_count == 0:
                logger.warning(f"Label '{self.target_labels[col_idx]}' has 0 negative samples; using ConstantPredictor(1)")
                self.estimators.append(ConstantPredictor(1))
                self.scale_pos_weights.append(1.0)
                continue

            # Scale positive weight: ratio of negative to positive samples
            spw = float(neg_count / max(pos_count, 1))
            self.scale_pos_weights.append(spw)

            clf = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                scale_pos_weight=spw,
                tree_method="hist",
                eval_metric="logloss",
                random_state=self.random_state + col_idx,
                n_jobs=self.n_jobs,
            )
            clf.fit(X, y_col)
            self.estimators.append(clf)

        self.is_trained = True
        self.train_time_sec = float(time.perf_counter() - start)
        logger.info(f"XGBoostSmellModel training completed in {self.train_time_sec:.2f}s")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability scores for each odor class."""
        self._check_is_trained()
        start = time.perf_counter()
        n_samples = X.shape[0]
        n_classes = len(self.estimators)
        proba_matrix = np.zeros((n_samples, n_classes), dtype=np.float32)

        for col_idx, est in enumerate(self.estimators):
            if isinstance(est, ConstantPredictor):
                proba_matrix[:, col_idx] = est.predict_proba(X)
            else:
                prob = est.predict_proba(X)
                if prob.shape[1] == 2:
                    proba_matrix[:, col_idx] = prob[:, 1]
                else:
                    proba_matrix[:, col_idx] = prob[:, 0]

        self.inference_time_sec = float(time.perf_counter() - start)
        return proba_matrix

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary multi-label outputs using standard 0.5 threshold."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def save(self, path: Path) -> Path:
        """Serialize XGBoost model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "version": self.version,
            "target_labels": self.target_labels,
            "is_trained": self.is_trained,
            "train_time_sec": self.train_time_sec,
            "inference_time_sec": self.inference_time_sec,
            "scale_pos_weights": self.scale_pos_weights,
            "estimators": self.estimators,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }
        joblib.dump(payload, path, compress=3)
        logger.info(f"XGBoost model saved to {path}")
        return path

    def load(self, path: Path) -> "XGBoostSmellModel":
        """Load serialized model from disk."""
        payload = joblib.load(path)
        self.name = payload["name"]
        self.version = payload["version"]
        self.target_labels = payload["target_labels"]
        self.is_trained = payload["is_trained"]
        self.train_time_sec = payload.get("train_time_sec", 0.0)
        self.inference_time_sec = payload.get("inference_time_sec", 0.0)
        self.scale_pos_weights = payload.get("scale_pos_weights", [])
        self.estimators = payload["estimators"]
        self.n_estimators = payload.get("n_estimators", 100)
        self.max_depth = payload.get("max_depth", 5)
        self.learning_rate = payload.get("learning_rate", 0.05)
        return self
