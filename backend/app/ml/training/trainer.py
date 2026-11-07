"""Training orchestration and cross-validation interface."""

from typing import Any, Dict, List, Optional
import numpy as np

from backend.app.core.logging import get_logger
from backend.app.ml.models.base import BaseSmellModel

logger = get_logger("ml.training.trainer")


class SmellModelTrainer:
    """Orchestrates model fitting, hyperparameter evaluation, and persistence."""

    def __init__(self, model: BaseSmellModel):
        self.model = model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        target_labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseSmellModel:
        """Execute model training on CPU."""
        logger.info(
            f"Initiating training for '{self.model.name}' on {X_train.shape[0]} samples "
            f"with {y_train.shape[1]} odor labels."
        )
        self.model.fit(X_train, y_train, target_labels=target_labels, **kwargs)
        return self.model

