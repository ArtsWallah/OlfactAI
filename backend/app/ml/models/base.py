"""Abstract base class for all OlfactAI smell prediction models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from backend.app.core.exceptions import ModelNotTrainedError


class BaseSmellModel(ABC):
    """Abstract interface defining the contract for smell prediction estimators."""

    def __init__(self, name: str, version: str = "0.1.0"):
        self.name = name
        self.version = version
        self.is_trained: bool = False
        self.target_labels: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def _check_is_trained(self) -> None:
        """Raise exception if model has not been trained."""
        if not self.is_trained:
            raise ModelNotTrainedError(
                model_name=self.name,
                message=(
                    f"Model '{self.name}' (v{self.version}) is an untrained placeholder. "
                    "Train this model on an olfactory dataset before running inference."
                ),
            )

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "BaseSmellModel":
        """Train model on molecular feature vectors and binary odor label matrix."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary multi-label indicators for molecules."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for odor attributes."""
        pass

    @abstractmethod
    def save(self, path: Path) -> Path:
        """Serialize model weights and metadata to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> "BaseSmellModel":
        """Deserialize model weights and metadata from disk."""
        pass

