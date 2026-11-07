"""Model explainability and SHAP attribution interface."""

from typing import Any, Dict, List, Optional
import numpy as np

from backend.app.core.exceptions import ModelNotTrainedError
from backend.app.ml.models.base import BaseSmellModel


class SmellExplainerService:
    """Interprets model predictions via SHAP feature attributions and sub-structural contributions."""

    def __init__(self, model: Optional[BaseSmellModel] = None):
        self.model = model

    def explain_prediction(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """Explain model prediction for a molecular feature vector.

        Raises:
            ModelNotTrainedError: When model is untrained.
        """
        if self.model is None or not self.model.is_trained:
            raise ModelNotTrainedError(
                model_name=self.model.name if self.model else "SmellExplainer",
                message="Cannot compute explanations without a trained model.",
            )

        raise NotImplementedError("SHAP explainer integration will be configured upon model training.")

