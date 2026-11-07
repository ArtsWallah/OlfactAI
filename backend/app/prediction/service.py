"""Molecular smell prediction service interface."""

from typing import Dict, List, Optional
from backend.app.core.exceptions import ModelNotTrainedError
from backend.app.ml.features.pipeline import MolecularFeaturePipeline
from backend.app.ml.models.base import BaseSmellModel


class SmellPredictionService:
    """Orchestrates smell prediction pipeline for chemical molecules."""

    def __init__(self, model: Optional[BaseSmellModel] = None):
        self.model = model
        self.feature_pipeline = MolecularFeaturePipeline()

    def predict_odor(self, smiles: str) -> Dict[str, float]:
        """Predict smell descriptors and probability scores for a molecule.

        Raises:
            ModelNotTrainedError: When no trained model is available.
        """
        if self.model is None or not self.model.is_trained:
            raise ModelNotTrainedError(
                model_name=self.model.name if self.model else "SmellPredictor",
                message=(
                    "No trained odor prediction model is loaded. Models must be trained "
                    "on verified olfactory datasets before generating predictions."
                ),
            )

        features = self.feature_pipeline.transform_single(smiles).reshape(1, -1)
        probs = self.model.predict_proba(features)[0]
        return {label: float(prob) for label, prob in zip(self.model.target_labels, probs)}

