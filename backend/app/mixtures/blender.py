"""Molecular mixture odor blending and additive/non-linear simulation interface."""

from typing import Dict, List, Tuple
import numpy as np

from backend.app.chemistry.parser import MoleculeParser
from backend.app.ml.features.pipeline import MolecularFeaturePipeline


class MixtureBlender:
    """Simulates chemical odor mixtures by combining weighted molecular representations."""

    def __init__(self):
        self.parser = MoleculeParser()
        self.pipeline = MolecularFeaturePipeline()

    def blend_representations(
        self, components: List[Tuple[str, float]]
    ) -> np.ndarray:
        """Blend molecular feature vectors based on relative concentrations/ratios.

        Args:
            components: List of tuples (smiles, ratio/weight).

        Returns:
            Weighted aggregate molecular feature vector.
        """
        if not components:
            raise ValueError("Mixture components list cannot be empty.")

        total_weight = sum(weight for _, weight in components)
        if total_weight <= 0:
            raise ValueError("Total concentration weight must be positive.")

        normalized_components = [(s, w / total_weight) for s, w in components]

        vectors = []
        weights = []
        for smiles, norm_w in normalized_components:
            vec = self.pipeline.transform_single(smiles)
            vectors.append(vec)
            weights.append(norm_w)

        vectors_matrix = np.array(vectors)
        weights_array = np.array(weights).reshape(-1, 1)

        # Linear combination of representation vectors
        blended = np.sum(vectors_matrix * weights_array, axis=0)
        return blended

