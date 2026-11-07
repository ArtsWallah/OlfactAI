"""Preprocessing transformations for molecular descriptor features."""

from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Standardizes molecular physicochemical descriptor features."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """Fit scaler on feature matrix."""
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before transforming data.")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

