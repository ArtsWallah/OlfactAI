"""Preprocessing transformations for molecular descriptor features."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Standardizes molecular physicochemical descriptor features (backward compatible)."""

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


class OlfactoryFeatureScaler:
    """Leakage-safe scaler for hybrid molecular feature sets.

    Standardizes continuous physicochemical descriptors using StandardScaler
    while strictly preserving binary Morgan fingerprint bits as 0.0/1.0 without scaling.
    Excludes identifiers such as 'molecule_id' and 'canonical_smiles'.
    """

    DEFAULT_DESCRIPTOR_COLS = [
        "molecular_weight",
        "logp",
        "tpsa",
        "h_bond_donors",
        "h_bond_acceptors",
        "rotatable_bonds",
        "ring_count",
        "heavy_atom_count",
        "fraction_csp3",
    ]

    def __init__(
        self,
        descriptor_cols: Optional[List[str]] = None,
        num_continuous: int = 9,
    ):
        self.descriptor_cols = descriptor_cols or self.DEFAULT_DESCRIPTOR_COLS
        self.num_continuous = num_continuous
        self.scaler = StandardScaler()
        self.is_fitted: bool = False
        self.feature_names: Optional[List[str]] = None

    def _extract_arrays(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Separate continuous descriptors from binary fingerprint bits."""
        if isinstance(X, pd.DataFrame):
            # Drop identifiers if present
            feature_df = X.drop(
                columns=[col for col in ["molecule_id", "canonical_smiles"] if col in X.columns]
            )
            desc_cols = [c for c in self.descriptor_cols if c in feature_df.columns]
            fp_cols = [c for c in feature_df.columns if c not in desc_cols]
            self.feature_names = desc_cols + fp_cols

            desc_arr = feature_df[desc_cols].to_numpy(dtype=np.float32)
            fp_arr = feature_df[fp_cols].to_numpy(dtype=np.float32)
            return desc_arr, fp_arr
        elif isinstance(X, np.ndarray):
            desc_arr = X[:, : self.num_continuous].astype(np.float32)
            fp_arr = X[:, self.num_continuous :].astype(np.float32)
            return desc_arr, fp_arr
        else:
            raise TypeError(f"Unsupported data type for X: {type(X)}")

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "OlfactoryFeatureScaler":
        """Fit StandardScaler strictly on continuous descriptors of training split."""
        desc_arr, _ = self._extract_arrays(X)
        self.scaler.fit(desc_arr)
        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Scale continuous descriptors and concatenate unscaled fingerprint bits."""
        if not self.is_fitted:
            raise RuntimeError("OlfactoryFeatureScaler must be fitted before transforming data.")
        desc_arr, fp_arr = self._extract_arrays(X)
        desc_scaled = self.scaler.transform(desc_arr).astype(np.float32)
        return np.hstack([desc_scaled, fp_arr])

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit on training continuous features and return transformed hybrid matrix."""
        return self.fit(X).transform(X)

    def save(self, path: Path) -> Path:
        """Serialize fitted scaler to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "is_fitted": self.is_fitted,
                "descriptor_cols": self.descriptor_cols,
                "num_continuous": self.num_continuous,
                "feature_names": self.feature_names,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "OlfactoryFeatureScaler":
        """Load fitted scaler from disk."""
        data = joblib.load(path)
        instance = cls(
            descriptor_cols=data["descriptor_cols"],
            num_continuous=data["num_continuous"],
        )
        instance.scaler = data["scaler"]
        instance.is_fitted = data["is_fitted"]
        instance.feature_names = data.get("feature_names")
        return instance
