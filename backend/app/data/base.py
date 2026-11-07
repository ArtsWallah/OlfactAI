"""Abstract data interfaces for molecular odor datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd


class BaseDatasetLoader(ABC):
    """Abstract interface for reading, parsing, and validating odor datasets."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    @abstractmethod
    def load(self, file_path: Path) -> pd.DataFrame:
        """Load dataset from disk into a Pandas DataFrame."""
        pass

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        """Verify presence of required columns (e.g., SMILES, odor labels)."""
        pass

    @abstractmethod
    def get_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for the dataset."""
        pass

