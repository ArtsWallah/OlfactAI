"""Dataset loaders for raw, processed, and external olfactory benchmarks."""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from backend.app.core.exceptions import ResourceNotFoundError
from backend.app.core.logging import get_logger
from backend.app.data.base import BaseDatasetLoader

logger = get_logger("data.loaders")


class OlfactoryDatasetLoader(BaseDatasetLoader):
    """Loader for CSV/Parquet odor datasets containing SMILES and multi-label descriptors."""

    REQUIRED_COLUMNS: List[str] = ["smiles"]

    def load(self, file_path: Path) -> pd.DataFrame:
        """Load tabular dataset from CSV or Parquet file format."""
        if not file_path.exists():
            raise ResourceNotFoundError("Dataset file", str(file_path))

        logger.info(f"Loading dataset from: {file_path}")
        if file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix.lower() in [".csv", ".tsv", ".txt"]:
            sep = "\t" if file_path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(file_path, sep=sep)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        self.validate(df)
        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Verify presence of required columns."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required column(s): {missing}")
        return True

    def get_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return dataset row count and column overview."""
        return {
            "total_molecules": len(df),
            "columns": list(df.columns),
            "null_smiles_count": int(df["smiles"].isnull().sum()),
        }

