"""Dataset loaders for raw, processed, and benchmark olfactory datasets."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from backend.app.core.exceptions import ResourceNotFoundError
from backend.app.core.logging import get_logger
from backend.app.data.base import BaseDatasetLoader

logger = get_logger("data.loaders")


class OlfactoryDatasetLoader(BaseDatasetLoader):
    """Loader for tabular odor datasets supporting raw, processed, and split formats."""

    DEFAULT_SMILES_COLUMNS = ["canonical_smiles", "smiles", "nonStereoSMILES", "input_smiles"]

    def __init__(self, data_dir: Optional[Path] = None, required_columns: Optional[List[str]] = None):
        super().__init__(data_dir=data_dir or Path("."))
        self.required_columns = required_columns

    def load(self, file_path: Path) -> pd.DataFrame:
        """Load tabular dataset from CSV or Parquet file format.

        Raises:
            ResourceNotFoundError: If file does not exist.
            ValueError: If file format is unsupported or required columns missing.
        """
        if not file_path.exists():
            raise ResourceNotFoundError("Dataset file", str(file_path))

        logger.info(f"Loading dataset from: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif suffix in [".csv", ".tsv", ".txt"]:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(file_path, sep=sep)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        self.validate(df)
        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Verify presence of required columns."""
        if self.required_columns:
            missing = [col for col in self.required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Dataset missing required column(s): {missing}")
            return True

        # Default: ensure at least one recognized molecular structure column exists
        has_smiles_col = any(col in df.columns for col in self.DEFAULT_SMILES_COLUMNS)
        if not has_smiles_col:
            raise ValueError(
                f"Dataset missing molecular structure column. "
                f"Expected one of: {self.DEFAULT_SMILES_COLUMNS}"
            )
        return True

    def get_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for the dataset."""
        smi_col = next((c for c in self.DEFAULT_SMILES_COLUMNS if c in df.columns), None)
        null_count = int(df[smi_col].isnull().sum()) if smi_col else 0

        return {
            "total_molecules": len(df),
            "columns_count": len(df.columns),
            "columns": list(df.columns),
            "structure_column": smi_col,
            "null_smiles_count": null_count,
        }

    @staticmethod
    def load_processed_bundle(processed_dir: Path) -> Dict[str, Any]:
        """Load molecules, features, labels, and metadata from processed directory."""
        if not processed_dir.exists():
            raise ResourceNotFoundError("Processed directory", str(processed_dir))

        molecules_file = processed_dir / "molecules.csv"
        features_file = processed_dir / "molecular_features.csv"
        labels_file = processed_dir / "odor_labels.csv"
        meta_file = processed_dir / "dataset_metadata.json"

        bundle = {
            "molecules": pd.read_csv(molecules_file) if molecules_file.exists() else None,
            "features": pd.read_csv(features_file) if features_file.exists() else None,
            "labels": pd.read_csv(labels_file) if labels_file.exists() else None,
            "metadata": None,
        }

        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                bundle["metadata"] = json.load(f)

        return bundle

    @staticmethod
    def load_split_partition(
        processed_dir: Path,
        split_type: str = "scaffold",
        partition: str = "train",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load feature and label matrices filtered to a specific split partition.

        Args:
            processed_dir: Directory containing processed CSVs.
            split_type: Either 'scaffold' or 'random'.
            partition: 'train', 'val', or 'test'.

        Returns:
            Tuple of (features_df, labels_df) corresponding to the partition.
        """
        molecules_df = pd.read_csv(processed_dir / "molecules.csv")
        features_df = pd.read_csv(processed_dir / "molecular_features.csv")
        labels_df = pd.read_csv(processed_dir / "odor_labels.csv")

        split_col = f"{split_type}_split"
        if split_col not in molecules_df.columns:
            raise ValueError(f"Split column '{split_col}' not found in molecules.csv")

        partition_mol_ids = set(molecules_df[molecules_df[split_col] == partition]["molecule_id"])

        part_features = features_df[features_df["molecule_id"].isin(partition_mol_ids)].copy()
        part_labels = labels_df[labels_df["molecule_id"].isin(partition_mol_ids)].copy()

        return part_features, part_labels
