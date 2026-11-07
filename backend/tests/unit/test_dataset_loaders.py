"""Unit tests for dataset loading, validation, and bundle utilities."""

import pytest
import pandas as pd
from pathlib import Path

from backend.app.core.exceptions import ResourceNotFoundError
from backend.app.data.loaders import OlfactoryDatasetLoader
from backend.tests.fixtures.synthetic_datasets import get_synthetic_raw_df


class TestOlfactoryDatasetLoader:
    """Test suite for OlfactoryDatasetLoader functionality."""

    def test_load_valid_csv(self, tmp_path: Path):
        """Verify successful loading of valid CSV file."""
        csv_file = tmp_path / "test_data.csv"
        df = get_synthetic_raw_df()
        df.to_csv(csv_file, index=False)

        loader = OlfactoryDatasetLoader()
        loaded_df = loader.load(csv_file)

        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)

    def test_load_nonexistent_file_raises_resource_not_found(self, tmp_path: Path):
        """Verify ResourceNotFoundError is raised when file does not exist."""
        loader = OlfactoryDatasetLoader()
        fake_path = tmp_path / "does_not_exist.csv"

        with pytest.raises(ResourceNotFoundError):
            loader.load(fake_path)

    def test_validate_missing_smiles_column_raises_value_error(self, tmp_path: Path):
        """Verify ValueError is raised when dataset lacks any molecular structure column."""
        csv_file = tmp_path / "bad_data.csv"
        bad_df = pd.DataFrame({"id": [1, 2], "value": [10.5, 20.2]})
        bad_df.to_csv(csv_file, index=False)

        loader = OlfactoryDatasetLoader()
        with pytest.raises(ValueError, match="missing molecular structure column"):
            loader.load(csv_file)

    def test_custom_required_columns(self, tmp_path: Path):
        """Verify custom required columns validation."""
        csv_file = tmp_path / "custom_data.csv"
        df = pd.DataFrame({"smiles": ["CCO"], "custom_target": [1]})
        df.to_csv(csv_file, index=False)

        # Passes when column is present
        loader_ok = OlfactoryDatasetLoader(required_columns=["smiles", "custom_target"])
        assert loader_ok.load(csv_file) is not None

        # Fails when expected column is absent
        loader_fail = OlfactoryDatasetLoader(required_columns=["smiles", "missing_col"])
        with pytest.raises(ValueError, match="missing required column"):
            loader_fail.load(csv_file)

    def test_get_summary(self):
        """Verify summary statistics output."""
        df = get_synthetic_raw_df()
        loader = OlfactoryDatasetLoader()
        summary = loader.get_summary(df)

        assert summary["total_molecules"] == 10
        assert summary["structure_column"] == "nonStereoSMILES"
        assert summary["null_smiles_count"] == 0

