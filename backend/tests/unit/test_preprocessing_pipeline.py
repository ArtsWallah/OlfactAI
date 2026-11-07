"""Unit tests for molecular preprocessing pipeline."""

import json
from pathlib import Path
import pandas as pd
import pytest

from backend.scripts.preprocess_dataset import preprocess_dataset
from backend.tests.fixtures.synthetic_datasets import (
    get_synthetic_raw_df,
    get_synthetic_with_invalid_smiles_df,
)


class TestPreprocessingPipeline:
    """Test suite for preprocessing workflow and feature-label matrix separation."""

    def test_preprocess_synthetic_clean_dataset(self, tmp_path: Path):
        """Verify feature extraction and matrix separation on clean synthetic data."""
        raw_csv = tmp_path / "raw.csv"
        proc_dir = tmp_path / "processed"
        raw_df = get_synthetic_raw_df()
        raw_df.to_csv(raw_csv, index=False)

        artifacts = preprocess_dataset(
            raw_csv_path=raw_csv,
            processed_dir=proc_dir,
            morgan_radius=2,
            morgan_nbits=2048,
        )

        # 1. Verify all 4 artifacts created
        assert artifacts["molecules"].exists()
        assert artifacts["features"].exists()
        assert artifacts["labels"].exists()
        assert artifacts["metadata"].exists()

        mols_df = pd.read_csv(artifacts["molecules"])
        feats_df = pd.read_csv(artifacts["features"])
        labels_df = pd.read_csv(artifacts["labels"])

        # 2. Check row counts
        assert len(mols_df) == 10
        assert len(feats_df) == 10
        assert len(labels_df) == 10

        # 3. Check feature matrix structure (id + canonical_smiles + 9 desc + 2048 fp = 2059 cols)
        assert feats_df.shape[1] == 2 + 9 + 2048
        assert "fp_0" in feats_df.columns
        assert "fp_2047" in feats_df.columns
        assert "molecular_weight" in feats_df.columns

        # 4. CRITICAL: Odor labels must NOT leak into the feature matrix!
        odor_label_names = ["sweet", "fruity", "floral", "woody"]
        for odor in odor_label_names:
            assert odor not in feats_df.columns, f"Leakage violation: {odor} found in features!"
            assert odor in labels_df.columns

        # 5. Check metadata
        with open(artifacts["metadata"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["dimensions"]["total_molecules"] == 10
        assert meta["fingerprint_parameters"]["nbits"] == 2048

    def test_preprocess_handles_invalid_smiles_gracefully(self, tmp_path: Path):
        """Verify pipeline isolates invalid SMILES without crashing."""
        raw_csv = tmp_path / "raw_with_invalid.csv"
        proc_dir = tmp_path / "processed_invalid"
        df = get_synthetic_with_invalid_smiles_df()
        df.to_csv(raw_csv, index=False)

        artifacts = preprocess_dataset(
            raw_csv_path=raw_csv,
            processed_dir=proc_dir,
            morgan_radius=2,
            morgan_nbits=2048,
        )

        mols_df = pd.read_csv(artifacts["molecules"])
        # Exactly 2 valid molecules in synthetic_with_invalid_smiles_df
        assert len(mols_df) == 2
        with open(artifacts["metadata"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["dimensions"]["failed_molecules"] == 2

