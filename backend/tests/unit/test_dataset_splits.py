"""Unit tests for dataset splitting and scaffold leakage prevention."""

from pathlib import Path
import pytest
import pandas as pd

from backend.scripts.create_splits import (
    generate_random_split,
    generate_scaffold_split,
    audit_label_coverage,
)
from backend.tests.fixtures.synthetic_datasets import get_synthetic_raw_df


class TestDatasetSplits:
    """Test suite for dataset splitting algorithms and leakage auditing."""

    @pytest.fixture
    def sample_molecules_df(self):
        """Prepare small sample dataframe with molecule_id and canonical_smiles."""
        raw_df = get_synthetic_raw_df()
        raw_df["molecule_id"] = [f"MOL_{i+1:03d}" for i in range(len(raw_df))]
        raw_df["canonical_smiles"] = raw_df["nonStereoSMILES"]
        return raw_df

    def test_random_split_reproducibility(self, sample_molecules_df):
        """Verify identical random seed produces identical split mapping."""
        split_a = generate_random_split(sample_molecules_df, random_seed=42)
        split_b = generate_random_split(sample_molecules_df, random_seed=42)
        split_c = generate_random_split(sample_molecules_df, random_seed=999)

        assert split_a == split_b
        assert split_a != split_c

    def test_scaffold_split_leakage_free(self, sample_molecules_df):
        """Strictly verify zero scaffold overlap between train and test sets."""
        split_map, audit = generate_scaffold_split(
            sample_molecules_df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )

        assert len(split_map) == len(sample_molecules_df)
        assert audit["scaffold_leakage_detected"] is False
        assert audit["train_test_overlap_count"] == 0
        assert audit["train_val_overlap_count"] == 0
        assert audit["val_test_overlap_count"] == 0

    def test_invalid_split_ratios_raise_assertion(self, sample_molecules_df):
        """Verify ratios not summing to 1.0 raise AssertionError."""
        with pytest.raises(AssertionError):
            generate_random_split(sample_molecules_df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)

    def test_audit_label_coverage_identifies_missing_labels(self, sample_molecules_df):
        """Verify audit detects rare labels missing from partition."""
        split_map = {m: "train" for m in sample_molecules_df["molecule_id"]}
        # All in train -> val and test have 0 labels
        audit = audit_label_coverage(
            sample_molecules_df, split_map, odor_cols=["sweet", "fruity", "floral", "woody"]
        )

        assert len(audit["missing_in_val"]) == 4
        assert len(audit["missing_in_test"]) == 4
        assert len(audit["missing_in_train"]) == 0

