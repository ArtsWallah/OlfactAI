"""Integration tests for processed dataset artifacts and EDA outputs."""

import json
from pathlib import Path
import pandas as pd
import pytest

from backend.app.core.config import get_settings


class TestProcessedArtifacts:
    """Integration test suite verifying the generated Phase 2 dataset files on disk."""

    @pytest.fixture
    def processed_dir(self):
        settings = get_settings()
        return settings.DATASETS_DIR / "processed"

    def test_processed_files_exist(self, processed_dir: Path):
        """Verify presence of core processed dataset files."""
        if not (processed_dir / "molecules.csv").exists():
            pytest.skip("Processed dataset not yet generated on disk.")

        assert (processed_dir / "molecules.csv").exists()
        assert (processed_dir / "molecular_features.csv").exists()
        assert (processed_dir / "odor_labels.csv").exists()
        assert (processed_dir / "dataset_metadata.json").exists()
        assert (processed_dir / "splits_summary.json").exists()

    def test_dataset_dimensions_match_benchmark(self, processed_dir: Path):
        """Verify exact benchmark dimensions: 4,983 molecules and 138 odor labels."""
        if not (processed_dir / "molecules.csv").exists():
            pytest.skip("Processed dataset not yet generated.")

        molecules_df = pd.read_csv(processed_dir / "molecules.csv")
        features_df = pd.read_csv(processed_dir / "molecular_features.csv")
        labels_df = pd.read_csv(processed_dir / "odor_labels.csv")

        # 4,983 molecules
        assert len(molecules_df) == 4983
        assert len(features_df) == 4983
        assert len(labels_df) == 4983

        # 138 odor labels (+ id, canonical_smiles = 140 columns)
        assert labels_df.shape[1] == 140

        # Features: id + canonical_smiles + 9 descriptors + 2048 Morgan bits = 2059 columns
        assert features_df.shape[1] == 2059

        # Verify split columns added to molecules
        assert "random_split" in molecules_df.columns
        assert "scaffold_split" in molecules_df.columns

    def test_eda_visualizations_exist(self, processed_dir: Path):
        """Verify all 7 EDA visualization image files exist with non-zero size."""
        eda_dir = processed_dir / "eda"
        if not eda_dir.exists():
            pytest.skip("EDA output directory not yet generated.")

        expected_plots = [
            "label_frequency.png",
            "label_cooccurrence.png",
            "molecular_weight_distribution.png",
            "logp_distribution.png",
            "tpsa_distribution.png",
            "missing_values.png",
            "split_label_distribution.png",
        ]

        for plot_name in expected_plots:
            plot_file = eda_dir / plot_name
            assert plot_file.exists(), f"Missing EDA plot: {plot_name}"
            assert plot_file.stat().st_size > 5000, f"EDA plot {plot_name} appears empty or truncated"

    def test_scaffold_split_leakage_audit_result(self, processed_dir: Path):
        """Verify splits_summary.json confirms zero scaffold leakage."""
        summary_file = processed_dir / "splits_summary.json"
        if not summary_file.exists():
            pytest.skip("splits_summary.json not found.")

        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        scaff_audit = summary["scaffold_split"]["leakage_audit"]
        assert scaff_audit["scaffold_leakage_detected"] is False
        assert scaff_audit["train_test_overlap_count"] == 0
        assert scaff_audit["train_val_overlap_count"] == 0
        assert scaff_audit["val_test_overlap_count"] == 0

