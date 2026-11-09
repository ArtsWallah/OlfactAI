"""Unit tests for the OlfactAI Research Playground backend integration."""

import pytest
import numpy as np

from backend.app.core.exceptions import InvalidSMILESError
from playground.services import PlaygroundService


@pytest.fixture(scope="module")
def playground_service():
    """Initialize PlaygroundService singleton for testing."""
    return PlaygroundService()


class TestPlaygroundService:
    """Test suite for research playground backend integration."""

    def test_parse_molecule_valid_vanillin(self, playground_service):
        """Verify molecular parsing, formula, image, and descriptor extraction for Vanillin."""
        vanillin_smiles = "O=Cc1ccc(O)c(OC)c1"
        data = playground_service.parse_molecule(vanillin_smiles)

        assert data["canonical_smiles"] == "COc1cc(C=O)ccc1O"
        assert data["molecular_formula"] == "C8H8O3"
        assert abs(data["descriptors"].molecular_weight - 152.15) < 0.1
        assert data["descriptors"].ring_count == 1
        assert data["descriptors"].h_bond_donors == 1
        assert data["descriptors"].h_bond_acceptors == 3
        assert data["image"] is not None
        # Vanillin is present in OpenPOM dataset
        assert data["dataset_molecule_id"] is not None
        assert data["dataset_molecule_id"].startswith("MOL_")

    def test_parse_molecule_invalid_smiles(self, playground_service):
        """Verify invalid SMILES raises InvalidSMILESError."""
        with pytest.raises(InvalidSMILESError):
            playground_service.parse_molecule("InvalidSmiles123!!!")

    def test_predict_odor_profiles_vanillin(self, playground_service):
        """Verify multi-model predictions on Vanillin across GNN, XGBoost, and MLP."""
        vanillin_smiles = "O=Cc1ccc(O)c(OC)c1"
        res = playground_service.predict_odor_profiles(vanillin_smiles, split="random", top_n=5)

        assert res["split"] == "random"
        assert len(res["labels"]) == 138

        models = res["model_results"]
        assert "GNN (MPNN Phase 4)" in models
        assert "XGBoost (Phase 3)" in models
        assert "PyTorch MLP (Phase 3)" in models

        # Check GNN output
        gnn_res = models["GNN (MPNN Phase 4)"]
        assert len(gnn_res["calibrated_probabilities"]) == 138
        assert (gnn_res["calibrated_probabilities"] >= 0.0).all()
        assert (gnn_res["calibrated_probabilities"] <= 1.0).all()
        assert gnn_res["predictive_entropy"] >= 0.0
        assert "vanilla" in gnn_res["active_labels"]

        # Check XGBoost output
        xgb_res = models["XGBoost (Phase 3)"]
        assert len(xgb_res["raw_probabilities"]) == 138
        assert "vanilla" in xgb_res["active_labels"]

        # Check PyTorch MLP output
        mlp_res = models["PyTorch MLP (Phase 3)"]
        assert len(mlp_res["raw_probabilities"]) == 138
        assert "vanilla" in mlp_res["active_labels"]

    def test_predict_odor_profiles_scaffold_split(self, playground_service):
        """Verify scaffold split model inference executes cleanly."""
        vanillin_smiles = "O=Cc1ccc(O)c(OC)c1"
        res = playground_service.predict_odor_profiles(vanillin_smiles, split="scaffold", top_n=5)

        models = res["model_results"]
        assert "GNN (MPNN Phase 4)" in models
        gnn_res = models["GNN (MPNN Phase 4)"]
        assert len(gnn_res["top_descriptors"]) == 5
        # Top descriptor in scaffold split GNN must also be vanilla
        top_descriptor_name = gnn_res["top_descriptors"][0][0]
        assert top_descriptor_name == "vanilla"

    def test_similarity_search(self, playground_service):
        """Verify similarity search returns matches and excludes exact query."""
        vanillin_smiles = "O=Cc1ccc(O)c(OC)c1"
        res = playground_service.search_similar(query=vanillin_smiles, top_k=5)

        assert len(res["matches"]) == 5
        # Self-match exclusion check
        match_smiles = [m["canonical_smiles"] for m in res["matches"]]
        assert "COc1cc(C=O)ccc1O" not in match_smiles
        assert res["matches"][0]["tanimoto_similarity"] <= 1.0

    def test_neighborhood_analysis(self, playground_service):
        """Verify structural neighborhood analysis returns valid metrics."""
        vanillin_smiles = "O=Cc1ccc(O)c(OC)c1"
        res = playground_service.analyze_neighborhood(query=vanillin_smiles, top_k=10)

        assert res["k"] == 10
        assert 0.0 <= res["mean_structural_similarity"] <= 1.0
        assert 0.0 <= res["mean_odor_jaccard"] <= 1.0
        assert len(res["top_neighbor_labels"]) > 0

    def test_chemical_space_data_loading(self, playground_service):
        """Verify 2D UMAP chemical space loads 4,983 molecules with required properties."""
        df = playground_service.load_chemical_space_data()

        assert len(df) == 4983
        expected_cols = ["molecule_id", "canonical_smiles", "umap_1", "umap_2", "dominant_odor", "molecular_weight", "logp", "tpsa"]
        for col in expected_cols:
            assert col in df.columns
        assert not df["umap_1"].isna().any()
        assert not df["umap_2"].isna().any()

