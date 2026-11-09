"""Unit tests for Phase 5 Molecular Neighborhood Analysis."""

import pytest
import numpy as np

from backend.app.similarity.engine import MolecularSimilarityEngine
from backend.app.similarity.neighborhood import (
    MolecularNeighborhoodAnalyzer,
    calculate_jaccard_similarity,
    calculate_label_overlap,
)


class TestJaccardAndOverlapFunctions:
    """Hand-verified unit tests for Jaccard and overlap helper functions."""

    def test_identical_label_sets(self):
        """Verify identical non-empty sets yield Jaccard of 1.0."""
        s1 = ["apple", "floral", "fruity"]
        s2 = ["fruity", "apple", "floral"]
        assert calculate_jaccard_similarity(s1, s2) == 1.0
        cnt, shared = calculate_label_overlap(s1, s2)
        assert cnt == 3
        assert shared == ["apple", "floral", "fruity"]

    def test_partially_overlapping_sets(self):
        """Verify partially overlapping sets compute exact fraction |A ∩ B| / |A ∪ B|."""
        # A = {apple, floral}, B = {floral, woody}
        # Inter = {floral} (1), Union = {apple, floral, woody} (3) -> 1/3
        s1 = ["apple", "floral"]
        s2 = ["floral", "woody"]
        jacc = calculate_jaccard_similarity(s1, s2)
        assert abs(jacc - (1.0 / 3.0)) < 1e-6
        cnt, shared = calculate_label_overlap(s1, s2)
        assert cnt == 1
        assert shared == ["floral"]

    def test_disjoint_sets(self):
        """Verify non-overlapping sets yield Jaccard of 0.0."""
        s1 = ["alliaceous", "garlic"]
        s2 = ["vanilla", "sweet"]
        assert calculate_jaccard_similarity(s1, s2) == 0.0
        cnt, shared = calculate_label_overlap(s1, s2)
        assert cnt == 0
        assert shared == []

    def test_empty_sets(self):
        """Verify empty set edge cases are handled consistently."""
        # Empty vs empty
        assert calculate_jaccard_similarity([], []) == 0.0
        assert calculate_jaccard_similarity([], [], empty_empty_val=1.0) == 1.0

        # Empty vs non-empty
        assert calculate_jaccard_similarity([], ["fruity"]) == 0.0
        assert calculate_jaccard_similarity(["fruity"], []) == 0.0

        cnt, shared = calculate_label_overlap([], ["fruity"])
        assert cnt == 0
        assert shared == []


@pytest.fixture(scope="module")
def neighborhood_analyzer():
    """Load analyzer once for test efficiency."""
    engine = MolecularSimilarityEngine(auto_load=True)
    return MolecularNeighborhoodAnalyzer(engine=engine)


class TestMolecularNeighborhoodAnalyzer:
    """Test suite for per-molecule and global neighborhood analysis."""

    def test_per_molecule_neighborhood_k_variants(self, neighborhood_analyzer):
        """Verify neighborhood retrieval for K = 5, 10, 20, 50."""
        for k in [5, 10, 20, 50]:
            res = neighborhood_analyzer.analyze_neighborhood(query="MOL_00001", k=k)
            assert res.k == k
            assert len(res.neighbors) == k
            assert 0.0 <= res.mean_structural_similarity <= 1.0
            assert 0.0 <= res.median_structural_similarity <= 1.0
            assert 0.0 <= res.mean_odor_jaccard <= 1.0
            assert 0.0 <= res.median_odor_jaccard <= 1.0
            assert 0.0 <= res.zero_odor_overlap_proportion <= 1.0

            # Verify descending rank order
            ranks = [n.rank for n in res.neighbors]
            assert ranks == list(range(1, k + 1))

    def test_empty_label_molecule_handling(self, neighborhood_analyzer):
        """Verify querying a novel molecule with no odor annotations executes without error."""
        # External novel query (Cubane, not present in the fragrance dataset)
        cubane_smiles = "C12C3C4C1C5C2C3C45"
        res = neighborhood_analyzer.analyze_neighborhood(query=cubane_smiles, k=5)
        assert len(res.neighbors) == 5
        assert res.query_odor_labels == []
        # When query has 0 labels, odor Jaccard should be 0.0
        assert res.mean_odor_jaccard == 0.0
        assert res.zero_odor_overlap_proportion == 1.0

    def test_invalid_k_raises_error(self, neighborhood_analyzer):
        """Verify invalid K values raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            neighborhood_analyzer.analyze_neighborhood(query="MOL_00001", k=0)

        with pytest.raises(ValueError, match="must be positive"):
            neighborhood_analyzer.analyze_neighborhood(query="MOL_00001", k=-10)

    def test_multi_k_batch_analysis(self, neighborhood_analyzer):
        """Verify multi-K neighborhood analysis dictionary output."""
        multi_res = neighborhood_analyzer.analyze_multi_k(query="MOL_00010", k_values=[5, 10, 20])
        assert set(multi_res.keys()) == {5, 10, 20}
        assert multi_res[5].k == 5
        assert multi_res[10].k == 10
        assert multi_res[20].k == 20

    def test_cluster_bootstrap_reproducibility(self, neighborhood_analyzer):
        """Verify molecule-level cluster bootstrap is reproducible given a fixed seed."""
        boot1 = neighborhood_analyzer.compute_molecule_cluster_bootstrap(
            n_iterations=5, subsample_size=200, seed=123
        )
        boot2 = neighborhood_analyzer.compute_molecule_cluster_bootstrap(
            n_iterations=5, subsample_size=200, seed=123
        )

        assert boot1.pearson_r_point_estimate == boot2.pearson_r_point_estimate
        assert boot1.pearson_r_ci_95 == boot2.pearson_r_ci_95
        assert boot1.spearman_rho_ci_95 == boot2.spearman_rho_ci_95

        # Check CI structure: low <= high
        assert boot1.pearson_r_ci_95[0] <= boot1.pearson_r_ci_95[1]
        assert boot1.spearman_rho_ci_95[0] <= boot1.spearman_rho_ci_95[1]
