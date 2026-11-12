"""Unit tests for molecular outlier analysis framework."""

import numpy as np
import pandas as pd
import pytest

from backend.app.clustering.molecular_outliers import (
    MolecularOutlierEvaluator,
    compute_annotation_outlier_metrics,
    compute_pairwise_tanimoto_matrix,
    compute_prediction_outlier_metrics,
    compute_structural_novelty,
)


class TestPairwiseTanimotoAndNovelty:
    """Test self-match exclusion, exact distances, and top-k rankings."""

    def test_pairwise_tanimoto_matrix_exact(self):
        # 3 simple bit vectors of length 4
        # v0: [1, 1, 0, 0]
        # v1: [1, 1, 0, 0]  (identical to v0)
        # v2: [0, 0, 1, 1]  (disjoint from v0)
        fps = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.float32,
        )
        sim = compute_pairwise_tanimoto_matrix(fps)

        assert sim.shape == (3, 3)
        assert np.isclose(sim[0, 0], 1.0)
        assert np.isclose(sim[0, 1], 1.0)
        assert np.isclose(sim[0, 2], 0.0)

    def test_self_match_exclusion(self):
        # 3 molecules: A and B are identical (sim=1.0), C is different
        fps = np.array(
            [
                [1, 1, 1, 0],  # A
                [1, 1, 1, 0],  # B (identical to A)
                [1, 0, 0, 0],  # C (sim to A: 1/3)
            ],
            dtype=np.float32,
        )
        sim = compute_pairwise_tanimoto_matrix(fps)
        ids = ["MOL_A", "MOL_B", "MOL_C"]
        df_novelty = compute_structural_novelty(sim, molecule_ids=ids)

        # For MOL_A:
        # Non-self nearest neighbor should be MOL_B with sim = 1.0, distance = 0.0
        row_a = df_novelty[df_novelty["molecule_id"] == "MOL_A"].iloc[0]
        assert row_a["nearest_neighbor_id"] == "MOL_B"
        assert np.isclose(row_a["max_tanimoto_similarity"], 1.0)
        assert np.isclose(row_a["nearest_neighbor_distance"], 0.0)

        # For MOL_C:
        # Non-self nearest neighbor is MOL_A (or MOL_B) with sim = 1/3
        row_c = df_novelty[df_novelty["molecule_id"] == "MOL_C"].iloc[0]
        assert np.isclose(row_c["max_tanimoto_similarity"], 1.0 / 3.0)
        assert np.isclose(row_c["nearest_neighbor_distance"], 2.0 / 3.0)

    def test_topk_similarity_handling_small_corpus(self):
        # Corpus smaller than 20: ensure top-5, top-10, top-20 means compute without crash
        fps = np.eye(6, dtype=np.float32)  # 6 mutually orthogonal molecules
        sim = compute_pairwise_tanimoto_matrix(fps)
        df = compute_structural_novelty(sim)

        assert len(df) == 6
        assert np.all(np.isclose(df["max_tanimoto_similarity"], 0.0))
        assert np.all(np.isclose(df["nearest_neighbor_distance"], 1.0))
        assert np.all(np.isclose(df["local_sparsity_k10"], 1.0))

    def test_dimension_mismatch_raises(self):
        non_square = np.zeros((4, 5))
        with pytest.raises(ValueError, match="Square matrix expected"):
            compute_structural_novelty(non_square)


class TestAnnotationOutlierMetrics:
    """Test annotation count and descriptor rarity calculations."""

    def test_annotation_rarity_and_unannotated(self):
        # 3 molecules, 2 descriptors
        # M0: possesses rare descriptor (d0=1, d1=0) - d0 appears in 1/3
        # M1: possesses common descriptor (d0=0, d1=1) - d1 appears in 2/3
        # M2: unannotated (d0=0, d1=0)
        odor_df = pd.DataFrame(
            {
                "molecule_id": ["M0", "M1", "M2"],
                "desc_rare": [1, 0, 0],
                "desc_common": [0, 1, 0],
            }
        )
        res = compute_annotation_outlier_metrics(odor_df)

        assert len(res) == 3
        # M0
        m0 = res[res["molecule_id"] == "M0"].iloc[0]
        assert m0["annotation_count"] == 1
        assert np.isclose(m0["mean_descriptor_frequency"], 1.0 / 3.0)
        assert np.isclose(m0["annotation_rarity_score"], 2.0 / 3.0)
        assert m0["is_unannotated"] == 0

        # M2
        m2 = res[res["molecule_id"] == "M2"].iloc[0]
        assert m2["annotation_count"] == 0
        assert m2["annotation_rarity_score"] == 0.0
        assert m2["is_unannotated"] == 1


class TestPredictionOutlierMetrics:
    """Test entropy and loss calculation on predictions."""

    def test_prediction_metrics_extremes(self):
        # Molecule 0: perfectly confident & correct (prob=1, y=1) -> low entropy, zero loss
        # Molecule 1: highly uncertain (prob=0.5, y=1) -> high entropy (1.0 bit), high loss
        y_true = np.array([[1, 0], [1, 0]])
        y_probs = np.array([[0.9999, 0.0001], [0.5, 0.5]])

        res = compute_prediction_outlier_metrics(
            y_true, y_probs, molecule_ids=["M_certain", "M_uncertain"]
        )

        assert len(res) == 2
        r_certain = res[res["molecule_id"] == "M_certain"].iloc[0]
        r_uncertain = res[res["molecule_id"] == "M_uncertain"].iloc[0]

        assert r_certain["predictive_entropy"] < r_uncertain["predictive_entropy"]
        assert r_certain["bce_loss"] < r_uncertain["bce_loss"]
        assert np.isclose(r_uncertain["predictive_entropy"], 1.0, atol=1e-2)


class TestMolecularOutlierEvaluator:
    """Test cross-dimensional consolidation and rank correlation."""

    def test_consolidate_and_correlate(self):
        n = 20
        mol_ids = [f"M_{i:02d}" for i in range(n)]

        # Mock novelty
        novelty_df = pd.DataFrame(
            {
                "molecule_id": mol_ids,
                "nearest_neighbor_distance": np.linspace(0.1, 0.9, n),
                "local_sparsity_k10": np.linspace(0.2, 0.95, n),
            }
        )

        # Mock HDBSCAN
        hdbscan_df = pd.DataFrame(
            {
                "molecule_id": mol_ids,
                "cluster_hdbscan": [0] * 15 + [-1] * 5,
                "membership_strength": np.random.uniform(0.5, 1.0, n),
                "outlier_score": np.linspace(0.0, 1.0, n),
            }
        )

        # Mock Annotation
        annot_df = pd.DataFrame(
            {
                "molecule_id": mol_ids,
                "annotation_count": np.random.randint(1, 10, n),
                "annotation_rarity_score": np.linspace(0.1, 0.8, n),
            }
        )

        # Mock Prediction
        pred_df = pd.DataFrame(
            {
                "molecule_id": mol_ids,
                "predictive_entropy": np.linspace(0.2, 0.6, n),
                "bce_loss": np.linspace(0.05, 0.4, n),
                "hamming_error": np.linspace(0.01, 0.1, n),
            }
        )

        evaluator = MolecularOutlierEvaluator(top_percentile=90.0)
        profile = evaluator.consolidate_outlier_profile(
            novelty_df, hdbscan_df, annot_df, pred_df
        )

        assert len(profile) == n
        assert "rank_structural_novelty" in profile.columns
        assert "is_structural_outlier" in profile.columns
        assert "is_glosh_outlier" in profile.columns
        assert "is_uncertainty_outlier" in profile.columns

        # Verify cross-dimension correlations
        corrs = evaluator.compute_cross_dimension_correlations(profile)
        assert corrs.shape[0] == corrs.shape[1]
        assert np.isclose(corrs.loc["nearest_neighbor_distance", "nearest_neighbor_distance"], 1.0)
