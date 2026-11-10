"""Unit tests for HDBSCAN molecular clustering component."""

import numpy as np
import pandas as pd
import pytest

from backend.app.clustering.hdbscan_clustering import (
    MolecularHDBSCAN,
    compare_representations,
    compute_jaccard_distance_matrix,
    estimate_distance_matrix_memory_mb,
    run_parameter_sensitivity_grid,
    validate_distance_matrix,
)


class TestDistanceMatrixComputation:
    """Test pairwise Jaccard distance calculation and validation."""

    def test_estimate_distance_matrix_memory_mb(self):
        """Verify memory estimation formula for dense matrices."""
        # 1000 x 1000 float64 = 8,000,000 bytes = 7.62939 MB
        mem = estimate_distance_matrix_memory_mb(1000, dtype=np.float64)
        expected = (1000 * 1000 * 8) / (1024 * 1024)
        assert abs(mem - expected) < 1e-5

        # 4983 x 4983 float64 = ~189.44 MB
        mem_4983 = estimate_distance_matrix_memory_mb(4983, dtype=np.float64)
        assert 185.0 < mem_4983 < 195.0

    def test_compute_jaccard_distance_matrix_properties(self):
        """Verify square, symmetric, zero-diagonal, bounded in [0, 1] Jaccard matrix."""
        # 4 toy binary fingerprints of length 8
        fps = np.array([
            [1, 1, 1, 0, 0, 0, 0, 0],  # A
            [1, 1, 0, 0, 0, 0, 0, 0],  # B (inter=2, union=3, Tanimoto=2/3, dist=1/3)
            [0, 0, 0, 1, 1, 1, 0, 0],  # C (disjoint from A, B: Tanimoto=0, dist=1.0)
            [1, 1, 1, 0, 0, 0, 0, 0],  # D (identical to A: Tanimoto=1.0, dist=0.0)
        ], dtype=np.float32)

        dists = compute_jaccard_distance_matrix(fps)

        assert dists.shape == (4, 4)
        assert dists.dtype == np.float64
        # Zero diagonal
        np.testing.assert_allclose(np.diag(dists), 0.0, atol=1e-7)
        # Symmetry
        np.testing.assert_allclose(dists, dists.T, atol=1e-7)
        # Bounds
        assert (dists >= 0.0).all() and (dists <= 1.0).all()
        # Analytical distance checks
        np.testing.assert_allclose(dists[0, 1], 1.0 - (2.0 / 3.0), atol=1e-5)
        np.testing.assert_allclose(dists[0, 2], 1.0, atol=1e-5)
        np.testing.assert_allclose(dists[0, 3], 0.0, atol=1e-5)

    def test_compute_jaccard_distance_with_zero_on_bits(self):
        """Verify handling of molecules with zero on-bits (e.g. single-atom inorganic like [Zn])."""
        fps = np.array([
            [1, 1, 0, 0],
            [0, 0, 0, 0],  # 0 on-bits
            [0, 1, 1, 0],
        ], dtype=np.float32)

        dists = compute_jaccard_distance_matrix(fps)
        assert dists.shape == (3, 3)
        # Zero diagonal preserved even for 0 on-bits
        assert dists[1, 1] == 0.0
        # Distance to non-identical molecules is 1.0
        assert dists[1, 0] == 1.0
        assert dists[1, 2] == 1.0

    def test_validate_distance_matrix_valid(self):
        """Verify validator accepts compliant distance matrix."""
        mat = np.array([
            [0.0, 0.4, 0.7],
            [0.4, 0.0, 0.2],
            [0.7, 0.2, 0.0],
        ], dtype=np.float64)
        # Should not raise
        validate_distance_matrix(mat, check_bounds=True)

    def test_validate_distance_matrix_errors(self):
        """Verify validator catches non-square, asymmetric, non-finite, non-zero diagonal matrices."""
        # Non-2D
        with pytest.raises(ValueError, match="must be square 2D"):
            validate_distance_matrix(np.zeros((3, 3, 3)))

        # Non-square
        with pytest.raises(ValueError, match="must be square 2D"):
            validate_distance_matrix(np.zeros((3, 4)))

        # Non-finite (NaN)
        with pytest.raises(ValueError, match="non-finite"):
            bad_nan = np.zeros((3, 3))
            bad_nan[0, 1] = np.nan
            validate_distance_matrix(bad_nan)

        # Non-zero diagonal
        with pytest.raises(ValueError, match="diagonal must be identically zero"):
            bad_diag = np.zeros((3, 3))
            bad_diag[1, 1] = 0.05
            validate_distance_matrix(bad_diag)

        # Asymmetric
        with pytest.raises(ValueError, match="asymmetric"):
            bad_asym = np.zeros((3, 3))
            bad_asym[0, 1] = 0.2
            bad_asym[1, 0] = 0.5
            validate_distance_matrix(bad_asym)

        # Out of bounds (< 0)
        with pytest.raises(ValueError, match="negative values"):
            bad_neg = np.zeros((3, 3))
            bad_neg[0, 1] = -0.1
            bad_neg[1, 0] = -0.1
            validate_distance_matrix(bad_neg, check_bounds=True)

        # Out of bounds (> 1.0)
        with pytest.raises(ValueError, match="exceeds upper bound"):
            bad_high = np.zeros((3, 3))
            bad_high[0, 1] = 1.5
            bad_high[1, 0] = 1.5
            validate_distance_matrix(bad_high, check_bounds=True)


class TestMolecularHDBSCAN:
    """Test HDBSCAN clustering engine and statistical summarization."""

    @pytest.fixture
    def synthetic_clustered_distance_matrix(self):
        """Create a synthetic 40x40 distance matrix with 2 distinct dense clusters and noise."""
        np.random.seed(42)
        # Cluster 1: indices 0..14 (tight: pairwise dist ~ 0.1)
        # Cluster 2: indices 15..29 (tight: pairwise dist ~ 0.1)
        # Noise: indices 30..39 (isolated: pairwise dist ~ 0.8)
        pts1 = np.random.normal(loc=[0, 0], scale=0.05, size=(15, 2))
        pts2 = np.random.normal(loc=[5, 5], scale=0.05, size=(15, 2))
        noise = np.random.uniform(low=-10, high=15, size=(10, 2))

        all_pts = np.vstack([pts1, pts2, noise])
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(all_pts, metric="euclidean"))
        # Normalize to [0, 1]
        dists = dists / np.max(dists)
        np.fill_diagonal(dists, 0.0)
        return dists.astype(np.float64)

    def test_hdbscan_fit_precomputed(self, synthetic_clustered_distance_matrix):
        """Verify fitting on precomputed distance matrix produces valid clusters and noise."""
        clusterer = MolecularHDBSCAN(min_cluster_size=5, min_samples=3, metric="precomputed")
        labels = clusterer.fit_predict(synthetic_clustered_distance_matrix)

        assert len(labels) == 40
        assert clusterer.is_fitted_
        assert clusterer.probabilities_ is not None
        assert clusterer.outlier_scores_ is not None

        # Verify cluster labels are integers >= -1
        assert all(isinstance(l, (int, np.integer)) for l in labels)
        assert all(l >= -1 for l in labels)

        # Verify membership strengths in [0, 1]
        assert (clusterer.probabilities_ >= 0.0).all()
        assert (clusterer.probabilities_ <= 1.0).all()

        # Should discover at least 2 clusters
        non_noise = [l for l in labels if l != -1]
        assert len(set(non_noise)) >= 2

    def test_hdbscan_summary_statistics(self, synthetic_clustered_distance_matrix):
        """Verify summary dataclass extracts all required statistical and persistence metrics."""
        clusterer = MolecularHDBSCAN(min_cluster_size=5, min_samples=3, metric="precomputed")
        clusterer.fit(synthetic_clustered_distance_matrix)
        summary = clusterer.get_cluster_summary(distance_matrix=synthetic_clustered_distance_matrix)

        assert summary.n_samples == 40
        assert summary.n_clusters >= 2
        assert 0.0 <= summary.noise_fraction <= 1.0
        assert summary.min_cluster_size_observed >= 5
        assert summary.max_cluster_size >= summary.median_cluster_size
        assert 0.0 <= summary.mean_membership_strength <= 1.0
        assert summary.runtime_seconds > 0.0
        assert summary.silhouette_score_clustered is not None
        assert "non-noise" in summary.silhouette_caveat

    def test_assign_clusters_to_dataframe_alignment(self, synthetic_clustered_distance_matrix):
        """Verify dataframe assignment preserves exact row alignment and columns."""
        df = pd.DataFrame({
            "molecule_id": [f"MOL_{i:04d}" for i in range(40)],
            "canonical_smiles": [f"C{i}" for i in range(40)],
        })
        clusterer = MolecularHDBSCAN(min_cluster_size=5, min_samples=3, metric="precomputed")
        clusterer.fit(synthetic_clustered_distance_matrix)

        assigned_df = clusterer.assign_clusters_to_dataframe(df)

        assert len(assigned_df) == 40
        assert "cluster_hdbscan" in assigned_df.columns
        assert "membership_strength" in assigned_df.columns
        assert "outlier_score" in assigned_df.columns

        # Verify row alignment
        assert list(assigned_df["molecule_id"]) == list(df["molecule_id"])
        np.testing.assert_array_equal(assigned_df["cluster_hdbscan"].values, clusterer.labels_)

    def test_assign_clusters_mismatched_length_raises(self, synthetic_clustered_distance_matrix):
        """Verify error is raised if dataframe row count differs from model sample count."""
        df_short = pd.DataFrame({"molecule_id": ["M1", "M2"]})
        clusterer = MolecularHDBSCAN(min_cluster_size=5, metric="precomputed")
        clusterer.fit(synthetic_clustered_distance_matrix)

        with pytest.raises(ValueError, match="row count .* does not match"):
            clusterer.assign_clusters_to_dataframe(df_short)

    def test_parameter_sensitivity_grid(self, synthetic_clustered_distance_matrix):
        """Verify parameter sensitivity grid executes and returns structured results."""
        mcs_list = [5, 10]
        ms_list = [None, 3]
        results_df, meta_dict = run_parameter_sensitivity_grid(
            distance_matrix=synthetic_clustered_distance_matrix,
            min_cluster_size_candidates=mcs_list,
            min_samples_candidates=ms_list,
        )

        assert len(results_df) == 4
        expected_cols = [
            "config_id", "min_cluster_size", "min_samples",
            "n_clusters", "n_noise", "noise_fraction", "runtime_seconds"
        ]
        for col in expected_cols:
            assert col in results_df.columns
        assert len(meta_dict) == 4

    def test_representation_comparison(self, synthetic_clustered_distance_matrix):
        """Verify representation comparison evaluates all three representations."""
        np.random.seed(42)
        X_10d = np.random.normal(size=(40, 10))
        X_2d = np.random.normal(size=(40, 2))

        comp_df, summaries = compare_representations(
            fingerprint_distance_matrix=synthetic_clustered_distance_matrix,
            umap_10d_matrix=X_10d,
            umap_2d_matrix=X_2d,
            min_cluster_size=5,
            min_samples=3,
        )

        assert len(comp_df) == 3
        assert "Primary: Morgan FP (Jaccard)" in comp_df["representation"].values
        assert "Exploratory: 10D UMAP (Euclidean)" in comp_df["representation"].values
        assert "Visualization: 2D UMAP (Euclidean)" in comp_df["representation"].values
        assert len(summaries) == 3

    def test_no_mutation_of_input_distance_matrix(self, synthetic_clustered_distance_matrix):
        """Verify that fitting HDBSCAN does not mutate the caller's distance matrix."""
        mat_copy = synthetic_clustered_distance_matrix.copy()
        clusterer = MolecularHDBSCAN(min_cluster_size=5, min_samples=3, metric="precomputed")
        clusterer.fit(synthetic_clustered_distance_matrix)

        np.testing.assert_array_equal(synthetic_clustered_distance_matrix, mat_copy)

