"""Unit tests for Phase 5 UMAP Chemical Space Dimensionality Reduction."""

import pytest
import numpy as np
import pandas as pd

from backend.app.clustering.umap_embeddings import MolecularUMAP, UMAPMetadata


@pytest.fixture(scope="module")
def sample_features_df():
    """Create a controlled sample features dataframe for fast unit testing."""
    n_samples = 30
    rng = np.random.default_rng(42)

    mol_ids = [f"MOL_{i:05d}" for i in range(1, n_samples + 1)]
    smiles = ["CCO" for _ in range(n_samples)]

    data = {
        "molecule_id": mol_ids,
        "canonical_smiles": smiles,
        "molecular_weight": rng.uniform(100.0, 300.0, n_samples),
        "logp": rng.uniform(-1.0, 4.0, n_samples),
        "tpsa": rng.uniform(10.0, 120.0, n_samples),
        "h_bond_donors": rng.integers(0, 4, n_samples),
        "h_bond_acceptors": rng.integers(0, 6, n_samples),
        "rotatable_bonds": rng.integers(0, 5, n_samples),
        "ring_count": rng.integers(0, 3, n_samples),
        "heavy_atom_count": rng.integers(5, 25, n_samples),
        "fraction_csp3": rng.uniform(0.1, 0.9, n_samples),
    }

    # Add 2048 binary fingerprint columns
    fp_mat = rng.integers(0, 2, size=(n_samples, 2048), dtype=np.uint8)
    for bit in range(2048):
        data[f"fp_{bit}"] = fp_mat[:, bit]

    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def sample_labels_df():
    """Create a controlled sample labels dataframe with known edge cases."""
    return pd.DataFrame(
        {
            "molecule_id": ["M_01", "M_02", "M_03", "M_04"],
            "canonical_smiles": ["C", "CC", "CCC", "CCCC"],
            "fruity": [1, 0, 1, 0],
            "sweet": [1, 1, 0, 0],
            "woody": [0, 0, 1, 0],
            "rare_label": [0, 0, 0, 0],
        }
    )


class TestMolecularUMAP:
    """Test suite for UMAP dimensionality reduction."""

    def test_umap_2d_output_shape_and_alignment(self, sample_features_df):
        """Verify 2D UMAP produces correct shape, column names, and row alignment."""
        reducer = MolecularUMAP(n_neighbors=5, min_dist=0.1, metric="jaccard", random_state=42)
        emb_df, meta = reducer.fit_transform_fingerprints(sample_features_df, n_components=2)

        assert isinstance(meta, UMAPMetadata)
        assert meta.n_components == 2
        assert meta.metric == "jaccard"
        assert meta.random_state == 42
        assert meta.corpus_size == len(sample_features_df)
        assert meta.execution_time_seconds > 0.0

        # Check DataFrame schema and dimensions
        assert emb_df.shape == (len(sample_features_df), 4)
        assert list(emb_df.columns) == ["molecule_id", "canonical_smiles", "umap_1", "umap_2"]

        # Verify exact row-by-row alignment
        assert (emb_df["molecule_id"].values == sample_features_df["molecule_id"].values).all()
        assert (emb_df["canonical_smiles"].values == sample_features_df["canonical_smiles"].values).all()

        # Coordinates should be finite numbers
        assert not emb_df["umap_1"].isna().any()
        assert not emb_df["umap_2"].isna().any()

    def test_umap_10d_output_shape(self, sample_features_df):
        """Verify 10D intermediate UMAP produces shape (N, 12)."""
        reducer = MolecularUMAP(n_neighbors=5, min_dist=0.1, metric="jaccard", random_state=42)
        emb_df, meta = reducer.fit_transform_fingerprints(sample_features_df, n_components=10)

        assert meta.n_components == 10
        assert emb_df.shape == (len(sample_features_df), 12)
        coord_cols = [f"umap_{i+1}" for i in range(10)]
        expected_cols = ["molecule_id", "canonical_smiles"] + coord_cols
        assert list(emb_df.columns) == expected_cols

    def test_reproducibility(self, sample_features_df):
        """Verify identical random seed produces bit-for-bit identical coordinates."""
        r1 = MolecularUMAP(n_neighbors=5, min_dist=0.1, metric="jaccard", random_state=42)
        emb1, _ = r1.fit_transform_fingerprints(sample_features_df, n_components=2)

        r2 = MolecularUMAP(n_neighbors=5, min_dist=0.1, metric="jaccard", random_state=42)
        emb2, _ = r2.fit_transform_fingerprints(sample_features_df, n_components=2)

        np.testing.assert_allclose(emb1["umap_1"].values, emb2["umap_1"].values, atol=1e-5)
        np.testing.assert_allclose(emb1["umap_2"].values, emb2["umap_2"].values, atol=1e-5)

    def test_physicochemical_umap(self, sample_features_df):
        """Verify secondary physicochemical UMAP standardizes features and projects to 2D."""
        reducer = MolecularUMAP(n_neighbors=5, min_dist=0.1, random_state=42)
        emb_df, meta = reducer.fit_transform_physicochemical(sample_features_df, n_components=2)

        assert meta.embedding_type == "physicochemical_descriptors"
        assert meta.scaling_applied is True
        assert meta.metric == "euclidean"
        assert emb_df.shape == (len(sample_features_df), 4)
        assert list(emb_df.columns) == ["molecule_id", "canonical_smiles", "umap_1", "umap_2"]

    def test_dominant_label_assignment_and_tie_breaking(self, sample_labels_df):
        """Verify dominant odor label assignment rules:
        - M_04 has 0 active labels -> 'unlabeled'
        - M_02 has only 'sweet' -> 'sweet'
        - M_01 has both 'fruity' and 'sweet'.
          In sample_labels_df: 'fruity' has sum=2, 'sweet' has sum=2 (tied prevalence).
          Alphabetical tie-break: 'fruity' < 'sweet' -> 'fruity'.
        - M_03 has 'fruity' (sum=2) and 'woody' (sum=1).
          Higher prevalence wins: 'fruity' > 'woody' -> 'fruity'.
        """
        dominant = MolecularUMAP.assign_dominant_odor_labels(sample_labels_df)

        assert dominant.iloc[3] == "unlabeled"
        assert dominant.iloc[1] == "sweet"
        assert dominant.iloc[0] == "fruity"
        assert dominant.iloc[2] == "fruity"

    def test_validation_errors(self, sample_features_df):
        """Verify validation and boundary error conditions."""
        reducer = MolecularUMAP()

        # Insufficient samples (< 5)
        small_df = sample_features_df.head(4)
        with pytest.raises(ValueError, match="requires at least 5 molecules"):
            reducer.fit_transform_fingerprints(small_df)

        # Missing required identity columns
        no_id_df = sample_features_df.drop(columns=["molecule_id"])
        with pytest.raises(ValueError, match="must contain 'molecule_id'"):
            reducer.fit_transform_fingerprints(no_id_df)

        # Missing fingerprint columns
        no_fp_df = sample_features_df[["molecule_id", "canonical_smiles", "molecular_weight"]]
        with pytest.raises(ValueError, match="No Morgan fingerprint columns"):
            reducer.fit_transform_fingerprints(no_fp_df)

        # Missing physicochemical properties
        with pytest.raises(ValueError, match="Missing required physicochemical"):
            reducer.fit_transform_physicochemical(sample_features_df.drop(columns=["tpsa"]))

