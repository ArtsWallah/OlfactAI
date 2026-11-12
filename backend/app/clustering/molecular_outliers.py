"""Molecular Outlier Analysis for OlfactAI Chemical Space.

Systematically identifies and evaluates five distinct classes of molecular outliers
within the OpenPOM dataset (4,983 molecules, 138 binary odor descriptors):

1. Structural Novelty Outliers:
   - Evaluated in native 2,048-bit Morgan fingerprint space via Tanimoto similarity.
   - Metrics: max non-self Tanimoto similarity, mean similarity to top 5, 10, 20 neighbors,
     and distance to nearest neighbor (1 - max_similarity).
2. Local Density Outliers:
   - Evaluated via average distance to k nearest neighbors (k=5, 10, 20).
   - Higher average distance denotes molecules residing in sparse manifolds.
3. Clustering Outliers (HDBSCAN GLOSH):
   - Global-Local Outlier Score for Hierarchies (GLOSH) from fitted HDBSCAN.
   - Evaluated alongside cluster assignments (core clusters vs. noise).
4. Prediction Uncertainty & Error Outliers:
   - Mean predictive entropy across 138 descriptors via trained models (e.g. XGBoost).
   - Prediction errors (Hamming distance, binary cross-entropy loss) against true labels.
5. Annotation Rarity Outliers:
   - Total number of active descriptors (sparsity vs. poly-olfaction).
   - Descriptor rarity score: mean background rarity of a molecule's annotated descriptors.

Crucial Scientific Caveats:
- These five dimensions represent fundamentally distinct properties and are NEVER
  collapsed into an arbitrary composite score.
- Low similarity or noise status does NOT denote synthetic artifact or natural origin.
- High predictive entropy reflects model hesitation under sparse training data, NOT
  perceptual sensory ambiguity for human observers.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StructuralNoveltyResult:
    """Nearest-neighbor structural novelty metrics for a single molecule."""

    molecule_id: str
    max_tanimoto_similarity: float
    nearest_neighbor_distance: float
    mean_top5_similarity: float
    mean_top10_similarity: float
    mean_top20_similarity: float
    nearest_neighbor_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_pairwise_tanimoto_matrix(fps: np.ndarray) -> np.ndarray:
    """Compute all-pairs Tanimoto similarity matrix from binary fingerprint array.

    Parameters
    ----------
    fps : np.ndarray
        Shape (N, n_bits), binary or float 0/1 array.

    Returns
    -------
    np.ndarray
        Shape (N, N), pairwise Tanimoto coefficients in [0, 1].
    """
    fps_float = np.asarray(fps, dtype=np.float32)
    dot = np.dot(fps_float, fps_float.T)
    norms = fps_float.sum(axis=1)
    denom = norms[:, None] + norms[None, :] - dot
    sim_matrix = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
    return sim_matrix


def compute_structural_novelty(
    sim_matrix: np.ndarray,
    molecule_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute nearest-neighbor structural novelty metrics, strictly excluding self-matches.

    Parameters
    ----------
    sim_matrix : np.ndarray
        Shape (N, N) pairwise similarity matrix.
    molecule_ids : Optional[List[str]], default=None
        Optional list of molecule identifiers.

    Returns
    -------
    pd.DataFrame
        DataFrame containing novelty metrics for all N molecules.
    """
    n = sim_matrix.shape[0]
    if sim_matrix.shape[1] != n:
        raise ValueError(f"Square matrix expected, got shape {sim_matrix.shape}")

    if molecule_ids is None:
        mol_ids = [f"MOL_{i:05d}" for i in range(n)]
    else:
        mol_ids = list(molecule_ids)
        if len(mol_ids) != n:
            raise ValueError(f"Length mismatch: {len(mol_ids)} IDs vs {n} rows")

    # Clone matrix and mask self-similarity on the diagonal
    sim_masked = sim_matrix.copy()
    np.fill_diagonal(sim_masked, -np.inf)

    # Sort descending along rows
    sorted_sim = np.sort(sim_masked, axis=1)[:, ::-1]
    sorted_idx = np.argsort(sim_masked, axis=1)[:, ::-1]

    # Max non-self similarity (first column after sorting)
    max_sim = sorted_sim[:, 0]
    nn_dist = 1.0 - max_sim

    # Nearest neighbor ID
    nn_ids = [mol_ids[idx[0]] for idx in sorted_idx]

    # Top-K means (handling cases where n <= k)
    k5 = min(5, n - 1) if n > 1 else 1
    k10 = min(10, n - 1) if n > 1 else 1
    k20 = min(20, n - 1) if n > 1 else 1

    mean_top5 = np.mean(sorted_sim[:, :k5], axis=1) if n > 1 else np.zeros(n)
    mean_top10 = np.mean(sorted_sim[:, :k10], axis=1) if n > 1 else np.zeros(n)
    mean_top20 = np.mean(sorted_sim[:, :k20], axis=1) if n > 1 else np.zeros(n)

    # For k nearest neighbors distance (local density inverse)
    mean_dist_k5 = 1.0 - mean_top5
    mean_dist_k10 = 1.0 - mean_top10
    mean_dist_k20 = 1.0 - mean_top20

    df = pd.DataFrame(
        {
            "molecule_id": mol_ids,
            "max_tanimoto_similarity": max_sim,
            "nearest_neighbor_distance": nn_dist,
            "nearest_neighbor_id": nn_ids,
            "mean_top5_similarity": mean_top5,
            "mean_top10_similarity": mean_top10,
            "mean_top20_similarity": mean_top20,
            "local_sparsity_k5": mean_dist_k5,
            "local_sparsity_k10": mean_dist_k10,
            "local_sparsity_k20": mean_dist_k20,
        }
    )
    return df


def compute_annotation_outlier_metrics(
    odor_df: pd.DataFrame,
    metadata_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Analyze descriptor count and annotation rarity patterns across molecules.

    Parameters
    ----------
    odor_df : pd.DataFrame
        DataFrame of binary odor descriptors.
    metadata_cols : Optional[List[str]], default=None
        Columns to exclude (e.g. ['molecule_id', 'canonical_smiles']).

    Returns
    -------
    pd.DataFrame
        Columns: molecule_id, annotation_count, mean_descriptor_frequency,
        annotation_rarity_score (1.0 - mean_frequency).
    """
    if metadata_cols is None:
        meta = {"molecule_id", "canonical_smiles", "cluster_hdbscan"}
    else:
        meta = set(metadata_cols)

    desc_cols = [c for c in odor_df.columns if c not in meta]
    Y = odor_df[desc_cols].to_numpy()
    n_molecules, n_descriptors = Y.shape

    # Frequency of each descriptor across the entire dataset
    desc_freq = Y.mean(axis=0)  # Shape (D,)

    # Number of active annotations per molecule
    annot_counts = Y.sum(axis=1)

    # Mean frequency of the descriptors possessed by each molecule
    # (If count == 0, frequency is 0.0)
    dot_freq = np.dot(Y, desc_freq)
    mean_desc_freq = np.divide(
        dot_freq, annot_counts, out=np.zeros(n_molecules, dtype=float), where=annot_counts > 0
    )

    # Rarity score: 1.0 - mean_frequency (higher = possessing rarer descriptors)
    rarity_score = np.where(annot_counts > 0, 1.0 - mean_desc_freq, 0.0)

    mol_ids = odor_df["molecule_id"].tolist() if "molecule_id" in odor_df.columns else [
        f"MOL_{i:05d}" for i in range(n_molecules)
    ]

    df = pd.DataFrame(
        {
            "molecule_id": mol_ids,
            "annotation_count": annot_counts,
            "mean_descriptor_frequency": mean_desc_freq,
            "annotation_rarity_score": rarity_score,
            "is_unannotated": (annot_counts == 0).astype(int),
        }
    )
    return df


def compute_prediction_outlier_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    molecule_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute predictive entropy, Hamming error, and BCE loss per molecule.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (N, D).
    y_probs : np.ndarray
        Model predicted probabilities in [0, 1] (N, D).
    molecule_ids : Optional[List[str]], default=None
        Molecule identifiers.

    Returns
    -------
    pd.DataFrame
        DataFrame of predictive uncertainty and error metrics.
    """
    n, d = y_true.shape
    if y_probs.shape != (n, d):
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_probs {y_probs.shape}")

    if molecule_ids is None:
        mol_ids = [f"MOL_{i:05d}" for i in range(n)]
    else:
        mol_ids = list(molecule_ids)

    # 1. Predictive entropy per molecule: mean of binary entropy across D descriptors
    eps = 1e-7
    p_clipped = np.clip(y_probs, eps, 1.0 - eps)
    binary_entropy = - (p_clipped * np.log2(p_clipped) + (1.0 - p_clipped) * np.log2(1.0 - p_clipped))
    mean_entropy = np.mean(binary_entropy, axis=1)

    # 2. Prediction error: Hamming loss (fraction of mismatched binary predictions at threshold 0.5)
    preds_bin = (y_probs >= 0.5).astype(int)
    hamming_error = np.mean(preds_bin != y_true, axis=1)

    # 3. Prediction error: Binary Cross-Entropy loss per molecule
    bce_loss = -np.mean(
        y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped), axis=1
    )

    df = pd.DataFrame(
        {
            "molecule_id": mol_ids,
            "predictive_entropy": mean_entropy,
            "hamming_error": hamming_error,
            "bce_loss": bce_loss,
        }
    )
    return df


class MolecularOutlierEvaluator:
    """Integrated evaluator for analyzing and correlating all five outlier dimensions."""

    def __init__(self, top_percentile: float = 95.0) -> None:
        self.top_percentile = top_percentile

    def consolidate_outlier_profile(
        self,
        novelty_df: pd.DataFrame,
        hdbscan_df: pd.DataFrame,
        annotation_df: pd.DataFrame,
        prediction_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Merge all outlier dimensions into a unified analytical profile.

        Parameters
        ----------
        novelty_df : pd.DataFrame
            Structural novelty & local density metrics.
        hdbscan_df : pd.DataFrame
            HDBSCAN output (cluster_hdbscan, membership_strength, outlier_score).
        annotation_df : pd.DataFrame
            Annotation count & descriptor rarity scores.
        prediction_df : Optional[pd.DataFrame], default=None
            Predictive entropy & loss metrics.

        Returns
        -------
        pd.DataFrame
            Unified profile with percentile rank tags for each distinct dimension.
        """
        # Ensure row alignment by molecule_id
        merged = novelty_df.merge(hdbscan_df, on="molecule_id", how="inner")
        merged = merged.merge(annotation_df, on="molecule_id", how="inner")
        if prediction_df is not None:
            merged = merged.merge(prediction_df, on="molecule_id", how="inner")

        # Compute percentile ranks for each distinct axis
        # Higher score = more extreme outlier on that dimension
        merged["rank_structural_novelty"] = stats.percentileofscore(
            merged["nearest_neighbor_distance"], merged["nearest_neighbor_distance"]
        )
        merged["rank_local_sparsity"] = stats.percentileofscore(
            merged["local_sparsity_k10"], merged["local_sparsity_k10"]
        )
        merged["rank_glosh_outlier"] = stats.percentileofscore(
            merged["outlier_score"], merged["outlier_score"]
        )
        merged["rank_annotation_rarity"] = stats.percentileofscore(
            merged["annotation_rarity_score"], merged["annotation_rarity_score"]
        )

        if "predictive_entropy" in merged.columns:
            merged["rank_predictive_entropy"] = stats.percentileofscore(
                merged["predictive_entropy"], merged["predictive_entropy"]
            )
        if "bce_loss" in merged.columns:
            merged["rank_prediction_error"] = stats.percentileofscore(
                merged["bce_loss"], merged["bce_loss"]
            )

        # Flag candidate outlier categories at top percentile (e.g. >= 95th percentile)
        p = self.top_percentile
        merged["is_structural_outlier"] = merged["nearest_neighbor_distance"] >= np.percentile(
            merged["nearest_neighbor_distance"], p
        )
        merged["is_density_outlier"] = merged["local_sparsity_k10"] >= np.percentile(
            merged["local_sparsity_k10"], p
        )
        merged["is_glosh_outlier"] = merged["outlier_score"] >= np.percentile(
            merged["outlier_score"], p
        )
        merged["is_annotation_outlier"] = merged["annotation_rarity_score"] >= np.percentile(
            merged["annotation_rarity_score"], p
        )
        if "predictive_entropy" in merged.columns:
            merged["is_uncertainty_outlier"] = merged["predictive_entropy"] >= np.percentile(
                merged["predictive_entropy"], p
            )

        return merged

    def compute_cross_dimension_correlations(
        self, profile_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute Spearman rank correlations between all outlier dimensions.

        Parameters
        ----------
        profile_df : pd.DataFrame
            Consolidated outlier profile.

        Returns
        -------
        pd.DataFrame
            Pairwise Spearman correlation matrix and p-values.
        """
        candidate_cols = [
            "nearest_neighbor_distance",
            "local_sparsity_k10",
            "outlier_score",
            "annotation_count",
            "annotation_rarity_score",
            "predictive_entropy",
            "bce_loss",
            "hamming_error",
        ]
        active_cols = [c for c in candidate_cols if c in profile_df.columns]

        corr_matrix = pd.DataFrame(index=active_cols, columns=active_cols, dtype=float)
        pval_matrix = pd.DataFrame(index=active_cols, columns=active_cols, dtype=float)

        for c1 in active_cols:
            for c2 in active_cols:
                r, p = stats.spearmanr(profile_df[c1], profile_df[c2])
                corr_matrix.loc[c1, c2] = float(r)
                pval_matrix.loc[c1, c2] = float(p)

        return corr_matrix

