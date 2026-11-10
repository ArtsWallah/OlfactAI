"""HDBSCAN unsupervised molecular clustering and chemical space partition.

This module implements a scientifically rigorous, density-based clustering workflow
designed for molecular datasets using chemical fingerprints (Morgan circular fingerprints)
and exploratory manifold embeddings (UMAP 10D and 2D).

Key scientific principles:
1. Primary representation: High-dimensional Morgan fingerprints (radius=2, 2048-bit)
   with Jaccard distance (1.0 - Tanimoto similarity).
2. Exploratory representation: 10D UMAP embeddings with Euclidean metric.
3. Visualization-only representation: 2D UMAP coordinates (strictly exploratory,
   never presented as primary chemical families).
4. Resource awareness: Efficient BLAS matrix operations for pairwise distance computation
   with exact memory tracking and validation (symmetry, zero diagonal, bounds).
5. Statistical integrity: Density-based noise (-1) is preserved and reported;
   cluster labels are arbitrary numerical identifiers, not categorical odor names.
"""

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from backend.app.core.logging import get_logger

logger = get_logger("clustering.hdbscan")


def estimate_distance_matrix_memory_mb(n_samples: int, dtype: Union[np.dtype, type] = np.float64) -> float:
    """Calculate the exact RAM consumption in megabytes for a dense n_samples x n_samples matrix.

    Args:
        n_samples: Number of molecules.
        dtype: Data type (default: np.float64, 8 bytes per element).

    Returns:
        Estimated memory in megabytes (MB).
    """
    bytes_per_elem = np.dtype(dtype).itemsize
    total_bytes = n_samples * n_samples * bytes_per_elem
    return float(total_bytes / (1024.0 * 1024.0))


def compute_jaccard_distance_matrix(fingerprint_matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise Jaccard distance matrix (1.0 - Tanimoto similarity) via BLAS.

    For binary molecular fingerprints X (n_samples, n_bits):
        Intersection_{ij} = (X @ X.T)_{ij}
        Union_{ij} = OnBits_i + OnBits_j - Intersection_{ij}
        Tanimoto_{ij} = Intersection_{ij} / Union_{ij}
        Distance_{ij} = 1.0 - Tanimoto_{ij}

    For molecules with zero on-bits (e.g. single inorganic atoms like [Zn]):
        - Self-distance Distance_{ii} is defined as 0.0.
        - Pairwise distance to any other molecule is 1.0.

    Args:
        fingerprint_matrix: 2D array of shape (n_samples, n_bits).

    Returns:
        Square symmetric matrix of shape (n_samples, n_samples) with dtype np.float64,
        diagonal identically 0.0, and values bounded in [0.0, 1.0].
    """
    if fingerprint_matrix.ndim != 2:
        raise ValueError(f"fingerprint_matrix must be 2D, got shape {fingerprint_matrix.shape}")
    n_samples, _ = fingerprint_matrix.shape
    if n_samples < 2:
        raise ValueError(f"At least 2 samples required to compute distance matrix, got {n_samples}")

    X = fingerprint_matrix.astype(np.float32)

    # 1. BLAS matrix multiplication for pairwise intersection: O(N^2 * D)
    intersection = np.dot(X, X.T)

    # 2. Extract on-bit counts along diagonal
    on_bits = np.diag(intersection)

    # 3. Pairwise union: OnBits_i + OnBits_j - Intersection_ij
    union = on_bits[:, None] + on_bits[None, :] - intersection

    # 4. Handle division by zero for molecules with 0 on-bits
    safe_union = np.maximum(union, 1e-9)
    tanimoto = np.where(union > 0, intersection / safe_union, 0.0)

    # 5. Jaccard distance = 1.0 - Tanimoto
    jaccard_dist = (1.0 - tanimoto).astype(np.float64)

    # 6. Strict metric guarantees: zero diagonal, symmetry, bounds [0, 1]
    np.fill_diagonal(jaccard_dist, 0.0)
    # Symmetrize to eliminate any floating-point rounding asymmetry
    jaccard_dist = 0.5 * (jaccard_dist + jaccard_dist.T)
    np.clip(jaccard_dist, 0.0, 1.0, out=jaccard_dist)
    np.fill_diagonal(jaccard_dist, 0.0)

    return jaccard_dist


def validate_distance_matrix(distance_matrix: np.ndarray, check_bounds: bool = True) -> None:
    """Validate that a precomputed distance matrix satisfies metric space requirements.

    Checks:
    1. 2D square matrix: shape (N, N).
    2. Finite values: no NaN, Inf, or -Inf.
    3. Zero diagonal: D[i, i] == 0 for all i.
    4. Exact symmetry: D[i, j] == D[j, i] within machine tolerance.
    5. Bounds: values in [0.0, 1.0] if check_bounds is True.

    Args:
        distance_matrix: Candidate distance matrix.
        check_bounds: Whether to enforce [0.0, 1.0] bounds.

    Raises:
        ValueError: If any metric requirement is violated.
    """
    if not isinstance(distance_matrix, np.ndarray):
        raise ValueError(f"Distance matrix must be a numpy ndarray, got {type(distance_matrix)}")
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(f"Distance matrix must be square 2D, got shape {distance_matrix.shape}")
    if distance_matrix.shape[0] < 2:
        raise ValueError(f"Distance matrix must have at least 2 rows/cols, got {distance_matrix.shape[0]}")

    if not np.isfinite(distance_matrix).all():
        raise ValueError("Distance matrix contains non-finite values (NaN or Inf).")

    diag = np.diag(distance_matrix)
    if not np.allclose(diag, 0.0, atol=1e-7):
        max_diag = float(np.max(np.abs(diag)))
        raise ValueError(f"Distance matrix diagonal must be identically zero; max diagonal value is {max_diag}")

    # Check symmetry
    diff = np.abs(distance_matrix - distance_matrix.T)
    if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-5):
        max_diff = float(np.max(diff))
        raise ValueError(f"Distance matrix is asymmetric; maximum asymmetry is {max_diff}")

    if check_bounds:
        if (distance_matrix < -1e-7).any():
            min_val = float(np.min(distance_matrix))
            raise ValueError(f"Distance matrix contains negative values; minimum value is {min_val}")
        if (distance_matrix > 1.0 + 1e-7).any():
            max_val = float(np.max(distance_matrix))
            raise ValueError(f"Distance matrix exceeds upper bound 1.0; maximum value is {max_val}")


@dataclass
class ClusterSummary:
    """Summary statistics for a fitted HDBSCAN clustering model."""
    n_samples: int
    n_clusters: int
    n_noise: int
    noise_fraction: float
    cluster_sizes: Dict[int, int]
    min_cluster_size_observed: int
    median_cluster_size: float
    max_cluster_size: int
    mean_membership_strength: float
    median_membership_strength: float
    iqr_membership_strength: float
    clustered_mean_membership_strength: float
    clustered_median_membership_strength: float
    cluster_persistence: Dict[int, float]
    silhouette_score_clustered: Optional[float]
    silhouette_caveat: str
    runtime_seconds: float
    peak_memory_mb: float


class MolecularHDBSCAN:
    """Configurable HDBSCAN clustering engine for molecular structure discovery.

    Supports:
    - Precomputed Jaccard distance matrices (primary Morgan fingerprint experiment).
    - Euclidean feature matrices (10D and 2D UMAP exploratory representations).
    - Faithful extraction of library attributes:
        * probabilities_: Membership strengths (relative density reachability, bounded in [0, 1]).
          Note: Membership strengths are heuristic density-tree values, NOT calibrated probabilities
          and NOT raw lambda values.
        * cluster_persistence_: Cluster stability across condensed tree simplifications from the
          underlying library model attribute.
        * outlier_scores_: GLOSH (Global-Local Outlier Score from Hierarchies).
    - Documented project convention for GLOSH: Detached singleton/noise points whose reachability
      is undefined in the MST and evaluate to NaN are imputed to 1.0 (maximal outlier status).
    - Molecule row alignment preservation.
    """

    def __init__(
        self,
        min_cluster_size: int = 15,
        min_samples: Optional[int] = None,
        metric: str = "precomputed",
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
        allow_single_cluster: bool = False,
    ):
        """Initialize HDBSCAN clustering configuration.

        Args:
            min_cluster_size: Sets the minimum group size for a selected cluster in the
                condensed tree hierarchy (controls cluster granularity / minimum cardinality).
            min_samples: Controls the local density requirement through the k-nearest-neighbor
                core distance (where k = min_samples, counting the molecule itself). Defines
                the mutual reachability distance d_mreach-k(a, b) = max(d_core(a), d_core(b), d(a, b)).
                If None, defaults to min_cluster_size inside HDBSCAN.
            metric: Distance metric ('precomputed' or standard sklearn metric like 'euclidean').
            cluster_selection_epsilon: Distance threshold below which clusters are merged.
            cluster_selection_method: 'eom' (Excess of Mass) or 'leaf'.
            allow_single_cluster: Whether to allow a single cluster solution.
        """
        if min_cluster_size < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {min_cluster_size}")
        if min_samples is not None and min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")
        if cluster_selection_method not in ("eom", "leaf"):
            raise ValueError(f"cluster_selection_method must be 'eom' or 'leaf', got {cluster_selection_method}")

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster

        self.clusterer_: Optional[hdbscan.HDBSCAN] = None
        self.labels_: Optional[np.ndarray] = None
        self.probabilities_: Optional[np.ndarray] = None
        self.outlier_scores_: Optional[np.ndarray] = None
        self.cluster_persistence_: Dict[int, float] = {}
        self.fit_time_seconds_: float = 0.0
        self.is_fitted_: bool = False

    def fit(
        self,
        X: np.ndarray,
        distance_matrix: Optional[np.ndarray] = None,
    ) -> "MolecularHDBSCAN":
        """Fit HDBSCAN model on feature matrix or precomputed distance matrix.

        Args:
            X: Input data matrix. If metric='precomputed', X can be None if distance_matrix is passed,
               or X can be the distance matrix itself.
            distance_matrix: Optional explicit precomputed distance matrix.

        Returns:
            self: Fitted instance.
        """
        t0 = time.perf_counter()

        # Resolve distance matrix or feature matrix
        if self.metric == "precomputed":
            target_data = distance_matrix if distance_matrix is not None else X
            if target_data is None:
                raise ValueError("For metric='precomputed', a distance matrix must be provided.")
            validate_distance_matrix(target_data, check_bounds=(target_data.max() <= 1.0 + 1e-5))
            # HDBSCAN expects float64 precomputed distance matrices
            fit_input = target_data.astype(np.float64, copy=False)
        else:
            if X is None:
                raise ValueError(f"For metric='{self.metric}', feature matrix X must be provided.")
            if not np.isfinite(X).all():
                raise ValueError("Input feature matrix contains non-finite values (NaN or Inf).")
            fit_input = X.astype(np.float64, copy=False)

        # Initialize HDBSCAN model
        self.clusterer_ = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
            gen_min_span_tree=True,
        )

        self.clusterer_.fit(fit_input)

        self.labels_ = self.clusterer_.labels_.copy()
        self.probabilities_ = self.clusterer_.probabilities_.copy()

        # Outlier scores (GLOSH: Global-Local Outlier Score from Hierarchies)
        if hasattr(self.clusterer_, "outlier_scores_") and self.clusterer_.outlier_scores_ is not None:
            raw_scores = self.clusterer_.outlier_scores_.copy()
            # In density trees, detached singleton/noise points can produce NaN; impute to 1.0 (maximum outlier)
            self.outlier_scores_ = np.nan_to_num(raw_scores, nan=1.0)
        else:
            self.outlier_scores_ = 1.0 - self.probabilities_

        # Cluster persistence extraction
        self.cluster_persistence_ = {}
        if hasattr(self.clusterer_, "cluster_persistence_") and self.clusterer_.cluster_persistence_ is not None:
            raw_persistence = self.clusterer_.cluster_persistence_
            # If available as 1D array corresponding to selected clusters
            if isinstance(raw_persistence, np.ndarray) and len(raw_persistence) > 0:
                unique_clusters = sorted([c for c in set(self.labels_) if c != -1])
                for idx, c_id in enumerate(unique_clusters):
                    if idx < len(raw_persistence):
                        self.cluster_persistence_[int(c_id)] = float(raw_persistence[idx])

        self.fit_time_seconds_ = time.perf_counter() - t0
        self.is_fitted_ = True
        return self

    @property
    def raw_cluster_persistence(self) -> Optional[np.ndarray]:
        """Return the raw cluster_persistence_ array directly from the underlying HDBSCAN model."""
        if self.clusterer_ is not None and hasattr(self.clusterer_, "cluster_persistence_"):
            return self.clusterer_.cluster_persistence_
        return None

    def fit_predict(
        self,
        X: np.ndarray,
        distance_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit HDBSCAN and return cluster labels (-1 indicates noise)."""
        self.fit(X, distance_matrix=distance_matrix)
        assert self.labels_ is not None
        return self.labels_

    def get_cluster_summary(
        self,
        distance_matrix: Optional[np.ndarray] = None,
        feature_matrix: Optional[np.ndarray] = None,
    ) -> ClusterSummary:
        """Calculate comprehensive cluster and membership strength summary statistics.

        Args:
            distance_matrix: Distance matrix if silhouette score on precomputed distances is desired.
            feature_matrix: Feature matrix if silhouette score on Euclidean features is desired.

        Returns:
            ClusterSummary dataclass.
        """
        if not self.is_fitted_ or self.labels_ is None or self.probabilities_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before requesting summary.")

        n_samples = len(self.labels_)
        unique_labels = sorted(set(self.labels_))
        non_noise_labels = [c for c in unique_labels if c != -1]
        n_clusters = len(non_noise_labels)
        n_noise = int(np.sum(self.labels_ == -1))
        noise_fraction = float(n_noise / n_samples)

        # Cluster size counts
        cluster_sizes: Dict[int, int] = {}
        for c in unique_labels:
            cluster_sizes[int(c)] = int(np.sum(self.labels_ == c))

        non_noise_sizes = [cluster_sizes[c] for c in non_noise_labels]
        if len(non_noise_sizes) > 0:
            min_c = int(np.min(non_noise_sizes))
            med_c = float(np.median(non_noise_sizes))
            max_c = int(np.max(non_noise_sizes))
        else:
            min_c, med_c, max_c = 0, 0.0, 0

        # Membership strengths
        all_probs = self.probabilities_
        clustered_mask = self.labels_ != -1
        clustered_probs = all_probs[clustered_mask] if np.any(clustered_mask) else np.array([0.0])

        q75, q25 = np.percentile(all_probs, [75, 25])
        iqr_strength = float(q75 - q25)

        # Silhouette score computation on clustered (non-noise) subset
        silhouette_clustered: Optional[float] = None
        sil_caveat = (
            "Silhouette score is computed exclusively on non-noise molecules (labels != -1) "
            "because density-based noise points are intentionally scattered and would artificially "
            "distort intra-cluster cohesion metrics. Silhouette scores cannot be directly compared "
            "across different representations or distance metrics."
        )

        if n_clusters >= 2 and np.sum(clustered_mask) >= n_clusters * 2:
            try:
                clustered_labels = self.labels_[clustered_mask]
                if self.metric == "precomputed" and distance_matrix is not None:
                    sub_dists = distance_matrix[np.ix_(clustered_mask, clustered_mask)]
                    silhouette_clustered = float(silhouette_score(sub_dists, clustered_labels, metric="precomputed"))
                elif self.metric != "precomputed" and feature_matrix is not None:
                    sub_feat = feature_matrix[clustered_mask]
                    silhouette_clustered = float(silhouette_score(sub_feat, clustered_labels, metric=self.metric))
            except Exception as e:
                logger.warning(f"Could not compute silhouette score: {e}")
                silhouette_clustered = None

        return ClusterSummary(
            n_samples=n_samples,
            n_clusters=n_clusters,
            n_noise=n_noise,
            noise_fraction=noise_fraction,
            cluster_sizes=cluster_sizes,
            min_cluster_size_observed=min_c,
            median_cluster_size=med_c,
            max_cluster_size=max_c,
            mean_membership_strength=float(np.mean(all_probs)),
            median_membership_strength=float(np.median(all_probs)),
            iqr_membership_strength=iqr_strength,
            clustered_mean_membership_strength=float(np.mean(clustered_probs)),
            clustered_median_membership_strength=float(np.median(clustered_probs)),
            cluster_persistence=self.cluster_persistence_,
            silhouette_score_clustered=silhouette_clustered,
            silhouette_caveat=sil_caveat,
            runtime_seconds=float(self.fit_time_seconds_),
            peak_memory_mb=estimate_distance_matrix_memory_mb(n_samples),
        )

    def assign_clusters_to_dataframe(
        self,
        df: pd.DataFrame,
        cluster_col: str = "cluster_hdbscan",
        prob_col: str = "membership_strength",
        outlier_col: str = "outlier_score",
    ) -> pd.DataFrame:
        """Assign cluster labels, membership strengths, and outlier scores to a DataFrame.

        Preserves exact input row alignment and verifies that row counts match.

        Args:
            df: Input DataFrame containing molecules.
            cluster_col: Destination column name for cluster labels.
            prob_col: Destination column name for membership strengths.
            outlier_col: Destination column name for outlier scores.

        Returns:
            DataFrame copy with assigned clustering columns.
        """
        if not self.is_fitted_ or self.labels_ is None or self.probabilities_ is None:
            raise RuntimeError("Model must be fitted before assigning clusters to DataFrame.")
        if len(df) != len(self.labels_):
            raise ValueError(f"DataFrame row count ({len(df)}) does not match fitted sample count ({len(self.labels_)})")

        out = df.copy()
        out[cluster_col] = self.labels_
        out[prob_col] = self.probabilities_
        out[outlier_col] = self.outlier_scores_ if self.outlier_scores_ is not None else (1.0 - self.probabilities_)
        return out


def run_parameter_sensitivity_grid(
    distance_matrix: np.ndarray,
    min_cluster_size_candidates: List[int],
    min_samples_candidates: List[Optional[int]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute a controlled parameter sensitivity grid for HDBSCAN on a precomputed distance matrix.

    Args:
        distance_matrix: Precomputed symmetric distance matrix (e.g. Morgan fingerprint Jaccard).
        min_cluster_size_candidates: List of candidate min_cluster_size values.
        min_samples_candidates: List of candidate min_samples values (or None).

    Returns:
        Tuple of (results_dataframe, detailed_metadata_dict).
    """
    records: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

    for mcs in min_cluster_size_candidates:
        for ms in min_samples_candidates:
            config_id = f"mcs_{mcs}_ms_{ms if ms is not None else 'default'}"
            t0 = time.perf_counter()
            clusterer = MolecularHDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric="precomputed",
            )
            clusterer.fit(X=distance_matrix)
            summary = clusterer.get_cluster_summary(distance_matrix=distance_matrix)
            elapsed = time.perf_counter() - t0

            record = {
                "config_id": config_id,
                "min_cluster_size": mcs,
                "min_samples": ms if ms is not None else -1,
                "n_clusters": summary.n_clusters,
                "n_noise": summary.n_noise,
                "noise_fraction": summary.noise_fraction,
                "min_cluster_size_observed": summary.min_cluster_size_observed,
                "median_cluster_size": summary.median_cluster_size,
                "max_cluster_size": summary.max_cluster_size,
                "mean_membership_strength": summary.mean_membership_strength,
                "median_membership_strength": summary.median_membership_strength,
                "clustered_mean_membership_strength": summary.clustered_mean_membership_strength,
                "clustered_median_membership_strength": summary.clustered_median_membership_strength,
                "silhouette_score_clustered": summary.silhouette_score_clustered,
                "runtime_seconds": round(elapsed, 3),
            }
            records.append(record)
            metadata[config_id] = asdict(summary)

    results_df = pd.DataFrame(records)
    return results_df, metadata


def compare_representations(
    fingerprint_distance_matrix: np.ndarray,
    umap_10d_matrix: np.ndarray,
    umap_2d_matrix: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: Optional[int] = 5,
) -> Tuple[pd.DataFrame, Dict[str, ClusterSummary]]:
    """Compare HDBSCAN clustering behavior across primary, exploratory, and visualization representations.

    Representations evaluated:
    A. Primary: Morgan fingerprints with Jaccard distance (metric='precomputed').
    B. Exploratory: 10D UMAP embeddings with Euclidean distance (metric='euclidean').
    C. Visualization: 2D UMAP embeddings with Euclidean distance (metric='euclidean').

    Args:
        fingerprint_distance_matrix: (N, N) precomputed Jaccard distance matrix.
        umap_10d_matrix: (N, 10) 10D UMAP coordinates.
        umap_2d_matrix: (N, 2) 2D UMAP coordinates.
        min_cluster_size: Minimum cluster size for all runs.
        min_samples: Minimum samples for all runs.

    Returns:
        Tuple of (comparison_dataframe, summary_dict).
    """
    experiments = [
        ("Primary: Morgan FP (Jaccard)", "precomputed", fingerprint_distance_matrix, fingerprint_distance_matrix, None),
        ("Exploratory: 10D UMAP (Euclidean)", "euclidean", umap_10d_matrix, None, umap_10d_matrix),
        ("Visualization: 2D UMAP (Euclidean)", "euclidean", umap_2d_matrix, None, umap_2d_matrix),
    ]

    records = []
    summaries: Dict[str, ClusterSummary] = {}

    for name, metric, data, dist_mat, feat_mat in experiments:
        clusterer = MolecularHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
        )
        if metric == "precomputed":
            clusterer.fit(X=data)
        else:
            clusterer.fit(X=data)

        summary = clusterer.get_cluster_summary(distance_matrix=dist_mat, feature_matrix=feat_mat)
        summaries[name] = summary

        records.append({
            "representation": name,
            "metric": metric,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "n_clusters": summary.n_clusters,
            "n_noise": summary.n_noise,
            "noise_fraction": summary.noise_fraction,
            "min_cluster_size": summary.min_cluster_size_observed,
            "median_cluster_size": summary.median_cluster_size,
            "max_cluster_size": summary.max_cluster_size,
            "clustered_mean_membership": summary.clustered_mean_membership_strength,
            "silhouette_clustered": summary.silhouette_score_clustered,
            "runtime_seconds": summary.runtime_seconds,
        })

    df = pd.DataFrame(records)
    return df, summaries
