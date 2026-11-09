"""Molecular structural neighborhood and odor-profile agreement analysis."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from backend.app.similarity.engine import MolecularSimilarityEngine


def calculate_jaccard_similarity(
    set_a: Union[Set[str], List[str]],
    set_b: Union[Set[str], List[str]],
    empty_empty_val: float = 0.0,
) -> float:
    """Calculate the Jaccard similarity coefficient between two sets of odor labels.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        set_a: First collection of odor label strings.
        set_b: Second collection of odor label strings.
        empty_empty_val: Value returned when both sets are empty (default 0.0).

    Returns:
        Float value between 0.0 and 1.0.
    """
    s_a = set(set_a)
    s_b = set(set_b)

    if not s_a and not s_b:
        return empty_empty_val

    intersection_len = len(s_a.intersection(s_b))
    union_len = len(s_a.union(s_b))

    if union_len == 0:
        return empty_empty_val

    return float(intersection_len / union_len)


def calculate_label_overlap(
    set_a: Union[Set[str], List[str]],
    set_b: Union[Set[str], List[str]],
) -> Tuple[int, List[str]]:
    """Compute count and list of shared odor labels between two molecules.

    Args:
        set_a: First collection of odor labels.
        set_b: Second collection of odor labels.

    Returns:
        Tuple of (overlap_count, sorted_shared_labels).
    """
    shared = sorted(list(set(set_a).intersection(set(set_b))))
    return len(shared), shared


@dataclass
class NeighborOdorMatch:
    """Detailed structural and odor comparison for a single nearest neighbor."""

    rank: int
    molecule_id: str
    canonical_smiles: str
    tanimoto_similarity: float
    neighbor_odor_labels: List[str]
    shared_odor_labels: List[str]
    label_overlap_count: int
    odor_jaccard_similarity: float


@dataclass
class NeighborhoodResult:
    """Result summary of a molecule's K-nearest structural neighborhood."""

    query: str
    query_molecule_id: Optional[str]
    query_canonical_smiles: str
    query_odor_labels: List[str]
    k: int
    mean_structural_similarity: float
    median_structural_similarity: float
    mean_odor_jaccard: float
    median_odor_jaccard: float
    zero_odor_overlap_proportion: float
    neighbor_label_frequencies: Dict[str, float]
    top_neighbor_labels: List[Tuple[str, int, float]]
    neighbors: List[NeighborOdorMatch] = field(default_factory=list)


@dataclass
class TanimotoBinSummary:
    """Aggregated statistics for a specific structural similarity interval."""

    bin_label: str
    tanimoto_min: float
    tanimoto_max: float
    pair_count: int
    proportion_of_total: float
    mean_tanimoto: float
    mean_odor_jaccard: float
    median_odor_jaccard: float
    std_odor_jaccard: float
    iqr_odor_jaccard: float
    zero_overlap_count: int
    zero_overlap_proportion: float


@dataclass
class ClusterBootstrapSummary:
    """Results from molecule-level cluster bootstrap for dependence-aware uncertainty."""

    resampling_unit: str  # "molecule"
    n_iterations: int
    sample_size_molecules: int
    pearson_r_point_estimate: float
    pearson_r_bootstrap_mean: float
    pearson_r_bootstrap_se: float
    pearson_r_ci_95: Tuple[float, float]
    spearman_rho_point_estimate: float
    spearman_rho_bootstrap_mean: float
    spearman_rho_bootstrap_se: float
    spearman_rho_ci_95: Tuple[float, float]
    methodological_note: str


@dataclass
class GlobalPairwiseSummary:
    """Complete global pairwise analysis across all 12,412,653 unique pairs."""

    total_molecules: int
    total_unique_pairs: int
    pearson_correlation: float
    spearman_correlation: float
    mean_tanimoto: float
    median_tanimoto: float
    mean_odor_jaccard: float
    median_odor_jaccard: float
    zero_odor_overlap_total_pairs: int
    zero_odor_overlap_overall_proportion: float
    binned_analysis: List[TanimotoBinSummary] = field(default_factory=list)
    cluster_bootstrap: Optional[ClusterBootstrapSummary] = None


class MolecularNeighborhoodAnalyzer:
    """Performs per-molecule and global neighborhood analysis over the indexed corpus."""

    DEFAULT_BINS = [
        (0.0, 0.20, "[0.00, 0.20)"),
        (0.20, 0.30, "[0.20, 0.30)"),
        (0.30, 0.40, "[0.30, 0.40)"),
        (0.40, 0.50, "[0.40, 0.50)"),
        (0.50, 0.60, "[0.50, 0.60)"),
        (0.60, 0.70, "[0.60, 0.70)"),
        (0.70, 0.80, "[0.70, 0.80)"),
        (0.80, 1.0001, "[0.80, 1.00]"),
    ]

    def __init__(self, engine: Optional[MolecularSimilarityEngine] = None):
        """Initialize analyzer with an existing or new MolecularSimilarityEngine.

        Args:
            engine: Loaded MolecularSimilarityEngine instance.
        """
        self.engine = engine if engine is not None else MolecularSimilarityEngine(auto_load=True)
        if not self.engine.is_loaded:
            self.engine.load_dataset()

        # Cache binary label matrix as uint8 for ultra-fast vectorized operations
        labels_path = self.engine.dataset_dir / "odor_labels.csv"
        labels_df = pd.read_csv(labels_path)
        self.odor_cols = [c for c in labels_df.columns if c not in ["molecule_id", "canonical_smiles"]]
        self.label_matrix = labels_df[self.odor_cols].values.astype(np.uint8)
        self.label_counts = self.label_matrix.sum(axis=1)

    def analyze_neighborhood(
        self,
        query: str,
        k: int = 10,
        empty_empty_val: float = 0.0,
    ) -> NeighborhoodResult:
        """Analyze the nearest structural neighborhood for a specific query molecule.

        Args:
            query: Dataset molecule ID (e.g. 'MOL_00001') or chemical SMILES.
            k: Neighborhood size (must be > 0).
            empty_empty_val: Default Jaccard value when both query and neighbor have no labels.

        Returns:
            NeighborhoodResult containing detailed structural and odor agreement metrics.
        """
        if k <= 0:
            raise ValueError(f"Neighborhood size k must be positive, got {k}")

        search_res = self.engine.search(query=query, top_k=k, exclude_self=True)

        # Retrieve active query labels
        if search_res.query_molecule_id and search_res.query_molecule_id in self.engine._id_to_index:
            q_idx = self.engine._id_to_index[search_res.query_molecule_id]
            query_labels = list(self.engine.active_labels[q_idx])
        else:
            query_labels = []

        query_label_set = set(query_labels)
        neighbor_matches: List[NeighborOdorMatch] = []
        tanimoto_scores: List[float] = []
        jaccard_scores: List[float] = []
        neighbor_label_counts: Dict[str, int] = {}

        for match in search_res.matches:
            neighbor_labels = list(match.known_odor_labels)
            overlap_cnt, shared_labels = calculate_label_overlap(query_label_set, neighbor_labels)
            jacc = calculate_jaccard_similarity(query_label_set, neighbor_labels, empty_empty_val=empty_empty_val)

            neighbor_matches.append(
                NeighborOdorMatch(
                    rank=match.rank,
                    molecule_id=match.molecule_id,
                    canonical_smiles=match.canonical_smiles,
                    tanimoto_similarity=match.tanimoto_similarity,
                    neighbor_odor_labels=neighbor_labels,
                    shared_odor_labels=shared_labels,
                    label_overlap_count=overlap_cnt,
                    odor_jaccard_similarity=round(jacc, 4),
                )
            )

            tanimoto_scores.append(match.tanimoto_similarity)
            jaccard_scores.append(jacc)

            for lbl in neighbor_labels:
                neighbor_label_counts[lbl] = neighbor_label_counts.get(lbl, 0) + 1

        # Summary statistics
        actual_k = len(neighbor_matches)
        if actual_k > 0:
            mean_struct = float(np.mean(tanimoto_scores))
            median_struct = float(np.median(tanimoto_scores))
            mean_jacc = float(np.mean(jaccard_scores))
            median_jacc = float(np.median(jaccard_scores))
            zero_overlap_prop = float(np.mean([1.0 if s == 0.0 else 0.0 for s in jaccard_scores]))
        else:
            mean_struct = median_struct = mean_jacc = median_jacc = zero_overlap_prop = 0.0

        label_freqs = {lbl: round(cnt / actual_k, 4) for lbl, cnt in neighbor_label_counts.items()} if actual_k > 0 else {}
        top_labels = sorted([(lbl, cnt, round(cnt / actual_k, 4)) for lbl, cnt in neighbor_label_counts.items()], key=lambda x: (-x[1], x[0])) if actual_k > 0 else []

        return NeighborhoodResult(
            query=query,
            query_molecule_id=search_res.query_molecule_id,
            query_canonical_smiles=search_res.query_canonical_smiles,
            query_odor_labels=query_labels,
            k=k,
            mean_structural_similarity=round(mean_struct, 4),
            median_structural_similarity=round(median_struct, 4),
            mean_odor_jaccard=round(mean_jacc, 4),
            median_odor_jaccard=round(median_jacc, 4),
            zero_odor_overlap_proportion=round(zero_overlap_prop, 4),
            neighbor_label_frequencies=label_freqs,
            top_neighbor_labels=top_labels,
            neighbors=neighbor_matches,
        )

    def analyze_multi_k(
        self,
        query: str,
        k_values: Optional[List[int]] = None,
        empty_empty_val: float = 0.0,
    ) -> Dict[int, NeighborhoodResult]:
        """Analyze neighborhood metrics across multiple configurable K values."""
        if k_values is None:
            k_values = [5, 10, 20, 50]

        results = {}
        for k in k_values:
            results[k] = self.analyze_neighborhood(query=query, k=k, empty_empty_val=empty_empty_val)
        return results

    def compute_all_pairwise_similarities(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute structural Tanimoto and odor Jaccard similarities for all 12,412,653 pairs.

        Returns:
            Tuple of:
            - tanimoto_arr (float32, length 12,412,653)
            - jaccard_arr (float32, length 12,412,653)
            - pair_i_arr (uint16, length 12,412,653)
            - pair_j_arr (uint16, length 12,412,653)
        """
        n_mols = self.engine.corpus_size
        total_pairs = n_mols * (n_mols - 1) // 2

        tanimoto_arr = np.empty(total_pairs, dtype=np.float32)
        jaccard_arr = np.empty(total_pairs, dtype=np.float32)
        pair_i = np.empty(total_pairs, dtype=np.uint16)
        pair_j = np.empty(total_pairs, dtype=np.uint16)

        offset = 0
        fps = self.engine.fingerprints
        L = self.label_matrix
        counts = self.label_counts

        for i in range(n_mols - 1):
            n_rest = n_mols - 1 - i
            end_offset = offset + n_rest

            # Pairwise Tanimoto via RDKit C++ bulk similarity
            sims = self.engine.DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :]) if hasattr(self.engine, 'DataStructs') else None
            if sims is None:
                from rdkit import DataStructs  # type: ignore
                sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])

            tanimoto_arr[offset:end_offset] = sims

            # Pairwise odor Jaccard via vectorized matrix-vector dot product
            inter = L[i + 1 :].dot(L[i]).astype(np.float32)
            union = (counts[i + 1 :] + counts[i] - inter).astype(np.float32)

            # Avoid division by zero where union == 0
            with np.errstate(divide="ignore", invalid="ignore"):
                jacc = np.where(union > 0, inter / union, 0.0)

            jaccard_arr[offset:end_offset] = jacc
            pair_i[offset:end_offset] = i
            pair_j[offset:end_offset] = np.arange(i + 1, n_mols, dtype=np.uint16)

            offset = end_offset

        return tanimoto_arr, jaccard_arr, pair_i, pair_j

    def compute_global_pairwise_analysis(
        self,
        tanimoto_arr: Optional[np.ndarray] = None,
        jaccard_arr: Optional[np.ndarray] = None,
        custom_bins: Optional[List[Tuple[float, float, str]]] = None,
    ) -> GlobalPairwiseSummary:
        """Execute global pairwise analysis across all 12,412,653 unique pairs.

        Args:
            tanimoto_arr: Precomputed Tanimoto array (or None to compute).
            jaccard_arr: Precomputed Jaccard array (or None to compute).
            custom_bins: Optional custom bin definitions.

        Returns:
            GlobalPairwiseSummary containing descriptive correlations and binned distributions.
        """
        if tanimoto_arr is None or jaccard_arr is None:
            tanimoto_arr, jaccard_arr, _, _ = self.compute_all_pairwise_similarities()

        total_pairs = len(tanimoto_arr)
        bins_def = custom_bins if custom_bins is not None else self.DEFAULT_BINS

        # Descriptive Pearson correlation
        p_corr = float(np.corrcoef(tanimoto_arr, jaccard_arr)[0, 1])

        # Descriptive Spearman rank correlation
        # Use fast subsample or scipy spearmanr on the full dataset
        s_corr, _ = stats.spearmanr(tanimoto_arr, jaccard_arr)
        s_corr = float(s_corr)

        mean_t = float(np.mean(tanimoto_arr))
        median_t = float(np.median(tanimoto_arr))
        mean_j = float(np.mean(jaccard_arr))
        median_j = float(np.median(jaccard_arr))

        zero_overlap_mask = jaccard_arr == 0.0
        total_zero_overlap = int(np.sum(zero_overlap_mask))
        overall_zero_prop = float(total_zero_overlap / total_pairs)

        # Binned analysis
        binned_summaries: List[TanimotoBinSummary] = []
        for t_min, t_max, b_label in bins_def:
            mask = (tanimoto_arr >= t_min) & (tanimoto_arr < t_max)
            bin_count = int(np.sum(mask))

            if bin_count > 0:
                bin_jacc = jaccard_arr[mask]
                bin_t = tanimoto_arr[mask]
                mean_bin_t = float(np.mean(bin_t))
                mean_bin_j = float(np.mean(bin_jacc))
                median_bin_j = float(np.median(bin_jacc))
                std_bin_j = float(np.std(bin_jacc))
                q75, q25 = np.percentile(bin_jacc, [75, 25])
                iqr_bin_j = float(q75 - q25)
                bin_zero_cnt = int(np.sum(bin_jacc == 0.0))
                bin_zero_prop = float(bin_zero_cnt / bin_count)
            else:
                mean_bin_t = mean_bin_j = median_bin_j = std_bin_j = iqr_bin_j = bin_zero_prop = 0.0
                bin_zero_cnt = 0

            binned_summaries.append(
                TanimotoBinSummary(
                    bin_label=b_label,
                    tanimoto_min=t_min,
                    tanimoto_max=t_max,
                    pair_count=bin_count,
                    proportion_of_total=round(float(bin_count / total_pairs), 6),
                    mean_tanimoto=round(mean_bin_t, 4),
                    mean_odor_jaccard=round(mean_bin_j, 4),
                    median_odor_jaccard=round(median_bin_j, 4),
                    std_odor_jaccard=round(std_bin_j, 4),
                    iqr_odor_jaccard=round(iqr_bin_j, 4),
                    zero_overlap_count=bin_zero_cnt,
                    zero_overlap_proportion=round(bin_zero_prop, 4),
                )
            )

        return GlobalPairwiseSummary(
            total_molecules=self.engine.corpus_size,
            total_unique_pairs=total_pairs,
            pearson_correlation=round(p_corr, 4),
            spearman_correlation=round(s_corr, 4),
            mean_tanimoto=round(mean_t, 4),
            median_tanimoto=round(median_t, 4),
            mean_odor_jaccard=round(mean_j, 4),
            median_odor_jaccard=round(median_j, 4),
            zero_odor_overlap_total_pairs=total_zero_overlap,
            zero_odor_overlap_overall_proportion=round(overall_zero_prop, 4),
            binned_analysis=binned_summaries,
        )

    def compute_molecule_cluster_bootstrap(
        self,
        n_iterations: int = 100,
        subsample_size: int = 1500,
        seed: int = 42,
    ) -> ClusterBootstrapSummary:
        """Perform molecule-level cluster bootstrap to assess dependence-aware uncertainty.

        Methodology:
        Rather than resampling pairs independently (which violates independence because pairs
        sharing a common molecule are correlated), we resample the primary observational units:
        molecules. For each iteration, a subset of molecules is sampled, and all induced pairwise
        relationships are reconstructed, correctly propagating molecular dependency.

        Args:
            n_iterations: Number of bootstrap resamples (default 100).
            subsample_size: Number of molecules sampled per cluster draw (default 1500).
            seed: Random seed for exact reproducibility.

        Returns:
            ClusterBootstrapSummary containing point estimates and 95% bootstrap confidence intervals.
        """
        rng = np.random.default_rng(seed)
        n_mols = self.engine.corpus_size
        fps = self.engine.fingerprints
        L = self.label_matrix
        counts = self.label_counts

        from rdkit import DataStructs  # type: ignore

        pearson_bootstraps = []
        spearman_bootstraps = []

        for _ in range(n_iterations):
            sampled_nodes = rng.choice(n_mols, size=subsample_size, replace=False)
            m = len(sampled_nodes)
            sub_pairs = m * (m - 1) // 2

            t_sub = np.empty(sub_pairs, dtype=np.float32)
            j_sub = np.empty(sub_pairs, dtype=np.float32)

            sub_fps = [fps[idx] for idx in sampled_nodes]
            sub_L = L[sampled_nodes]
            sub_counts = counts[sampled_nodes]

            off = 0
            for i in range(m - 1):
                n_r = m - 1 - i
                end_off = off + n_r
                sims = DataStructs.BulkTanimotoSimilarity(sub_fps[i], sub_fps[i + 1 :])
                t_sub[off:end_off] = sims

                inter = sub_L[i + 1 :].dot(sub_L[i]).astype(np.float32)
                union = (sub_counts[i + 1 :] + sub_counts[i] - inter).astype(np.float32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    j_sub[off:end_off] = np.where(union > 0, inter / union, 0.0)

                off = end_off

            r = float(np.corrcoef(t_sub, j_sub)[0, 1])
            rho, _ = stats.spearmanr(t_sub, j_sub)
            pearson_bootstraps.append(r)
            spearman_bootstraps.append(float(rho))

        p_point = float(np.mean(pearson_bootstraps))
        s_point = float(np.mean(spearman_bootstraps))

        p_ci = (float(np.percentile(pearson_bootstraps, 2.5)), float(np.percentile(pearson_bootstraps, 97.5)))
        s_ci = (float(np.percentile(spearman_bootstraps, 2.5)), float(np.percentile(spearman_bootstraps, 97.5)))

        return ClusterBootstrapSummary(
            resampling_unit="molecule",
            n_iterations=n_iterations,
            sample_size_molecules=subsample_size,
            pearson_r_point_estimate=round(p_point, 4),
            pearson_r_bootstrap_mean=round(float(np.mean(pearson_bootstraps)), 4),
            pearson_r_bootstrap_se=round(float(np.std(pearson_bootstraps)), 4),
            pearson_r_ci_95=(round(p_ci[0], 4), round(p_ci[1], 4)),
            spearman_rho_point_estimate=round(s_point, 4),
            spearman_rho_bootstrap_mean=round(float(np.mean(spearman_bootstraps)), 4),
            spearman_rho_bootstrap_se=round(float(np.std(spearman_bootstraps)), 4),
            spearman_rho_ci_95=(round(s_ci[0], 4), round(s_ci[1], 4)),
            methodological_note=(
                "Molecule-level cluster bootstrap resamples primary molecular units to preserve dyadic network dependencies. "
                "Pairs are reconstructed from the resampled subgraphs rather than assuming independent pair observations."
            ),
        )

    def run_full_neighborhood_study(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        n_bootstrap: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Execute end-to-end neighborhood analysis and export experimental artifacts.

        Artifacts saved:
        - pairwise_correlation_summary.json
        - tanimoto_vs_odor_bins.csv
        - neighborhood_k_summary.json
        - bootstrap_ci_summary.json
        """
        out_path = Path(output_dir) if output_dir else Path("backend/experiments/phase5/neighborhoods")
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Global pairwise computation
        tanimoto_arr, jaccard_arr, _, _ = self.compute_all_pairwise_similarities()
        global_summary = self.compute_global_pairwise_analysis(
            tanimoto_arr=tanimoto_arr, jaccard_arr=jaccard_arr
        )

        # 2. Molecule-level cluster bootstrap
        boot_summary = self.compute_molecule_cluster_bootstrap(
            n_iterations=n_bootstrap, seed=seed
        )
        global_summary.cluster_bootstrap = boot_summary

        # 3. Neighborhood summaries across K for representative molecules
        representative_mols = [
            "MOL_00001",
            "MOL_00010",
            "MOL_00050",
            "MOL_00100",
            "MOL_00500",
            "MOL_01000",
            "MOL_02000",
            "MOL_03000",
            "MOL_04000",
            "MOL_04983",
        ]
        k_results = {}
        for m_id in representative_mols:
            k_results[m_id] = {
                str(k): asdict(self.analyze_neighborhood(query=m_id, k=k))
                for k in [5, 10, 20, 50]
            }

        # 4. Save JSON and CSV artifacts
        corr_summary = {
            "total_molecules": global_summary.total_molecules,
            "total_unique_pairs": global_summary.total_unique_pairs,
            "pearson_correlation": global_summary.pearson_correlation,
            "spearman_correlation": global_summary.spearman_correlation,
            "mean_tanimoto": global_summary.mean_tanimoto,
            "median_tanimoto": global_summary.median_tanimoto,
            "mean_odor_jaccard": global_summary.mean_odor_jaccard,
            "median_odor_jaccard": global_summary.median_odor_jaccard,
            "zero_odor_overlap_total_pairs": global_summary.zero_odor_overlap_total_pairs,
            "zero_odor_overlap_overall_proportion": global_summary.zero_odor_overlap_overall_proportion,
        }
        with open(out_path / "pairwise_correlation_summary.json", "w", encoding="utf-8") as f:
            json.dump(corr_summary, f, indent=2)

        # Save binned data to CSV
        bins_data = [asdict(b) for b in global_summary.binned_analysis]
        pd.DataFrame(bins_data).to_csv(out_path / "tanimoto_vs_odor_bins.csv", index=False)

        # Save bootstrap summary
        with open(out_path / "bootstrap_ci_summary.json", "w", encoding="utf-8") as f:
            json.dump(asdict(boot_summary), f, indent=2)

        # Save representative K results
        with open(out_path / "neighborhood_k_summary.json", "w", encoding="utf-8") as f:
            json.dump(k_results, f, indent=2)

        return {
            "global_summary": asdict(global_summary),
            "bootstrap_summary": asdict(boot_summary),
            "output_directory": str(out_path),
        }

