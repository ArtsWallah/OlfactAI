"""Statistical Odor-Descriptor Enrichment Analysis for OlfactAI Molecular Clusters.

Implements observational association testing between HDBSCAN molecular clusters
and binary odor descriptors from the OpenPOM dataset.

Statistical Framework:
- 2x2 Contingency Table for cluster k and odor descriptor j:
    a: in cluster k, descriptor j = 1
    b: in cluster k, descriptor j = 0
    c: outside cluster k (other non-noise clusters), descriptor j = 1
    d: outside cluster k (other non-noise clusters), descriptor j = 0
- Fisher's Exact Test (two-sided) for exact small-count hypothesis testing.
- Odds Ratios and Confidence Intervals:
    - Raw sample odds ratio: (a * d) / (b * c).
    - Exact conditional odds ratio and exact 95% conditional confidence intervals
      computed via `scipy.stats.contingency.odds_ratio` (Fisher's exact non-central hypergeometric distribution),
      valid even when individual cells are 0 or counts are small.
    - Approximate Woolf 95% log-linear CI maintained as a documented diagnostic,
      flagged as NaN for zero-count cells.
- Haldane-Anscombe corrected log2 Odds Ratio for stable visualization and ranking:
    log2_OR = log2(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))
- Benjamini-Hochberg (BH) False Discovery Rate (FDR) control applied across the
  entire family of hypothesis tests (e.g. 37 clusters * 138 descriptors = 5,106 tests).
- Separate comparison of Clustered vs. Noise molecule populations (138 tests).
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
class ContingencyTable:
    """2x2 contingency table counts for a cluster and descriptor."""

    a: int  # in cluster, descriptor = 1
    b: int  # in cluster, descriptor = 0
    c: int  # background (reference), descriptor = 1
    d: int  # background (reference), descriptor = 0

    @property
    def n_cluster(self) -> int:
        return self.a + self.b

    @property
    def n_background(self) -> int:
        return self.c + self.d

    @property
    def n_total(self) -> int:
        return self.a + self.b + self.c + self.d

    def as_matrix(self) -> np.ndarray:
        return np.array([[self.a, self.b], [self.c, self.d]], dtype=int)


@dataclass(frozen=True)
class EnrichmentResult:
    """Statistical outcome for a single (cluster, descriptor) pair."""

    cluster_id: int
    descriptor: str
    a_in_cluster_pos: int
    b_in_cluster_neg: int
    c_out_cluster_pos: int
    d_out_cluster_neg: int
    cluster_size: int
    background_size: int
    cluster_prevalence: float
    background_prevalence: float
    prevalence_diff: float
    odds_ratio: float
    ci_lower: float
    ci_upper: float
    ci_method: str
    woolf_ci_lower: float
    woolf_ci_upper: float
    haldane_log2_odds_ratio: float
    p_value: float
    q_value: float = 1.0
    is_significant: bool = False
    direction: str = "neutral"  # "enriched", "depleted", or "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_contingency_table(
    cluster_labels: np.ndarray,
    descriptor_values: np.ndarray,
    target_cluster: int,
    reference_mask: Optional[np.ndarray] = None,
) -> ContingencyTable:
    """Build a 2x2 contingency table for target_cluster vs reference background.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Array of integer cluster IDs for each molecule.
    descriptor_values : np.ndarray
        Binary 0/1 array for a specific odor descriptor.
    target_cluster : int
        The cluster ID being evaluated.
    reference_mask : Optional[np.ndarray], default=None
        Boolean mask of molecules included in the analysis. If None, all molecules
        with cluster_labels != -1 (non-noise) are included.

    Returns
    -------
    ContingencyTable
        Object containing counts a, b, c, d.
    """
    if len(cluster_labels) != len(descriptor_values):
        raise ValueError(
            f"Length mismatch: cluster_labels ({len(cluster_labels)}) vs "
            f"descriptor_values ({len(descriptor_values)})"
        )

    cluster_labels = np.asarray(cluster_labels)
    descriptor_values = np.asarray(descriptor_values)

    if reference_mask is None:
        # Default: only non-noise molecules form the universe
        reference_mask = cluster_labels != -1
    else:
        reference_mask = np.asarray(reference_mask, dtype=bool)

    if len(reference_mask) != len(cluster_labels):
        raise ValueError("reference_mask length must match cluster_labels length.")

    in_target = (cluster_labels == target_cluster) & reference_mask
    in_bg = (cluster_labels != target_cluster) & reference_mask

    pos = descriptor_values == 1
    neg = descriptor_values == 0

    a = int(np.sum(in_target & pos))
    b = int(np.sum(in_target & neg))
    c = int(np.sum(in_bg & pos))
    d = int(np.sum(in_bg & neg))

    return ContingencyTable(a=a, b=b, c=c, d=d)


def compute_odds_ratio_and_ci(
    a: int, b: int, c: int, d: int, alpha: float = 0.05
) -> Tuple[float, float, float, str, float, float, float]:
    """Compute raw Odds Ratio, exact conditional CI, Woolf CI, and Haldane-Anscombe log2 OR.

    Returns
    -------
    Tuple[float, float, float, str, float, float, float]
        (raw_or, exact_ci_lower, exact_ci_upper, ci_method, woolf_ci_lower, woolf_ci_upper, haldane_log2_or)
    """
    # 1. Raw Sample Odds Ratio
    if b * c == 0:
        if a * d == 0:
            raw_or = np.nan
        else:
            raw_or = np.inf
    else:
        raw_or = (a * d) / (b * c)

    # 2. Exact Conditional Odds Ratio and 95% Confidence Interval via SciPy
    try:
        scipy_or = stats.contingency.odds_ratio([[a, b], [c, d]], kind="conditional")
        ci = scipy_or.confidence_interval(confidence_level=1.0 - alpha)
        exact_ci_l = float(ci.low)
        exact_ci_u = float(ci.high)
        ci_method = "exact_conditional"
    except Exception:
        exact_ci_l = np.nan
        exact_ci_u = np.nan
        ci_method = "undefined"

    # 3. Approximate Woolf 95% CI (asymptotic log-normal, undefined for zero cells)
    if a == 0 or b == 0 or c == 0 or d == 0:
        woolf_ci_l = np.nan
        woolf_ci_u = np.nan
    else:
        se = np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        ln_or = np.log(raw_or)
        woolf_ci_l = float(np.exp(ln_or - z * se))
        woolf_ci_u = float(np.exp(ln_or + z * se))

    # 4. Haldane-Anscombe corrected log2 Odds Ratio (+0.5 to all cells)
    a_h = a + 0.5
    b_h = b + 0.5
    c_h = c + 0.5
    d_h = d + 0.5
    haldane_or = (a_h * d_h) / (b_h * c_h)
    haldane_log2_or = float(np.log2(haldane_or))

    return (
        float(raw_or),
        exact_ci_l,
        exact_ci_u,
        ci_method,
        woolf_ci_l,
        woolf_ci_u,
        haldane_log2_or,
    )


def apply_benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Apply the Benjamini-Hochberg (BH) procedure for False Discovery Rate (FDR).

    Controls FDR at alpha across the complete set of tests.

    Parameters
    ----------
    p_values : np.ndarray
        1D array of p-values in [0, 1].

    Returns
    -------
    np.ndarray
        Array of adjusted p-values (q-values), bounded in [0, 1] and monotonic.
    """
    p_vals = np.asarray(p_values, dtype=float)
    n = len(p_vals)
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(p_vals)
    sorted_p = p_vals[order]

    # Adjusted p-value: p * n / rank
    ranks = np.arange(1, n + 1)
    q_vals_sorted = sorted_p * n / ranks

    # Enforce monotonicity from right to left: q[i] = min(q[i], q[i+1])
    q_vals_sorted = np.minimum.accumulate(q_vals_sorted[::-1])[::-1]
    # Bound by 1.0
    q_vals_sorted = np.clip(q_vals_sorted, 0.0, 1.0)

    # Reorder back to original input indexing
    q_vals = np.empty(n, dtype=float)
    q_vals[order] = q_vals_sorted
    return q_vals


def run_single_enrichment_test(
    table: ContingencyTable,
    cluster_id: int,
    descriptor: str,
) -> EnrichmentResult:
    """Execute Fisher's exact test and compute odds ratios for a single table."""
    # Two-sided Fisher's Exact Test
    _, p_value = stats.fisher_exact(table.as_matrix(), alternative="two-sided")
    p_value = float(p_value)

    cluster_size = table.n_cluster
    background_size = table.n_background

    cluster_prev = table.a / cluster_size if cluster_size > 0 else 0.0
    bg_prev = table.c / background_size if background_size > 0 else 0.0
    prev_diff = cluster_prev - bg_prev

    raw_or, ci_l, ci_u, ci_m, w_ci_l, w_ci_u, haldane_log2_or = compute_odds_ratio_and_ci(
        table.a, table.b, table.c, table.d
    )

    return EnrichmentResult(
        cluster_id=cluster_id,
        descriptor=descriptor,
        a_in_cluster_pos=table.a,
        b_in_cluster_neg=table.b,
        c_out_cluster_pos=table.c,
        d_out_cluster_neg=table.d,
        cluster_size=cluster_size,
        background_size=background_size,
        cluster_prevalence=float(cluster_prev),
        background_prevalence=float(bg_prev),
        prevalence_diff=float(prev_diff),
        odds_ratio=raw_or,
        ci_lower=ci_l,
        ci_upper=ci_u,
        ci_method=ci_m,
        woolf_ci_lower=w_ci_l,
        woolf_ci_upper=w_ci_u,
        haldane_log2_odds_ratio=haldane_log2_or,
        p_value=p_value,
    )


class OdorEnrichmentAnalyzer:
    """Analyzer for running full-family odor descriptor enrichment on clusters.

    Parameters
    ----------
    fdr_threshold : float, default=0.05
        False Discovery Rate cutoff (q-value threshold) for statistical significance.
    """

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def analyze_clusters(
        self,
        cluster_labels: np.ndarray,
        odor_df: pd.DataFrame,
        descriptor_columns: Optional[List[str]] = None,
        reference_mask: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Run two-sided Fisher's exact test and Benjamini-Hochberg FDR correction.

        Parameters
        ----------
        cluster_labels : np.ndarray
            Array of cluster assignments for each molecule (shape N,).
        odor_df : pd.DataFrame
            DataFrame containing binary odor descriptor columns (shape N, D).
        descriptor_columns : Optional[List[str]], default=None
            List of descriptor column names to test. If None, all binary columns
            excluding metadata columns ('molecule_id', 'canonical_smiles') are used.
        reference_mask : Optional[np.ndarray], default=None
            Boolean mask for molecules in the primary cluster universe.
            If None, molecules with cluster_labels != -1 (non-noise) are used.

        Returns
        -------
        pd.DataFrame
            DataFrame with all tested (cluster, descriptor) pairs, raw and adjusted
            p-values, odds ratios, exact and approximate CIs, prevalence, and classification.
        """
        cluster_labels = np.asarray(cluster_labels)
        if len(cluster_labels) != len(odor_df):
            raise ValueError(
                f"Cluster labels length ({len(cluster_labels)}) != "
                f"odor_df length ({len(odor_df)})"
            )

        if descriptor_columns is None:
            metadata_cols = {"molecule_id", "canonical_smiles", "cluster_hdbscan"}
            descriptor_columns = [c for c in odor_df.columns if c not in metadata_cols]

        if reference_mask is None:
            # Default: non-noise clusters only
            reference_mask = cluster_labels != -1
        else:
            reference_mask = np.asarray(reference_mask, dtype=bool)

        unique_clusters = sorted(
            [int(c) for c in np.unique(cluster_labels[reference_mask]) if c != -1]
        )

        logger.info(
            f"Running enrichment analysis for {len(unique_clusters)} clusters across "
            f"{len(descriptor_columns)} descriptors ({len(unique_clusters) * len(descriptor_columns)} tests)."
        )

        results: List[EnrichmentResult] = []

        for cid in unique_clusters:
            for desc in descriptor_columns:
                desc_vals = odor_df[desc].to_numpy()
                table = build_contingency_table(
                    cluster_labels=cluster_labels,
                    descriptor_values=desc_vals,
                    target_cluster=cid,
                    reference_mask=reference_mask,
                )
                res = run_single_enrichment_test(
                    table=table,
                    cluster_id=cid,
                    descriptor=desc,
                )
                results.append(res)

        if not results:
            return pd.DataFrame()

        # Apply Benjamini-Hochberg across the ENTIRE family of hypothesis tests
        p_values = np.array([r.p_value for r in results], dtype=float)
        q_values = apply_benjamini_hochberg(p_values)

        final_rows = []
        for res, q_val in zip(results, q_values):
            d = res.to_dict()
            d["q_value"] = float(q_val)
            is_sig = bool(q_val < self.fdr_threshold)
            d["is_significant"] = is_sig

            if is_sig:
                if res.haldane_log2_odds_ratio > 0:
                    d["direction"] = "enriched"
                else:
                    d["direction"] = "depleted"
            else:
                d["direction"] = "neutral"

            final_rows.append(d)

        df_results = pd.DataFrame(final_rows)
        return df_results

    def analyze_clustered_vs_noise(
        self,
        cluster_labels: np.ndarray,
        odor_df: pd.DataFrame,
        descriptor_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare odor descriptor prevalence between clustered (non-noise) vs noise molecules.

        Applies separate Benjamini-Hochberg FDR control across its own hypothesis family (D tests).

        Parameters
        ----------
        cluster_labels : np.ndarray
            Array of cluster labels (shape N,).
        odor_df : pd.DataFrame
            DataFrame of odor descriptors (shape N, D).
        descriptor_columns : Optional[List[str]], default=None
            Descriptor columns to analyze.

        Returns
        -------
        pd.DataFrame
            Comparative statistics (clustered vs noise) with Fisher tests and separate FDR.
        """
        cluster_labels = np.asarray(cluster_labels)
        if descriptor_columns is None:
            metadata_cols = {"molecule_id", "canonical_smiles", "cluster_hdbscan"}
            descriptor_columns = [c for c in odor_df.columns if c not in metadata_cols]

        is_clustered = cluster_labels != -1
        n_clustered = int(np.sum(is_clustered))
        n_noise = int(np.sum(~is_clustered))

        results = []
        for desc in descriptor_columns:
            desc_vals = odor_df[desc].to_numpy()

            a = int(np.sum(is_clustered & (desc_vals == 1)))
            b = int(np.sum(is_clustered & (desc_vals == 0)))
            c = int(np.sum((~is_clustered) & (desc_vals == 1)))
            d = int(np.sum((~is_clustered) & (desc_vals == 0)))

            _, p_val = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
            raw_or, ci_l, ci_u, ci_m, w_ci_l, w_ci_u, log2_or = compute_odds_ratio_and_ci(
                a, b, c, d
            )

            p_clustered = a / n_clustered if n_clustered > 0 else 0.0
            p_noise = c / n_noise if n_noise > 0 else 0.0

            results.append(
                {
                    "descriptor": desc,
                    "clustered_pos": a,
                    "clustered_neg": b,
                    "noise_pos": c,
                    "noise_neg": d,
                    "n_clustered": n_clustered,
                    "n_noise": n_noise,
                    "prevalence_clustered": float(p_clustered),
                    "prevalence_noise": float(p_noise),
                    "prevalence_diff": float(p_clustered - p_noise),
                    "odds_ratio": raw_or,
                    "ci_lower": ci_l,
                    "ci_upper": ci_u,
                    "ci_method": ci_m,
                    "woolf_ci_lower": w_ci_l,
                    "woolf_ci_upper": w_ci_u,
                    "haldane_log2_odds_ratio": log2_or,
                    "p_value": float(p_val),
                }
            )

        df = pd.DataFrame(results)
        # Separate BH FDR control for the global clustered-vs-noise comparison family
        df["q_value"] = apply_benjamini_hochberg(df["p_value"].to_numpy())
        df["is_significant"] = df["q_value"] < self.fdr_threshold
        df["direction"] = np.where(
            ~df["is_significant"],
            "neutral",
            np.where(df["haldane_log2_odds_ratio"] > 0, "higher_in_clustered", "higher_in_noise"),
        )
        return df.sort_values(by="q_value")
