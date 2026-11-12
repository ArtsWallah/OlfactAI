"""Unit tests for statistical odor-descriptor enrichment analysis."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from backend.app.clustering.odor_enrichment import (
    ContingencyTable,
    OdorEnrichmentAnalyzer,
    apply_benjamini_hochberg,
    build_contingency_table,
    compute_odds_ratio_and_ci,
    run_single_enrichment_test,
)


class TestContingencyTableAndFisher:
    """Test 2x2 table construction and exact test calculations."""

    def test_build_contingency_table_counts(self):
        # 10 molecules: 4 in cluster 0, 4 in cluster 1, 2 noise (-1)
        cluster_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1])
        # Descriptors:
        # Cluster 0: 3 pos, 1 neg
        # Cluster 1: 1 pos, 3 neg
        # Noise: 2 pos, 0 neg
        descriptor_values = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 1])

        # Test cluster 0 with default reference (noise excluded)
        table_0 = build_contingency_table(cluster_labels, descriptor_values, target_cluster=0)
        assert table_0.a == 3
        assert table_0.b == 1
        assert table_0.c == 1  # Only from cluster 1, noise excluded
        assert table_0.d == 3  # Only from cluster 1, noise excluded
        assert table_0.n_cluster == 4
        assert table_0.n_background == 4
        assert table_0.n_total == 8

        # Test cluster 1
        table_1 = build_contingency_table(cluster_labels, descriptor_values, target_cluster=1)
        assert table_1.a == 1
        assert table_1.b == 3
        assert table_1.c == 3
        assert table_1.d == 1

    def test_build_contingency_table_with_custom_mask(self):
        cluster_labels = np.array([0, 0, 1, 1])
        descriptor_values = np.array([1, 0, 1, 0])
        # Mask only first 3 molecules
        mask = np.array([True, True, True, False])
        table = build_contingency_table(
            cluster_labels, descriptor_values, target_cluster=0, reference_mask=mask
        )
        assert table.a == 1
        assert table.b == 1
        assert table.c == 1
        assert table.d == 0

    def test_build_contingency_table_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            build_contingency_table(np.array([0, 1]), np.array([0, 1, 0]), target_cluster=0)

    def test_odds_ratio_and_ci_standard(self):
        a, b, c, d = 20, 10, 10, 30
        raw_or, ci_l, ci_u, ci_m, w_ci_l, w_ci_u, log2_or = compute_odds_ratio_and_ci(a, b, c, d)
        expected_or = (20 * 30) / (10 * 10)  # 6.0
        assert np.isclose(raw_or, expected_or)
        assert ci_m == "exact_conditional"
        assert not np.isnan(ci_l)
        assert not np.isnan(ci_u)
        assert ci_l < raw_or < ci_u
        assert not np.isnan(w_ci_l)
        assert not np.isnan(w_ci_u)
        assert log2_or > 0

    def test_odds_ratio_zero_cells(self):
        # Zero in cell b -> raw OR = inf
        a, b, c, d = 10, 0, 5, 20
        raw_or, ci_l, ci_u, ci_m, w_ci_l, w_ci_u, log2_or = compute_odds_ratio_and_ci(a, b, c, d)
        assert np.isinf(raw_or)
        assert ci_m == "exact_conditional"
        assert np.isfinite(ci_l)  # Exact conditional lower bound is finite
        assert np.isinf(ci_u)     # Exact conditional upper bound is inf
        # Woolf intervals should be NaN on zero cells
        assert np.isnan(w_ci_l)
        assert np.isnan(w_ci_u)
        # Haldane log2 OR should be finite and positive
        assert np.isfinite(log2_or)
        assert log2_or > 0

        # Zero in cell a -> raw OR = 0
        a, b, c, d = 0, 10, 5, 20
        raw_or, ci_l, ci_u, ci_m, w_ci_l, w_ci_u, log2_or = compute_odds_ratio_and_ci(a, b, c, d)
        assert raw_or == 0.0
        assert ci_m == "exact_conditional"
        assert ci_l == 0.0        # Exact conditional lower bound is 0
        assert np.isfinite(ci_u)  # Exact conditional upper bound is finite
        assert np.isnan(w_ci_l)
        assert np.isnan(w_ci_u)
        # Haldane log2 OR should be finite and negative
        assert np.isfinite(log2_or)
        assert log2_or < 0

    def test_odds_ratio_all_zeros_corner_case(self):
        # Degenerate table where both numerator and denominator product are 0
        a, b, c, d = 0, 0, 5, 20
        raw_or, ci_l, ci_u, ci_m, w_ci_l, w_ci_u, log2_or = compute_odds_ratio_and_ci(a, b, c, d)
        assert np.isnan(raw_or)
        assert np.isnan(w_ci_l)
        assert np.isnan(w_ci_u)


class TestMultipleTestingCorrection:
    """Test Benjamini-Hochberg FDR control across distinct families."""

    def test_benjamini_hochberg_known_values(self):
        raw_p = np.array([0.01, 0.04, 0.03, 0.20])
        q_vals = apply_benjamini_hochberg(raw_p)
        assert np.isclose(q_vals[0], 0.04)
        assert np.isclose(q_vals[1], 0.04 * 4 / 3)
        assert np.isclose(q_vals[2], 0.04 * 4 / 3)
        assert np.isclose(q_vals[3], 0.20)

    def test_benjamini_hochberg_empty(self):
        assert len(apply_benjamini_hochberg(np.array([]))) == 0

    def test_benjamini_hochberg_all_ones(self):
        raw_p = np.array([1.0, 1.0, 1.0])
        q_vals = apply_benjamini_hochberg(raw_p)
        assert np.all(q_vals == 1.0)

    def test_separate_family_fdr_independence(self):
        # Verify that running FDR on Family A vs Family B keeps families strictly separate
        fam_a_p = np.array([0.001, 0.002, 0.01])
        fam_b_p = np.array([0.001, 0.5, 0.8, 0.9])

        q_a = apply_benjamini_hochberg(fam_a_p)
        q_b = apply_benjamini_hochberg(fam_b_p)

        # In family A (size 3), p=0.001 -> q = 0.001 * 3 / 1 = 0.003
        assert np.isclose(q_a[0], 0.003)
        # In family B (size 4), p=0.001 -> q = 0.001 * 4 / 1 = 0.004
        assert np.isclose(q_b[0], 0.004)


class TestOdorEnrichmentAnalyzer:
    """Test full enrichment workflow on synthetic data."""

    def test_synthetic_enrichment_and_depletion(self):
        np.random.seed(42)
        n_mol = 110
        clusters = np.array([0] * 30 + [1] * 30 + [2] * 30 + [-1] * 20)

        # Odor A: enriched in cluster 0
        odor_a = np.zeros(n_mol, dtype=int)
        odor_a[:27] = 1
        odor_a[30:33] = 1
        odor_a[60:63] = 1

        # Odor B: depleted in cluster 0
        odor_b = np.zeros(n_mol, dtype=int)
        odor_b[30:45] = 1
        odor_b[60:75] = 1

        # Odor C: uniform random
        odor_c = np.random.binomial(1, 0.2, n_mol)

        df_odor = pd.DataFrame(
            {
                "molecule_id": [f"MOL_{i:03d}" for i in range(n_mol)],
                "canonical_smiles": ["CC"] * n_mol,
                "odor_A": odor_a,
                "odor_B": odor_b,
                "odor_C": odor_c,
            }
        )

        analyzer = OdorEnrichmentAnalyzer(fdr_threshold=0.05)
        res = analyzer.analyze_clusters(clusters, df_odor)

        assert len(res) == 3 * 3  # 3 clusters * 3 descriptors = 9 tests
        assert "q_value" in res.columns
        assert "ci_lower" in res.columns
        assert "ci_upper" in res.columns
        assert "ci_method" in res.columns
        assert "woolf_ci_lower" in res.columns
        assert "woolf_ci_upper" in res.columns
        assert "direction" in res.columns

        # Verify Odor A in Cluster 0 is significantly enriched
        row_c0_a = res[(res["cluster_id"] == 0) & (res["descriptor"] == "odor_A")].iloc[0]
        assert row_c0_a["is_significant"]
        assert row_c0_a["direction"] == "enriched"
        assert row_c0_a["cluster_prevalence"] == 27 / 30
        assert row_c0_a["q_value"] < 0.05
        assert row_c0_a["ci_lower"] > 1.0

        # Verify Odor B in Cluster 0 is significantly depleted
        row_c0_b = res[(res["cluster_id"] == 0) & (res["descriptor"] == "odor_B")].iloc[0]
        assert row_c0_b["is_significant"]
        assert row_c0_b["direction"] == "depleted"
        assert row_c0_b["cluster_prevalence"] == 0.0
        assert row_c0_b["q_value"] < 0.05
        assert row_c0_b["ci_lower"] == 0.0
        assert np.isnan(row_c0_b["woolf_ci_lower"])

    def test_clustered_vs_noise_analysis(self):
        clusters = np.array([0] * 20 + [1] * 20 + [-1] * 30)
        odor_1 = np.array([1] * 30 + [0] * 10 + [0] * 25 + [1] * 5)
        odor_2 = np.array([0] * 35 + [1] * 5 + [1] * 25 + [0] * 5)

        df_odor = pd.DataFrame({"odor_1": odor_1, "odor_2": odor_2})
        analyzer = OdorEnrichmentAnalyzer(fdr_threshold=0.05)
        comp = analyzer.analyze_clustered_vs_noise(clusters, df_odor)

        assert len(comp) == 2
        assert "prevalence_clustered" in comp.columns
        assert "prevalence_noise" in comp.columns
        assert "ci_method" in comp.columns
        assert comp.loc[comp["descriptor"] == "odor_1", "direction"].values[0] in [
            "higher_in_clustered",
            "neutral",
        ]
