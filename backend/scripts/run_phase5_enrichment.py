"""Execution script for Phase 5, Step 5: Statistical Odor-Descriptor Enrichment Analysis.

Analyzes observational associations between the 37 primary HDBSCAN molecular clusters
and 138 binary odor descriptors in the OpenPOM dataset.

Produces:
- Full cluster-odor enrichment statistics (CSV, JSON) with exact and approximate CIs
- Clustered vs Noise global comparison (CSV) with separate family-wise FDR
- Cluster-odor prevalence matrix (CSV)
- Significant enrichments and depletions tables (CSV)
- Metadata with run duration, memory, parameters, and hypothesis family counts (JSON)
- 5 Publication-quality plots (300 DPI):
  1. cluster_odor_log2_odds_ratio_heatmap.png
  2. significant_enrichment_overview.png
  3. significant_depletion_overview.png
  4. global_clustered_vs_noise_prevalence.png
  5. cluster_level_prevalence_summary.png
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from backend.app.clustering.odor_enrichment import OdorEnrichmentAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase5_enrichment")

# Set paths
REPO_ROOT = Path(__file__).resolve().parents[2]
CLUSTER_PATH = REPO_ROOT / "backend/experiments/phase5/clustering/hdbscan_fingerprints_primary.csv"
ODOR_PATH = REPO_ROOT / "backend/datasets/processed/odor_labels.csv"
MOL_PATH = REPO_ROOT / "backend/datasets/processed/molecules.csv"

OUT_DIR = REPO_ROOT / "backend/experiments/phase5/enrichment"
PLOT_DIR = REPO_ROOT / "backend/experiments/phase5/plots"


def main():
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading cluster assignments and odor labels...")
    hdb_df = pd.read_csv(CLUSTER_PATH)
    odor_df = pd.read_csv(ODOR_PATH)
    mol_df = pd.read_csv(MOL_PATH)

    assert len(hdb_df) == len(odor_df) == 4983, "Dataset length mismatch!"
    assert (hdb_df["molecule_id"] == odor_df["molecule_id"]).all(), "Molecule ID alignment mismatch!"
    assert (hdb_df["molecule_id"] == mol_df["molecule_id"]).all(), "Molecule ID alignment mismatch with molecules.csv!"

    cluster_labels = hdb_df["cluster_hdbscan"].to_numpy()
    non_noise_mask = cluster_labels != -1

    unique_clusters = sorted([int(c) for c in np.unique(cluster_labels[non_noise_mask])])
    n_clustered = int(np.sum(non_noise_mask))
    n_noise = int(np.sum(~non_noise_mask))

    metadata_cols = {"molecule_id", "canonical_smiles", "cluster_hdbscan"}
    descriptor_cols = [c for c in odor_df.columns if c not in metadata_cols]

    logger.info(
        f"Dataset loaded: {len(hdb_df)} molecules ({n_clustered} clustered across {len(unique_clusters)} clusters, "
        f"{n_noise} noise). Descriptors: {len(descriptor_cols)}."
    )

    analyzer = OdorEnrichmentAnalyzer(fdr_threshold=0.05)

    # 1. Primary Cluster-Odor Enrichment Analysis (Family A: 37 clusters x 138 descriptors = 5,106 hypotheses)
    logger.info("Running Fisher exact tests and Benjamini-Hochberg FDR across Family A (5,106 hypotheses)...")
    enrichment_df = analyzer.analyze_clusters(
        cluster_labels=cluster_labels,
        odor_df=odor_df,
        descriptor_columns=descriptor_cols,
        reference_mask=non_noise_mask,
    )

    # Sort by q_value ascending, then prevalence_diff descending
    enrichment_df = enrichment_df.sort_values(by=["q_value", "prevalence_diff"], ascending=[True, False])

    # Save full enrichment results
    enrichment_csv = OUT_DIR / "cluster_odor_enrichment.csv"
    enrichment_df.to_csv(enrichment_csv, index=False)
    logger.info(f"Saved full enrichment results: {enrichment_csv} ({len(enrichment_df)} tests)")

    enrichment_json = OUT_DIR / "cluster_odor_enrichment.json"
    enrichment_df.to_json(enrichment_json, orient="records", indent=2)

    # Filter significant enrichments and depletions
    sig_enrichments = enrichment_df[enrichment_df["direction"] == "enriched"]
    sig_depletions = enrichment_df[enrichment_df["direction"] == "depleted"]

    sig_enrich_csv = OUT_DIR / "significant_enrichments.csv"
    sig_deplete_csv = OUT_DIR / "significant_depletions.csv"
    sig_enrichments.to_csv(sig_enrich_csv, index=False)
    sig_depletions.to_csv(sig_deplete_csv, index=False)
    logger.info(f"Significant enrichments (q < 0.05, OR > 1): {len(sig_enrichments)}")
    logger.info(f"Significant depletions (q < 0.05, OR < 1): {len(sig_depletions)}")

    # 2. Build cluster-odor prevalence matrix
    logger.info("Building cluster-odor prevalence matrix...")
    prevalence_records = []
    for cid in unique_clusters:
        c_mask = cluster_labels == cid
        c_size = int(np.sum(c_mask))
        row = {"cluster_id": cid, "cluster_size": c_size}
        for desc in descriptor_cols:
            row[desc] = float(np.mean(odor_df.loc[c_mask, desc]))
        prevalence_records.append(row)

    # Add background prevalence row (all clustered)
    bg_row = {"cluster_id": "background_all_clustered", "cluster_size": n_clustered}
    for desc in descriptor_cols:
        bg_row[desc] = float(np.mean(odor_df.loc[non_noise_mask, desc]))
    prevalence_records.append(bg_row)

    # Add noise prevalence row
    noise_row = {"cluster_id": "noise_unclustered", "cluster_size": n_noise}
    for desc in descriptor_cols:
        noise_row[desc] = float(np.mean(odor_df.loc[~non_noise_mask, desc]))
    prevalence_records.append(noise_row)

    prevalence_df = pd.DataFrame(prevalence_records)
    prevalence_csv = OUT_DIR / "cluster_odor_prevalence.csv"
    prevalence_df.to_csv(prevalence_csv, index=False)
    logger.info(f"Saved cluster prevalence table: {prevalence_csv}")

    # 3. Global Clustered vs Noise Comparison (Family B: 138 descriptor hypotheses)
    logger.info("Analyzing Family B: global clustered vs noise populations (138 hypotheses)...")
    clustered_vs_noise_df = analyzer.analyze_clustered_vs_noise(
        cluster_labels=cluster_labels,
        odor_df=odor_df,
        descriptor_columns=descriptor_cols,
    )
    cvn_csv = OUT_DIR / "global_clustered_vs_noise.csv"
    clustered_vs_noise_df.to_csv(cvn_csv, index=False)
    logger.info(f"Saved clustered vs noise comparison: {cvn_csv}")

    # 4. Generate Publication-Quality Plots
    logger.info("Generating publication-quality visualization figures (300 DPI)...")

    # Plot 1: Cluster-Odor Log2 Odds Ratio Heatmap
    top_descriptors = enrichment_df[enrichment_df["is_significant"]]["descriptor"].unique()
    if len(top_descriptors) == 0:
        top_descriptors = descriptor_cols[:25]
    elif len(top_descriptors) > 35:
        top_descriptors = (
            enrichment_df[enrichment_df["is_significant"]]
            .groupby("descriptor")["q_value"]
            .min()
            .sort_values()
            .head(35)
            .index.tolist()
        )

    pivot_or = enrichment_df[enrichment_df["descriptor"].isin(top_descriptors)].pivot(
        index="cluster_id", columns="descriptor", values="haldane_log2_odds_ratio"
    )

    plt.figure(figsize=(18, 12))
    sns.heatmap(
        pivot_or,
        cmap="coolwarm",
        center=0.0,
        cbar_kws={"label": "Haldane-Anscombe Log2 Odds Ratio"},
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title(
        "Cluster-Odor Association Landscape (Haldane-Anscombe Log2 Odds Ratio)\n"
        "Top Significant Descriptors across 37 Molecular Clusters (OpenPOM HDBSCAN)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Odor Descriptor", fontsize=12, labelpad=10)
    plt.ylabel("HDBSCAN Cluster ID", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plot1_path = PLOT_DIR / "cluster_odor_log2_odds_ratio_heatmap.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    logger.info(f"Saved Plot 1: {plot1_path}")

    # Plot 2: Top Significant Enrichments Overview
    top_enrich = sig_enrichments.head(25)
    plt.figure(figsize=(14, 8))
    labels = [f"Cluster {row.cluster_id}: {row.descriptor}" for _, row in top_enrich.iterrows()]
    y_pos = np.arange(len(top_enrich))
    bars = plt.barh(y_pos, top_enrich["haldane_log2_odds_ratio"], color="#2ca02c", alpha=0.85, edgecolor="black")
    plt.yticks(y_pos, labels, fontsize=10)
    plt.gca().invert_yaxis()
    plt.xlabel("Haldane-Anscombe Log2 Odds Ratio", fontsize=12, labelpad=10)
    plt.title(
        "Top 25 Statistically Significant Odor Enrichments in Molecular Clusters\n"
        "(FDR q < 0.05, Two-sided Fisher's Exact Test vs Background Non-Noise Clusters)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    for i, (bar, q_val) in enumerate(zip(bars, top_enrich["q_value"])):
        plt.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"q={q_val:.1e}",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot2_path = PLOT_DIR / "significant_enrichment_overview.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    logger.info(f"Saved Plot 2: {plot2_path}")

    # Plot 3: Top Significant Depletions Overview
    top_deplete = sig_depletions.head(25)
    plt.figure(figsize=(14, 8))
    if len(top_deplete) > 0:
        labels_dep = [f"Cluster {row.cluster_id}: {row.descriptor}" for _, row in top_deplete.iterrows()]
        y_pos_dep = np.arange(len(top_deplete))
        bars_dep = plt.barh(
            y_pos_dep, top_deplete["haldane_log2_odds_ratio"], color="#d62728", alpha=0.85, edgecolor="black"
        )
        plt.yticks(y_pos_dep, labels_dep, fontsize=10)
        plt.gca().invert_yaxis()
        plt.xlabel("Haldane-Anscombe Log2 Odds Ratio", fontsize=12, labelpad=10)
        plt.title(
            "Top Statistically Significant Odor Depletions in Molecular Clusters\n"
            "(FDR q < 0.05, Two-sided Fisher's Exact Test vs Background Non-Noise Clusters)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        for i, (bar, q_val) in enumerate(zip(bars_dep, top_deplete["q_value"])):
            plt.text(
                bar.get_width() - 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"q={q_val:.1e}",
                va="center",
                ha="right",
                fontsize=8,
                color="#333333",
            )
        plt.grid(axis="x", linestyle="--", alpha=0.6)
    else:
        plt.text(
            0.5,
            0.5,
            "No odor descriptors met depletion significance threshold (q < 0.05)\n"
            "due to baseline annotation sparsity in small background partitions.",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.title("Odor Descriptor Depletion Analysis (No Significant Depletions)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot3_path = PLOT_DIR / "significant_depletion_overview.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    logger.info(f"Saved Plot 3: {plot3_path}")

    # Plot 4: Global Clustered vs Noise Prevalence
    top_diff_desc = clustered_vs_noise_df.sort_values(by="prevalence_diff", key=abs, ascending=False).head(25)
    plt.figure(figsize=(14, 8))
    x = np.arange(len(top_diff_desc))
    width = 0.38
    plt.bar(
        x - width / 2,
        top_diff_desc["prevalence_clustered"] * 100,
        width,
        label=f"Clustered Molecules (N={n_clustered})",
        color="#1f77b4",
        alpha=0.85,
        edgecolor="black",
    )
    plt.bar(
        x + width / 2,
        top_diff_desc["prevalence_noise"] * 100,
        width,
        label=f"Noise Molecules (N={n_noise})",
        color="#ff7f0e",
        alpha=0.85,
        edgecolor="black",
    )
    plt.xticks(x, top_diff_desc["descriptor"], rotation=45, ha="right", fontsize=10)
    plt.ylabel("Prevalence (%)", fontsize=12, labelpad=10)
    plt.title(
        "Odor Descriptor Prevalence: Clustered vs Noise Populations\n"
        "Top 25 Descriptors by Absolute Difference in Frequency (OpenPOM Dataset)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot4_path = PLOT_DIR / "global_clustered_vs_noise_prevalence.png"
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    logger.info(f"Saved Plot 4: {plot4_path}")

    # Plot 5: Cluster-Level Prevalence Summary (Top Enriched Descriptor per Cluster)
    top_per_cluster = (
        enrichment_df[enrichment_df["is_significant"]]
        .sort_values(by="q_value")
        .groupby("cluster_id")
        .first()
        .reset_index()
    )

    plt.figure(figsize=(16, 9))
    if len(top_per_cluster) > 0:
        top_per_cluster = top_per_cluster.sort_values(by="cluster_id")
        cluster_x = [f"C{r.cluster_id}\n({r.descriptor})" for _, r in top_per_cluster.iterrows()]
        x_idx = np.arange(len(top_per_cluster))
        plt.bar(
            x_idx,
            top_per_cluster["cluster_prevalence"] * 100,
            color="#3b528b",
            alpha=0.85,
            edgecolor="black",
            label="Cluster Prevalence",
        )
        plt.scatter(
            x_idx,
            top_per_cluster["background_prevalence"] * 100,
            color="#d95f02",
            s=80,
            zorder=5,
            label="Background Prevalence",
        )
        plt.xticks(x_idx, cluster_x, rotation=45, ha="right", fontsize=9)
        plt.ylabel("Prevalence (%)", fontsize=12, labelpad=10)
        plt.title(
            f"Most Significant Enriched Descriptor for {len(top_per_cluster)}/37 Clusters\n"
            "Comparison of In-Cluster Prevalence vs Background Clustered Prevalence",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        plt.legend(fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
    else:
        plt.text(0.5, 0.5, "No clusters had statistically significant enrichments at FDR q < 0.05.", ha="center")
    plt.tight_layout()
    plot5_path = PLOT_DIR / "cluster_level_prevalence_summary.png"
    plt.savefig(plot5_path, dpi=300)
    plt.close()
    logger.info(f"Saved Plot 5: {plot5_path}")

    # 5. Save run metadata
    elapsed_time = time.time() - start_time
    metadata = {
        "dataset_size": len(hdb_df),
        "n_clustered": n_clustered,
        "n_noise": n_noise,
        "n_clusters": len(unique_clusters),
        "n_descriptors": len(descriptor_cols),
        "hypothesis_families": {
            "family_a_cluster_enrichment": {
                "description": "Cluster-specific enrichment across all non-noise clusters and descriptors",
                "n_hypotheses": len(enrichment_df),
                "fdr_method": "Benjamini-Hochberg across all 5,106 tests simultaneously",
                "fdr_threshold": 0.05,
                "n_significant_enrichments": int(len(sig_enrichments)),
                "n_significant_depletions": int(len(sig_depletions)),
                "clusters_with_at_least_one_enrichment": int(sig_enrichments["cluster_id"].nunique()),
            },
            "family_b_clustered_vs_noise": {
                "description": "Global comparison between clustered (1,110) and noise (3,873) molecule populations",
                "n_hypotheses": len(clustered_vs_noise_df),
                "fdr_method": "Benjamini-Hochberg across all 138 descriptors separately",
                "fdr_threshold": 0.05,
                "n_significant": int((clustered_vs_noise_df["q_value"] < 0.05).sum()),
            },
        },
        "odds_ratio_methods": {
            "raw_odds_ratio": "Sample odds ratio (a*d)/(b*c)",
            "exact_confidence_interval": "Exact conditional 95% CI via scipy.stats.contingency.odds_ratio",
            "approximate_confidence_interval": "Woolf asymptotic log-normal 95% CI (NaN on zero cells)",
            "haldane_anscombe_log2_or": "log2(((a+0.5)*(d+0.5))/((b+0.5)*(c+0.5)))",
        },
        "execution_time_seconds": round(elapsed_time, 2),
        "input_cluster_file": str(CLUSTER_PATH.relative_to(REPO_ROOT)),
        "input_odor_file": str(ODOR_PATH.relative_to(REPO_ROOT)),
        "plots_generated": [
            str(plot1_path.name),
            str(plot2_path.name),
            str(plot3_path.name),
            str(plot4_path.name),
            str(plot5_path.name),
        ],
    }

    meta_json = OUT_DIR / "enrichment_metadata.json"
    with open(meta_json, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved enrichment metadata: {meta_json}")

    print("\n" + "=" * 60)
    print("PHASE 5 STEP 5: ODOR ENRICHMENT ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Family A hypotheses (Clusters x Descriptors): {len(enrichment_df)} tests")
    print(f"  Significant enrichments (q < 0.05, OR > 1.0): {len(sig_enrichments)}")
    print(f"  Significant depletions (q < 0.05, OR < 1.0): {len(sig_depletions)}")
    print(f"Family B hypotheses (Clustered vs Noise): {len(clustered_vs_noise_df)} tests")
    print(f"Execution time: {elapsed_time:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
