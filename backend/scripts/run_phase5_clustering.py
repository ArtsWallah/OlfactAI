"""Execute Phase 5 HDBSCAN molecular clustering experiments and generate artifacts."""

import json
from pathlib import Path
import time
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from backend.app.clustering.hdbscan_clustering import (
    MolecularHDBSCAN,
    compare_representations,
    compute_jaccard_distance_matrix,
    estimate_distance_matrix_memory_mb,
    run_parameter_sensitivity_grid,
)
from backend.app.core.logging import get_logger

logger = get_logger("scripts.run_phase5_clustering")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "backend" / "experiments" / "phase5" / "clustering"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("OlfactAI Phase 5: HDBSCAN Molecular Structure Discovery")
    print("==================================================================")

    # 1. Load dataset features and metadata
    feat_path = root / "backend" / "datasets" / "processed" / "molecular_features.csv"
    mols_path = root / "backend" / "datasets" / "processed" / "molecules.csv"
    umap2_path = root / "backend" / "experiments" / "phase5" / "embeddings" / "umap_2d_embeddings.csv"
    umap10_path = root / "backend" / "experiments" / "phase5" / "embeddings" / "umap_10d_embeddings.csv"

    print("Loading molecular features and embeddings...")
    feat_df = pd.read_csv(feat_path)
    mols_df = pd.read_csv(mols_path)
    umap2_df = pd.read_csv(umap2_path)
    umap10_df = pd.read_csv(umap10_path)

    n_samples = len(feat_df)
    print(f"Loaded {n_samples} molecules.")

    # 2. Extract Morgan fingerprints
    fp_cols = [c for c in feat_df.columns if c.startswith("fp_")]
    X_fp = feat_df[fp_cols].values.astype(np.float32)

    # 3. Compute Jaccard distance matrix
    mem_mb = estimate_distance_matrix_memory_mb(n_samples)
    print(f"Computing pairwise Jaccard distance matrix via BLAS (Estimated memory: {mem_mb:.2f} MB)...")
    t0 = time.perf_counter()
    jaccard_dists = compute_jaccard_distance_matrix(X_fp)
    jaccard_time = time.perf_counter() - t0
    print(f"Jaccard distance matrix computed in {jaccard_time:.3f} s (shape: {jaccard_dists.shape}).")

    # 4. Primary experiment: Morgan Fingerprints with min_cluster_size=15, min_samples=5
    print("\nFitting Primary HDBSCAN Model (Morgan Fingerprints + Jaccard distance)...")
    primary_mcs = 15
    primary_ms = 5
    primary_clusterer = MolecularHDBSCAN(
        min_cluster_size=primary_mcs,
        min_samples=primary_ms,
        metric="precomputed",
    )
    primary_clusterer.fit(X=jaccard_dists)
    primary_summary = primary_clusterer.get_cluster_summary(distance_matrix=jaccard_dists)

    print(f"Primary Model Result:")
    print(f"  - Non-noise clusters: {primary_summary.n_clusters}")
    print(f"  - Noise molecules: {primary_summary.n_noise} / {n_samples} ({primary_summary.noise_fraction*100:.2f}%)")
    print(f"  - Min / Median / Max cluster size: {primary_summary.min_cluster_size_observed} / {primary_summary.median_cluster_size:.1f} / {primary_summary.max_cluster_size}")
    print(f"  - Mean membership strength: {primary_summary.mean_membership_strength:.4f} (clustered: {primary_summary.clustered_mean_membership_strength:.4f})")
    print(f"  - Silhouette score (clustered): {primary_summary.silhouette_score_clustered:.4f}")

    # Assign and save primary clusters
    primary_assigned = primary_clusterer.assign_clusters_to_dataframe(
        mols_df[["molecule_id", "canonical_smiles"]],
        cluster_col="cluster_hdbscan",
        prob_col="membership_strength",
        outlier_col="outlier_score",
    )
    primary_csv_path = out_dir / "hdbscan_fingerprints_primary.csv"
    primary_assigned.to_csv(primary_csv_path, index=False)
    print(f"Saved primary cluster assignments to: {primary_csv_path}")

    # 5. Parameter sensitivity experiments
    print("\nRunning Parameter Sensitivity Grid...")
    mcs_candidates = [5, 10, 15, 20, 25, 30]
    ms_candidates = [None, 3, 5, 10]
    grid_df, grid_metadata = run_parameter_sensitivity_grid(
        distance_matrix=jaccard_dists,
        min_cluster_size_candidates=mcs_candidates,
        min_samples_candidates=ms_candidates,
    )
    grid_csv_path = out_dir / "parameter_sensitivity_results.csv"
    grid_df.to_csv(grid_csv_path, index=False)
    with open(out_dir / "parameter_sensitivity_results.json", "w", encoding="utf-8") as f:
        json.dump(grid_metadata, f, indent=2)
    print(f"Saved parameter sensitivity results to: {grid_csv_path}")

    # 6. Representation comparison: Primary (FP) vs 10D UMAP vs 2D UMAP
    print("\nRunning Representation Comparison (Primary FP vs 10D UMAP vs 2D UMAP)...")
    u10_cols = [c for c in umap10_df.columns if c.startswith("umap_")]
    X_u10 = umap10_df[u10_cols].fillna(0.0).values.astype(np.float64)

    u2_cols = [c for c in umap2_df.columns if c.startswith("umap_")]
    X_u2 = umap2_df[u2_cols].fillna(0.0).values.astype(np.float64)

    rep_df, rep_summaries = compare_representations(
        fingerprint_distance_matrix=jaccard_dists,
        umap_10d_matrix=X_u10,
        umap_2d_matrix=X_u2,
        min_cluster_size=primary_mcs,
        min_samples=primary_ms,
    )
    rep_csv_path = out_dir / "representation_comparison.csv"
    rep_df.to_csv(rep_csv_path, index=False)

    rep_meta = {k: {attr: getattr(v, attr) for attr in dir(v) if not attr.startswith("_") and not callable(getattr(v, attr))}
                for k, v in rep_summaries.items()}
    with open(out_dir / "representation_comparison.json", "w", encoding="utf-8") as f:
        json.dump(rep_meta, f, indent=2)
    print(f"Saved representation comparison to: {rep_csv_path}")

    # Fit and save 10D and 2D assignments for analysis
    c_10d = MolecularHDBSCAN(min_cluster_size=primary_mcs, min_samples=primary_ms, metric="euclidean").fit(X_u10)
    assigned_10d = c_10d.assign_clusters_to_dataframe(mols_df[["molecule_id", "canonical_smiles"]])
    assigned_10d.to_csv(out_dir / "hdbscan_umap_10d_exploratory.csv", index=False)

    c_2d = MolecularHDBSCAN(min_cluster_size=primary_mcs, min_samples=primary_ms, metric="euclidean").fit(X_u2)
    assigned_2d = c_2d.assign_clusters_to_dataframe(mols_df[["molecule_id", "canonical_smiles"]])
    assigned_2d.to_csv(out_dir / "hdbscan_umap_2d_visualization.csv", index=False)

    # 7. Cluster persistence and size distributions
    with open(out_dir / "cluster_size_distribution.json", "w", encoding="utf-8") as f:
        json.dump(primary_summary.cluster_sizes, f, indent=2)

    with open(out_dir / "cluster_persistence.json", "w", encoding="utf-8") as f:
        json.dump(primary_summary.cluster_persistence, f, indent=2)

    # Experiment metadata
    exp_meta = {
        "dataset_size": n_samples,
        "representation_primary": "Morgan Fingerprint (radius=2, 2048-bit, chirality=True)",
        "distance_metric": "Jaccard distance (1.0 - Tanimoto)",
        "primary_hyperparameters": {
            "min_cluster_size": primary_mcs,
            "min_samples": primary_ms,
            "cluster_selection_method": "eom",
            "cluster_selection_epsilon": 0.0,
        },
        "results_summary": {
            "n_clusters": primary_summary.n_clusters,
            "n_noise": primary_summary.n_noise,
            "noise_fraction": primary_summary.noise_fraction,
            "min_cluster_size": primary_summary.min_cluster_size_observed,
            "median_cluster_size": primary_summary.median_cluster_size,
            "max_cluster_size": primary_summary.max_cluster_size,
            "silhouette_clustered": primary_summary.silhouette_score_clustered,
            "runtime_seconds": primary_summary.runtime_seconds,
            "memory_mb": primary_summary.peak_memory_mb,
        },
        "comparison_across_representations": rep_df.to_dict(orient="records"),
        "caveats": [
            "Cluster labels are arbitrary numerical identifiers, not categorical odor labels.",
            "Noise molecules (-1) are valid chemical structures that reside in low-density chemical space.",
            "Membership strengths are density-based heuristic scores, not calibrated probabilities.",
            "Structural clusters do not automatically correspond to human odor categories.",
            "2D UMAP clustering introduces artificial topological compression and is strictly exploratory.",
        ],
    }
    with open(out_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(exp_meta, f, indent=2)
    print(f"Saved experiment metadata to: {out_dir / 'experiment_metadata.json'}")

    # 8. Visualizations
    print("\nGenerating Visualizations...")
    plot_dir = out_dir
    umap_x = umap2_df["umap_1"].fillna(0.0).values
    umap_y = umap2_df["umap_2"].fillna(0.0).values
    primary_labels = primary_clusterer.labels_
    primary_probs = primary_clusterer.probabilities_

    # Plot 1: HDBSCAN Clusters over 2D UMAP
    plt.figure(figsize=(10, 8), dpi=300)
    noise_mask = (primary_labels == -1)
    clustered_mask = ~noise_mask

    # Plot noise points in subtle grey
    plt.scatter(
        umap_x[noise_mask],
        umap_y[noise_mask],
        c="#D1D5DB",
        alpha=0.35,
        s=12,
        label=f"Noise / Sparsity (N={np.sum(noise_mask)})",
        edgecolors="none",
    )

    # Plot clustered points with discrete palette
    cmap = plt.get_cmap("tab20", primary_summary.n_clusters)
    scatter = plt.scatter(
        umap_x[clustered_mask],
        umap_y[clustered_mask],
        c=primary_labels[clustered_mask],
        cmap=cmap,
        alpha=0.85,
        s=22,
        edgecolors="none",
    )
    plt.colorbar(scatter, label="Cluster ID (Morgan FP Jaccard HDBSCAN)")
    plt.title(
        f"HDBSCAN Chemical Space Clusters over 2D UMAP\n(Primary Morgan Fingerprints, Jaccard distance, {primary_summary.n_clusters} clusters, {primary_summary.noise_fraction*100:.1f}% noise)",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("UMAP Dimension 1", fontsize=10)
    plt.ylabel("UMAP Dimension 2", fontsize=10)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plot1_path = plot_dir / "hdbscan_clusters_umap2d.png"
    plt.savefig(plot1_path)
    plt.close()
    print(f"Generated Plot 1: {plot1_path}")

    # Also update the Phase 5 placeholder plot in backend/experiments/phase5/plots/umap_clusters_placeholder.png!
    ph_plot = root / "backend" / "experiments" / "phase5" / "plots" / "umap_clusters_placeholder.png"
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(umap_x[noise_mask], umap_y[noise_mask], c="#E5E7EB", alpha=0.3, s=10, label="Noise")
    plt.scatter(umap_x[clustered_mask], umap_y[clustered_mask], c=primary_labels[clustered_mask], cmap=cmap, alpha=0.85, s=20)
    plt.title("Chemical Space: HDBSCAN Clustering Assignments", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ph_plot)
    plt.close()

    # Plot 2: Noise versus Clustered Molecules
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(
        umap_x[noise_mask],
        umap_y[noise_mask],
        c="#EF4444",
        alpha=0.45,
        s=14,
        label=f"Noise Points (N={np.sum(noise_mask)}, {np.mean(noise_mask)*100:.1f}%)",
        edgecolors="none",
    )
    plt.scatter(
        umap_x[clustered_mask],
        umap_y[clustered_mask],
        c="#10B981",
        alpha=0.75,
        s=20,
        label=f"Clustered Molecules (N={np.sum(clustered_mask)}, {np.mean(clustered_mask)*100:.1f}%)",
        edgecolors="none",
    )
    plt.title(
        "Density Partition: Core Congeneric Clusters vs. Sparse Chemical Noise",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("UMAP Dimension 1", fontsize=10)
    plt.ylabel("UMAP Dimension 2", fontsize=10)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plot2_path = plot_dir / "noise_vs_clustered_umap2d.png"
    plt.savefig(plot2_path)
    plt.close()
    print(f"Generated Plot 2: {plot2_path}")

    # Also update the Phase 5 placeholder outlier plot!
    ph_outlier = root / "backend" / "experiments" / "phase5" / "plots" / "umap_outliers_placeholder.png"
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(umap_x[clustered_mask], umap_y[clustered_mask], c="#10B981", alpha=0.6, s=15, label="Clustered")
    plt.scatter(umap_x[noise_mask], umap_y[noise_mask], c="#EF4444", alpha=0.5, s=15, label="Noise / Isolated")
    plt.title("Chemical Space: Structural Outlier and Noise Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ph_outlier)
    plt.close()

    # Plot 3: Cluster-size Distribution
    plt.figure(figsize=(10, 6), dpi=300)
    non_noise_sizes = sorted([v for k, v in primary_summary.cluster_sizes.items() if k != -1], reverse=True)
    sns.barplot(x=list(range(1, len(non_noise_sizes) + 1)), y=non_noise_sizes, color="#3B82F6")
    plt.axhline(primary_summary.median_cluster_size, color="#EF4444", linestyle="--", label=f"Median Size ({primary_summary.median_cluster_size:.1f})")
    plt.title(f"Rank-Ordered Cluster Size Distribution ({len(non_noise_sizes)} Clusters)", fontsize=12, fontweight="bold")
    plt.xlabel("Cluster Rank", fontsize=10)
    plt.ylabel("Molecule Count", fontsize=10)
    plt.legend()
    plt.tight_layout()
    plot3_path = plot_dir / "cluster_size_distribution.png"
    plt.savefig(plot3_path)
    plt.close()
    print(f"Generated Plot 3: {plot3_path}")

    # Plot 4: Membership-strength Distribution
    plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(primary_probs[clustered_mask], bins=30, kde=True, color="#10B981", label="Clustered Molecules", stat="density", alpha=0.6)
    sns.histplot(primary_probs[noise_mask], bins=30, kde=False, color="#EF4444", label="Noise Molecules (all 0.0)", stat="density", alpha=0.4)
    plt.title("HDBSCAN Membership Strength (Probability) Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Membership Strength $\\lambda$", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend()
    plt.tight_layout()
    plot4_path = plot_dir / "membership_strength_distribution.png"
    plt.savefig(plot4_path)
    plt.close()
    print(f"Generated Plot 4: {plot4_path}")

    # Plot 5: Parameter Sensitivity Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    pivot_clusters = grid_df.pivot(index="min_cluster_size", columns="min_samples", values="n_clusters")
    pivot_noise = grid_df.pivot(index="min_cluster_size", columns="min_samples", values="noise_fraction")

    sns.heatmap(pivot_clusters, annot=True, fmt="d", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("Number of Clusters across (min_cluster_size, min_samples)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("min_samples (-1 = default)", fontsize=10)
    axes[0].set_ylabel("min_cluster_size", fontsize=10)

    sns.heatmap(pivot_noise, annot=True, fmt=".2f", cmap="Reds", ax=axes[1])
    axes[1].set_title("Noise Fraction across (min_cluster_size, min_samples)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("min_samples (-1 = default)", fontsize=10)
    axes[1].set_ylabel("min_cluster_size", fontsize=10)

    plt.tight_layout()
    plot5_path = plot_dir / "parameter_sensitivity_comparison.png"
    plt.savefig(plot5_path)
    plt.close()
    print(f"Generated Plot 5: {plot5_path}")

    print("\nAll Phase 5 HDBSCAN experiments completed successfully!")


if __name__ == "__main__":
    main()
