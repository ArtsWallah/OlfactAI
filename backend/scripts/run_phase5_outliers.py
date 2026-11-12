"""Execution script for Phase 5, Step 6: Molecular Outlier Analysis.

Analyzes 4,983 molecules in OpenPOM across five distinct outlier dimensions:
1. Structural Novelty (Max non-self Tanimoto, NN distance, Top-K means)
2. Local Density / Sparsity (Average distance to 10 nearest neighbors)
3. Clustering Outliers (HDBSCAN GLOSH outlier scores)
4. Predictive Uncertainty & Error (Model predictive entropy, Hamming error, BCE loss)
5. Annotation Rarity (Descriptor counts, mean descriptor frequency)

Outputs:
- Consolidated outlier profile (CSV, JSON)
- Candidate molecule inspection tables for each category
- Statistical cross-dimension correlation matrix (CSV, JSON)
- 7 Publication-quality plots (300 DPI):
  1. molecular_novelty_distribution.png
  2. local_density_distribution.png
  3. hdbscan_glosh_distribution.png
  4. structural_novelty_vs_uncertainty.png
  5. structural_novelty_vs_prediction_error.png
  6. umap_chemical_space_outliers.png
  7. outlier_categories_comparison.png
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from backend.app.clustering.molecular_outliers import (
    MolecularOutlierEvaluator,
    compute_annotation_outlier_metrics,
    compute_pairwise_tanimoto_matrix,
    compute_prediction_outlier_metrics,
    compute_structural_novelty,
)
from backend.app.ml.models.xgboost_model import XGBoostSmellModel
from backend.app.ml.preprocessing.scaler import OlfactoryFeatureScaler
from backend.app.similarity.engine import MolecularSimilarityEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase5_outliers")

REPO_ROOT = Path(__file__).resolve().parents[2]
MOL_PATH = REPO_ROOT / "backend/datasets/processed/molecules.csv"
FEAT_PATH = REPO_ROOT / "backend/datasets/processed/molecular_features.csv"
ODOR_PATH = REPO_ROOT / "backend/datasets/processed/odor_labels.csv"
CLUSTER_PATH = REPO_ROOT / "backend/experiments/phase5/clustering/hdbscan_fingerprints_primary.csv"
UMAP_PATH = REPO_ROOT / "backend/experiments/phase5/embeddings/umap_2d_embeddings.csv"
MODEL_PATH = REPO_ROOT / "backend/models/phase3/random/xgboost/model.joblib"
SCALER_PATH = REPO_ROOT / "backend/models/phase3/random/feature_scaler.joblib"

OUT_DIR = REPO_ROOT / "backend/experiments/phase5/outliers"
PLOT_DIR = REPO_ROOT / "backend/experiments/phase5/plots"


def inspect_candidate_molecules(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Compute exact physicochemical properties for candidate outlier molecules."""
    records = []
    for _, row in candidates_df.iterrows():
        smiles = row["canonical_smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            formula = rdMolDescriptors.CalcMolFormula(mol)
            mw = float(Descriptors.MolWt(mol))
            logp = float(Descriptors.MolLogP(mol))
            tpsa = float(Descriptors.TPSA(mol))
            hbd = int(Descriptors.NumHDonors(mol))
            hba = int(Descriptors.NumHAcceptors(mol))
            rotb = int(Descriptors.NumRotatableBonds(mol))
            rings = int(Descriptors.RingCount(mol))
        else:
            formula, mw, logp, tpsa, hbd, hba, rotb, rings = "N/A", np.nan, np.nan, np.nan, 0, 0, 0, 0

        rec = {
            "molecule_id": row["molecule_id"],
            "canonical_smiles": smiles,
            "formula": formula,
            "molecular_weight": mw,
            "logp": logp,
            "tpsa": tpsa,
            "num_h_donors": hbd,
            "num_h_acceptors": hba,
            "num_rotatable_bonds": rotb,
            "ring_count": rings,
            "max_tanimoto_similarity": float(row["max_tanimoto_similarity"]),
            "nearest_neighbor_distance": float(row["nearest_neighbor_distance"]),
            "local_sparsity_k10": float(row["local_sparsity_k10"]),
            "cluster_hdbscan": int(row["cluster_hdbscan"]),
            "outlier_score": float(row["outlier_score"]),
            "annotation_count": int(row["annotation_count"]),
            "annotation_rarity_score": float(row["annotation_rarity_score"]),
            "predictive_entropy": float(row.get("predictive_entropy", np.nan)),
            "bce_loss": float(row.get("bce_loss", np.nan)),
        }
        records.append(rec)
    return pd.DataFrame(records)


def main():
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("1. Loading datasets, cluster assignments, and molecular features...")
    mol_df = pd.read_csv(MOL_PATH)
    feat_df = pd.read_csv(FEAT_PATH)
    odor_df = pd.read_csv(ODOR_PATH)
    cluster_df = pd.read_csv(CLUSTER_PATH)
    umap_df = pd.read_csv(UMAP_PATH)

    n_mols = len(mol_df)
    assert len(feat_df) == n_mols and len(odor_df) == n_mols and len(cluster_df) == n_mols
    logger.info(f"Loaded {n_mols} molecules.")

    # 2. Structural Novelty & Local Density via pairwise Tanimoto
    logger.info("2. Computing pairwise Tanimoto similarity and structural novelty...")
    fp_cols = [c for c in feat_df.columns if c.startswith("fp_")]
    fps = feat_df[fp_cols].to_numpy(dtype=np.float32)

    sim_matrix = compute_pairwise_tanimoto_matrix(fps)
    novelty_df = compute_structural_novelty(sim_matrix, molecule_ids=mol_df["molecule_id"].tolist())
    logger.info(
        f"Structural novelty computed. Max similarity range: [{novelty_df['max_tanimoto_similarity'].min():.3f}, "
        f"{novelty_df['max_tanimoto_similarity'].max():.3f}], Mean: {novelty_df['max_tanimoto_similarity'].mean():.3f}"
    )

    # 3. Annotation Rarity
    logger.info("3. Computing annotation counts and descriptor rarity scores...")
    annotation_df = compute_annotation_outlier_metrics(odor_df)

    # 4. Model Predictions, Uncertainty & Errors
    logger.info("4. Running trained model predictions to extract predictive entropy and losses...")
    scaler = OlfactoryFeatureScaler.load(SCALER_PATH)
    scaled_X = scaler.transform(feat_df)

    model = XGBoostSmellModel()
    model.load(MODEL_PATH)
    probs = model.predict_proba(scaled_X)

    desc_cols = [c for c in odor_df.columns if c not in {"molecule_id", "canonical_smiles"}]
    y_true = odor_df[desc_cols].to_numpy()

    prediction_df = compute_prediction_outlier_metrics(
        y_true=y_true, y_probs=probs, molecule_ids=mol_df["molecule_id"].tolist()
    )

    # 5. Consolidate Outlier Profiles
    logger.info("5. Consolidating multi-dimensional outlier profile...")
    evaluator = MolecularOutlierEvaluator(top_percentile=95.0)

    # Prepare cluster DF subset
    cluster_subset = cluster_df[["molecule_id", "canonical_smiles", "cluster_hdbscan", "membership_strength", "outlier_score"]]

    profile_df = evaluator.consolidate_outlier_profile(
        novelty_df=novelty_df,
        hdbscan_df=cluster_subset,
        annotation_df=annotation_df,
        prediction_df=prediction_df,
    )

    # Merge UMAP coordinates for chemical-space visualization
    profile_df = profile_df.merge(umap_df[["molecule_id", "umap_1", "umap_2"]], on="molecule_id", how="left")

    # Save consolidated profile
    profile_csv = OUT_DIR / "molecular_outlier_profile.csv"
    profile_df.to_csv(profile_csv, index=False)
    profile_json = OUT_DIR / "molecular_outlier_profile.json"
    profile_df.to_json(profile_json, orient="records", indent=2)
    logger.info(f"Saved consolidated outlier profile: {profile_csv}")

    # 6. Statistical Cross-Dimension Correlation Analysis
    logger.info("6. Analyzing cross-dimension Spearman rank correlations...")
    corrs = evaluator.compute_cross_dimension_correlations(profile_df)
    corrs_csv = OUT_DIR / "cross_dimension_correlations.csv"
    corrs.to_csv(corrs_csv)
    logger.info(f"Saved correlation matrix: {corrs_csv}")

    # 7. Candidate Outlier Selection & Structural Inspection
    logger.info("7. Selecting and inspecting representative candidate molecules...")
    # Category 1: Top 5 Structural Novelty (highest nearest_neighbor_distance)
    top_structural = profile_df.sort_values(by="nearest_neighbor_distance", ascending=False).head(5)
    inspect_structural = inspect_candidate_molecules(top_structural)
    inspect_structural.to_csv(OUT_DIR / "candidates_structural_novelty.csv", index=False)

    # Category 2: Top 5 Local Density Sparsity (highest local_sparsity_k10)
    top_density = profile_df.sort_values(by="local_sparsity_k10", ascending=False).head(5)
    inspect_density = inspect_candidate_molecules(top_density)
    inspect_density.to_csv(OUT_DIR / "candidates_local_density.csv", index=False)

    # Category 3: Top 5 HDBSCAN GLOSH Outliers
    top_glosh = profile_df.sort_values(by="outlier_score", ascending=False).head(5)
    inspect_glosh = inspect_candidate_molecules(top_glosh)
    inspect_glosh.to_csv(OUT_DIR / "candidates_hdbscan_glosh.csv", index=False)

    # Category 4: Top 5 Predictive Uncertainty (highest predictive_entropy)
    top_entropy = profile_df.sort_values(by="predictive_entropy", ascending=False).head(5)
    inspect_entropy = inspect_candidate_molecules(top_entropy)
    inspect_entropy.to_csv(OUT_DIR / "candidates_predictive_entropy.csv", index=False)

    # Category 5: Top 5 Annotation Rarity
    top_annot = profile_df.sort_values(by="annotation_rarity_score", ascending=False).head(5)
    inspect_annot = inspect_candidate_molecules(top_annot)
    inspect_annot.to_csv(OUT_DIR / "candidates_annotation_rarity.csv", index=False)

    # 8. Generate 7 Publication-Quality Visualizations (300 DPI)
    logger.info("8. Generating publication-quality visualizations (300 DPI)...")

    # Plot 1: Molecular Novelty Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(profile_df["max_tanimoto_similarity"], bins=50, kde=True, color="#2b5c8f")
    plt.axvline(np.percentile(profile_df["max_tanimoto_similarity"], 5.0), color="#d95f02", linestyle="--", label="5th Percentile (Most Novel)")
    plt.xlabel("Maximum Non-Self Tanimoto Similarity", fontsize=12)
    plt.ylabel("Molecule Count", fontsize=12)
    plt.title("Distribution of Maximum Non-Self Tanimoto Similarity (OpenPOM N=4,983)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p1 = PLOT_DIR / "molecular_novelty_distribution.png"
    plt.savefig(p1, dpi=300)
    plt.close()

    # Plot 2: Local Density Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(profile_df["local_sparsity_k10"], bins=50, kde=True, color="#1b9e77")
    plt.axvline(np.percentile(profile_df["local_sparsity_k10"], 95.0), color="#e7298a", linestyle="--", label="95th Percentile (Sparsest)")
    plt.xlabel("Average Distance to 10 Nearest Neighbors (Local Sparsity)", fontsize=12)
    plt.ylabel("Molecule Count", fontsize=12)
    plt.title("Local Density & Sparsity Distribution across Fingerprint Space", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p2 = PLOT_DIR / "local_density_distribution.png"
    plt.savefig(p2, dpi=300)
    plt.close()

    # Plot 3: HDBSCAN GLOSH Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(profile_df["outlier_score"], bins=50, kde=False, color="#7570b3")
    plt.axvline(np.percentile(profile_df["outlier_score"], 95.0), color="#d95f02", linestyle="--", label="95th Percentile")
    plt.xlabel("HDBSCAN GLOSH Outlier Score", fontsize=12)
    plt.ylabel("Molecule Count", fontsize=12)
    plt.title("HDBSCAN GLOSH Outlier Score Distribution (Primary Jaccard Hierarchy)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p3 = PLOT_DIR / "hdbscan_glosh_distribution.png"
    plt.savefig(p3, dpi=300)
    plt.close()

    # Plot 4: Structural Novelty vs. Predictive Uncertainty
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=profile_df,
        x="nearest_neighbor_distance",
        y="predictive_entropy",
        hue="cluster_hdbscan",
        palette="tab20",
        alpha=0.6,
        s=25,
        legend=False,
    )
    plt.xlabel("Nearest Neighbor Distance (1.0 - Max Non-Self Tanimoto)", fontsize=12)
    plt.ylabel("Mean Predictive Entropy (bits)", fontsize=12)
    plt.title(
        f"Structural Novelty vs. Model Predictive Uncertainty\n"
        f"Spearman r = {corrs.loc['nearest_neighbor_distance', 'predictive_entropy']:.3f}",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p4 = PLOT_DIR / "structural_novelty_vs_uncertainty.png"
    plt.savefig(p4, dpi=300)
    plt.close()

    # Plot 5: Structural Novelty vs. Prediction Error (BCE Loss)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=profile_df,
        x="nearest_neighbor_distance",
        y="bce_loss",
        color="#e41a1c",
        alpha=0.5,
        s=25,
    )
    plt.xlabel("Nearest Neighbor Distance (1.0 - Max Non-Self Tanimoto)", fontsize=12)
    plt.ylabel("Binary Cross-Entropy Loss per Molecule", fontsize=12)
    plt.title(
        f"Structural Novelty vs. Prediction Loss\n"
        f"Spearman r = {corrs.loc['nearest_neighbor_distance', 'bce_loss']:.3f}",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p5 = PLOT_DIR / "structural_novelty_vs_prediction_error.png"
    plt.savefig(p5, dpi=300)
    plt.close()

    # Plot 6: UMAP Chemical Space Highlighting Candidate Outliers
    plt.figure(figsize=(12, 8))
    # Background: all molecules
    plt.scatter(profile_df["umap_1"], profile_df["umap_2"], c="#cccccc", s=15, alpha=0.4, label="Dataset Molecules")
    # Structural outliers
    plt.scatter(
        profile_df.loc[profile_df["is_structural_outlier"], "umap_1"],
        profile_df.loc[profile_df["is_structural_outlier"], "umap_2"],
        c="#e41a1c",
        s=40,
        alpha=0.8,
        label="Structural Novelty (Top 5%)",
    )
    # Density outliers
    plt.scatter(
        profile_df.loc[profile_df["is_density_outlier"], "umap_1"],
        profile_df.loc[profile_df["is_density_outlier"], "umap_2"],
        facecolors="none",
        edgecolors="#377eb8",
        s=70,
        linewidths=1.2,
        label="Sparse Density (Top 5%)",
    )
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.title("Chemical Space Projection: Candidate Outliers across Manifold", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    p6 = PLOT_DIR / "umap_chemical_space_outliers.png"
    plt.savefig(p6, dpi=300)
    plt.close()

    # Plot 7: Outlier Categories Comparison (Venn/Overlap counts)
    cat_overlap = pd.DataFrame(
        {
            "Structural Novelty": profile_df["is_structural_outlier"],
            "Sparse Density": profile_df["is_density_outlier"],
            "HDBSCAN Noise": profile_df["cluster_hdbscan"] == -1,
            "High Uncertainty": profile_df["is_uncertainty_outlier"],
            "Rare Annotation": profile_df["is_annotation_outlier"],
        }
    )
    overlap_matrix = cat_overlap.T.dot(cat_overlap)

    plt.figure(figsize=(10, 8))
    sns.heatmap(overlap_matrix, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={"label": "Molecule Count"})
    plt.title(
        "Pairwise Overlap Between Outlier Categories (Top 5% Thresholds)\n"
        "Demonstrates Conceptual and Empirical Separation Between Categories",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    p7 = PLOT_DIR / "outlier_categories_comparison.png"
    plt.savefig(p7, dpi=300)
    plt.close()

    # 9. Save Metadata
    elapsed_time = time.time() - start_time
    metadata = {
        "dataset_size": n_mols,
        "n_descriptors": len(desc_cols),
        "execution_time_seconds": round(elapsed_time, 2),
        "percentile_threshold": 95.0,
        "outlier_counts": {
            "structural_novelty": int(profile_df["is_structural_outlier"].sum()),
            "local_density_sparsity": int(profile_df["is_density_outlier"].sum()),
            "hdbscan_noise": int((profile_df["cluster_hdbscan"] == -1).sum()),
            "hdbscan_glosh": int(profile_df["is_glosh_outlier"].sum()),
            "predictive_uncertainty": int(profile_df["is_uncertainty_outlier"].sum()),
            "annotation_rarity": int(profile_df["is_annotation_outlier"].sum()),
        },
        "spearman_correlations": {
            "novelty_vs_uncertainty": round(float(corrs.loc["nearest_neighbor_distance", "predictive_entropy"]), 4),
            "novelty_vs_bce_loss": round(float(corrs.loc["nearest_neighbor_distance", "bce_loss"]), 4),
            "novelty_vs_density": round(float(corrs.loc["nearest_neighbor_distance", "local_sparsity_k10"]), 4),
            "novelty_vs_glosh": round(float(corrs.loc["nearest_neighbor_distance", "outlier_score"]), 4),
            "novelty_vs_annot_rarity": round(float(corrs.loc["nearest_neighbor_distance", "annotation_rarity_score"]), 4),
        },
        "plots_generated": [str(p.name) for p in [p1, p2, p3, p4, p5, p6, p7]],
    }
    with open(OUT_DIR / "outlier_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved outlier metadata: {OUT_DIR / 'outlier_metadata.json'}")

    print("\n" + "=" * 65)
    print("PHASE 5 STEP 6: MOLECULAR OUTLIER ANALYSIS COMPLETE")
    print("=" * 65)
    print(f"Total molecules evaluated: {n_mols}")
    print(f"Spearman r (Novelty vs Predictive Uncertainty): {metadata['spearman_correlations']['novelty_vs_uncertainty']:.4f}")
    print(f"Spearman r (Novelty vs Prediction Loss): {metadata['spearman_correlations']['novelty_vs_bce_loss']:.4f}")
    print(f"Spearman r (Novelty vs Local Density Sparsity): {metadata['spearman_correlations']['novelty_vs_density']:.4f}")
    print(f"Execution time: {elapsed_time:.2f} s")
    print("=" * 65)


if __name__ == "__main__":
    main()

