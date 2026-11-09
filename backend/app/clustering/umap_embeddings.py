"""Reproducible UMAP manifold learning and chemical space dimensionality reduction."""

import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap

from backend.app.core.logging import get_logger

logger = get_logger("clustering.umap")

# Physicochemical descriptor column names established in Phase 2
PHYSICOCHEMICAL_COLUMNS = [
    "molecular_weight",
    "logp",
    "tpsa",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "ring_count",
    "heavy_atom_count",
    "fraction_csp3",
]


@dataclass
class UMAPMetadata:
    """Metadata schema documenting parameters, reproducibility settings, and environment."""

    embedding_type: str
    n_components: int
    n_neighbors: int
    min_dist: float
    metric: str
    random_state: int
    input_dimension: int
    corpus_size: int
    execution_time_seconds: float
    peak_memory_mb: float
    umap_version: str
    python_version: str
    platform: str
    scaling_applied: bool
    scientific_disclaimer: str


class MolecularUMAP:
    """Computes reproducible 2D and multi-dimensional UMAP embeddings over chemical representations."""

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "jaccard",
        random_state: int = 42,
    ):
        """Initialize UMAP configuration.

        Args:
            n_neighbors: Number of nearest neighbors for local manifold approximation (default 15).
            min_dist: Minimum distance between points in low-dimensional space (default 0.1).
            metric: Distance metric for input space (default 'jaccard' for binary fingerprints).
            random_state: Random seed for deterministic reproducibility (default 42).
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

    def _get_peak_memory_mb(self) -> float:
        """Measure peak process memory in megabytes."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On Linux ru_maxrss is in kilobytes; on macOS it is in bytes
            if platform.system() == "Darwin":
                return round(float(usage) / (1024.0 * 1024.0), 2)
            return round(float(usage) / 1024.0, 2)
        except Exception:
            return 0.0

    def fit_transform_fingerprints(
        self,
        features_df: pd.DataFrame,
        n_components: int = 2,
    ) -> Tuple[pd.DataFrame, UMAPMetadata]:
        """Project binary Morgan circular fingerprints into low-dimensional UMAP coordinates.

        Args:
            features_df: DataFrame containing molecule_id, canonical_smiles, and fp_0...fp_2047.
            n_components: Target embedding dimension (e.g. 2 for visualization, 10 for clustering).

        Returns:
            Tuple of (DataFrame with preserved row alignment, UMAPMetadata).
        """
        if "molecule_id" not in features_df.columns or "canonical_smiles" not in features_df.columns:
            raise ValueError("features_df must contain 'molecule_id' and 'canonical_smiles' columns.")

        fp_cols = [c for c in features_df.columns if c.startswith("fp_")]
        if len(fp_cols) == 0:
            raise ValueError("No Morgan fingerprint columns (fp_*) found in features_df.")

        X = features_df[fp_cols].values.astype(np.bool_)
        n_samples, n_bits = X.shape

        if n_samples < 5:
            raise ValueError(f"UMAP requires at least 5 molecules, got {n_samples}.")

        logger.info(
            f"Fitting UMAP (components={n_components}, metric={self.metric}, neighbors={self.n_neighbors}, "
            f"seed={self.random_state}) on {n_samples} molecules x {n_bits} bits..."
        )

        t0 = time.time()
        # When random_state is set, UMAP internally sets n_jobs=1 for strict reproducibility
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=n_components,
            metric=self.metric,
            random_state=self.random_state,
        )
        embeddings = reducer.fit_transform(X)
        duration = round(float(time.time() - t0), 2)
        peak_mem = self._get_peak_memory_mb()

        # Build output dataframe preserving strict row alignment and IDs
        coord_cols = [f"umap_{i+1}" for i in range(n_components)]
        emb_df = pd.DataFrame(embeddings, columns=coord_cols)
        emb_df.insert(0, "canonical_smiles", features_df["canonical_smiles"].values)
        emb_df.insert(0, "molecule_id", features_df["molecule_id"].values)

        metadata = UMAPMetadata(
            embedding_type="morgan_fingerprints",
            n_components=n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            input_dimension=n_bits,
            corpus_size=n_samples,
            execution_time_seconds=duration,
            peak_memory_mb=peak_mem,
            umap_version=umap.__version__,
            python_version=platform.python_version(),
            platform=platform.platform(),
            scaling_applied=False,
            scientific_disclaimer=(
                "UMAP is a non-linear stochastic manifold learning technique. Low-dimensional Euclidean distances "
                "do not reflect exact chemical distances or quantitative sensory perception differences. "
                "Embeddings are exploratory representations and do not prove biological olfactory categories."
            ),
        )

        return emb_df, metadata

    def fit_transform_physicochemical(
        self,
        features_df: pd.DataFrame,
        n_components: int = 2,
    ) -> Tuple[pd.DataFrame, UMAPMetadata]:
        """Project standardized physicochemical properties into 2D UMAP space (secondary experiment).

        Args:
            features_df: DataFrame containing the 9 physicochemical properties.
            n_components: Output coordinates (default 2).

        Returns:
            Tuple of (DataFrame with preserved IDs and coordinates, UMAPMetadata).
        """
        missing_cols = [c for c in PHYSICOCHEMICAL_COLUMNS if c not in features_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required physicochemical columns: {missing_cols}")

        X_raw = features_df[PHYSICOCHEMICAL_COLUMNS].values.astype(np.float32)
        n_samples, n_features = X_raw.shape

        # Standardize features to zero mean and unit variance for Euclidean distance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        t0 = time.time()
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=n_components,
            metric="euclidean",
            random_state=self.random_state,
        )
        embeddings = reducer.fit_transform(X_scaled)
        duration = round(float(time.time() - t0), 2)
        peak_mem = self._get_peak_memory_mb()

        coord_cols = [f"umap_{i+1}" for i in range(n_components)]
        emb_df = pd.DataFrame(embeddings, columns=coord_cols)
        emb_df.insert(0, "canonical_smiles", features_df["canonical_smiles"].values)
        emb_df.insert(0, "molecule_id", features_df["molecule_id"].values)

        metadata = UMAPMetadata(
            embedding_type="physicochemical_descriptors",
            n_components=n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric="euclidean",
            random_state=self.random_state,
            input_dimension=n_features,
            corpus_size=n_samples,
            execution_time_seconds=duration,
            peak_memory_mb=peak_mem,
            umap_version=umap.__version__,
            python_version=platform.python_version(),
            platform=platform.platform(),
            scaling_applied=True,
            scientific_disclaimer=(
                "Physicochemical UMAP embeds 9 standardized molecular descriptors (MW, LogP, TPSA, H-bonding, etc.). "
                "Because standard scaling treats all 9 features equally, proximity reflects shared bulk physical properties "
                "rather than fine stereochemical or circular substructural similarity."
            ),
        )

        return emb_df, metadata

    @staticmethod
    def assign_dominant_odor_labels(labels_df: pd.DataFrame) -> pd.Series:
        """Assign an explicit dominant odor descriptor for each molecule with deterministic tie-breaking.

        Rules:
        1. If a molecule has 0 positive annotations -> 'unlabeled'.
        2. If a molecule has multiple active descriptors -> select the label with the highest
           overall background prevalence in the dataset.
        3. If multiple active labels are tied for highest background prevalence -> select the
           lexicographically first descriptor string (alphabetical tie-breaking).

        Scientific note:
        Dataset annotations are binary presence indicators without sensory intensity or concentration ratings.
        A 'dominant' descriptor here is an annotation summary for exploratory coloring, NOT proof of subjective
        perceptual sensory dominance in human olfactory psychophysics.

        Args:
            labels_df: DataFrame containing molecule_id and binary odor label columns.

        Returns:
            pd.Series of dominant label strings aligned with the input rows.
        """
        odor_cols = [c for c in labels_df.columns if c not in ["molecule_id", "canonical_smiles"]]
        label_matrix = labels_df[odor_cols].values

        # Calculate overall dataset prevalence for each descriptor
        prevalence = labels_df[odor_cols].sum(axis=0).to_dict()

        dominant_list = []
        for i in range(len(labels_df)):
            active_indices = np.where(label_matrix[i] == 1)[0]
            if len(active_indices) == 0:
                dominant_list.append("unlabeled")
            elif len(active_indices) == 1:
                dominant_list.append(odor_cols[active_indices[0]])
            else:
                candidate_labels = [odor_cols[idx] for idx in active_indices]
                # Rank candidates by: (-prevalence, label_name)
                best_label = min(candidate_labels, key=lambda l: (-prevalence[l], l))
                dominant_list.append(best_label)

        return pd.Series(dominant_list, index=labels_df.index, name="dominant_odor")

    def generate_all_visualizations(
        self,
        umap_2d: pd.DataFrame,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        molecules_df: pd.DataFrame,
        output_dir: Union[str, Path],
        phys_umap_2d: Optional[pd.DataFrame] = None,
    ) -> Dict[str, str]:
        """Generate and save publication-quality scientific visualizations of the chemical manifold.

        Plots created:
        1. umap_dominant_odor.png
        2. umap_physicochemical_properties.png
        3. umap_clusters_placeholder.png
        4. umap_outliers_placeholder.png
        5. umap_physicochemical_space.png (if phys_umap_2d supplied)

        Args:
            umap_2d: 2D Morgan fingerprint UMAP coordinates.
            features_df: Physicochemical descriptors and features.
            labels_df: Odor label annotations.
            molecules_df: Molecule metadata including scaffold and random split assignments.
            output_dir: Directory where PNG plots are saved.
            phys_umap_2d: Optional 2D physicochemical UMAP coordinates.

        Returns:
            Dict mapping plot name to saved file path.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved_plots = {}

        # Merge for plotting alignment
        plot_df = umap_2d.copy()
        dominant_series = self.assign_dominant_odor_labels(labels_df)
        plot_df["dominant_odor"] = dominant_series.values

        # Add physicochemical properties
        for prop in ["molecular_weight", "logp", "tpsa"]:
            if prop in features_df.columns:
                plot_df[prop] = features_df[prop].values

        if "scaffold_split" in molecules_df.columns:
            plot_df["scaffold_split"] = molecules_df["scaffold_split"].values

        # Set clean aesthetic styling
        sns.set_theme(style="whitegrid", font="sans-serif")
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

        # -------------------------------------------------------------
        # Plot 1: Dominant Odor Descriptors
        # -------------------------------------------------------------
        top_n = 10
        dominant_counts = plot_df["dominant_odor"].value_counts()
        top_descriptors = dominant_counts.index[:top_n].tolist()

        # Group rarer descriptors as 'other' for visual legibility
        plot_df["display_odor"] = plot_df["dominant_odor"].apply(
            lambda x: x if x in top_descriptors or x == "unlabeled" else "other"
        )

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        palette = sns.color_palette("tab10", n_colors=len(top_descriptors))
        color_map = {lbl: palette[i] for i, lbl in enumerate(top_descriptors)}
        color_map["other"] = (0.75, 0.75, 0.75, 0.6)  # muted gray
        color_map["unlabeled"] = (0.2, 0.2, 0.2, 0.8)

        # Plot 'other' first as background
        other_mask = plot_df["display_odor"] == "other"
        ax.scatter(
            plot_df.loc[other_mask, "umap_1"],
            plot_df.loc[other_mask, "umap_2"],
            c=[color_map["other"]],
            s=12,
            alpha=0.4,
            label=f"other ({np.sum(other_mask):,})",
            edgecolors="none",
        )

        for desc in top_descriptors:
            mask = plot_df["display_odor"] == desc
            cnt = int(np.sum(mask))
            ax.scatter(
                plot_df.loc[mask, "umap_1"],
                plot_df.loc[mask, "umap_2"],
                c=[color_map[desc]],
                s=18,
                alpha=0.75,
                label=f"{desc} ({cnt:,})",
                edgecolors="none",
            )

        ax.set_title(
            "OlfactAI Chemical Space — 2D UMAP by Dominant Odor Descriptor\n"
            "(Morgan 2048-bit Fingerprints, Jaccard Distance, random_state=42)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("UMAP Dimension 1", fontsize=10)
        ax.set_ylabel("UMAP Dimension 2", fontsize=10)
        ax.legend(title="Dominant Descriptor", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        plt.tight_layout()

        p1_path = out_path / "umap_dominant_odor.png"
        fig.savefig(p1_path, dpi=300)
        plt.close(fig)
        saved_plots["dominant_odor"] = str(p1_path)

        # -------------------------------------------------------------
        # Plot 2: Selected Physicochemical Properties (MW, LogP, TPSA)
        # -------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
        props_meta = [
            ("molecular_weight", "Molecular Weight (g/mol)", "viridis"),
            ("logp", "Calculated LogP", "coolwarm"),
            ("tpsa", "Topological Polar Surface Area (Å²)", "plasma"),
        ]

        for ax, (prop, title, cmap) in zip(axes, props_meta):
            vals = plot_df[prop].values
            scatter = ax.scatter(
                plot_df["umap_1"],
                plot_df["umap_2"],
                c=vals,
                cmap=cmap,
                s=14,
                alpha=0.75,
                edgecolors="none",
            )
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(title, fontsize=9)
            ax.set_title(f"UMAP Colored by {title.split('(')[0].strip()}", fontsize=11, fontweight="bold")
            ax.set_xlabel("UMAP Dimension 1", fontsize=9)
            ax.set_ylabel("UMAP Dimension 2", fontsize=9)

        fig.suptitle(
            "Physicochemical Distribution Across 2D Morgan Fingerprint Manifold",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        p2_path = out_path / "umap_physicochemical_properties.png"
        fig.savefig(p2_path, dpi=300)
        plt.close(fig)
        saved_plots["physicochemical_properties"] = str(p2_path)

        # -------------------------------------------------------------
        # Plot 3: Cluster Assignments (Placeholder / Scaffold Overlay)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        split_colors = {"train": "#2b5c8f", "val": "#f39c12", "test": "#c0392b"}

        for split_name, color in split_colors.items():
            mask = plot_df["scaffold_split"] == split_name
            cnt = int(np.sum(mask))
            ax.scatter(
                plot_df.loc[mask, "umap_1"],
                plot_df.loc[mask, "umap_2"],
                c=color,
                s=16,
                alpha=0.65,
                label=f"Scaffold Partition: {split_name} ({cnt:,})",
                edgecolors="none",
            )

        ax.set_title(
            "Chemical Space Partitioning — Structural Scaffold Split Distribution\n"
            "(Baseline cluster overlay prepared for Step 6 HDBSCAN clustering)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("UMAP Dimension 1", fontsize=10)
        ax.set_ylabel("UMAP Dimension 2", fontsize=10)
        ax.legend(title="Structural Grouping", loc="upper right", frameon=True)
        plt.tight_layout()

        p3_path = out_path / "umap_clusters_placeholder.png"
        fig.savefig(p3_path, dpi=300)
        plt.close(fig)
        saved_plots["clusters_placeholder"] = str(p3_path)

        # -------------------------------------------------------------
        # Plot 4: Manifold Sparsity & Isolated Outlier Overlay (Step 6 Prep)
        # -------------------------------------------------------------
        # Highlight points with extreme UMAP distance from centroid as exploratory density proxy
        centroid = np.array([plot_df["umap_1"].mean(), plot_df["umap_2"].mean()])
        dists = np.sqrt((plot_df["umap_1"] - centroid[0]) ** 2 + (plot_df["umap_2"] - centroid[1]) ** 2)
        outlier_cutoff = np.percentile(dists, 97.5)
        is_isolated = dists >= outlier_cutoff

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        ax.scatter(
            plot_df.loc[~is_isolated, "umap_1"],
            plot_df.loc[~is_isolated, "umap_2"],
            c="#7f8c8d",
            s=14,
            alpha=0.5,
            label=f"Core Manifold Density ({np.sum(~is_isolated):,})",
            edgecolors="none",
        )
        ax.scatter(
            plot_df.loc[is_isolated, "umap_1"],
            plot_df.loc[is_isolated, "umap_2"],
            c="#e74c3c",
            s=28,
            alpha=0.9,
            label=f"Peripheral / Isolated Structures ({np.sum(is_isolated):,})",
            edgecolors="black",
            linewidths=0.5,
        )

        ax.set_title(
            "Chemical Manifold Density & Structural Outliers\n"
            "(Preliminary outlier highlight prepared for Step 6 HDBSCAN noise discovery)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("UMAP Dimension 1", fontsize=10)
        ax.set_ylabel("UMAP Dimension 2", fontsize=10)
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()

        p4_path = out_path / "umap_outliers_placeholder.png"
        fig.savefig(p4_path, dpi=300)
        plt.close(fig)
        saved_plots["outliers_placeholder"] = str(p4_path)

        # -------------------------------------------------------------
        # Plot 5: Secondary Experiment — Physicochemical 2D UMAP Space
        # -------------------------------------------------------------
        if phys_umap_2d is not None:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
            phys_plot = phys_umap_2d.copy()
            phys_plot["dominant_odor"] = plot_df["display_odor"].values

            other_phys = phys_plot["dominant_odor"] == "other"
            ax.scatter(
                phys_plot.loc[other_phys, "umap_1"],
                phys_plot.loc[other_phys, "umap_2"],
                c=[color_map["other"]],
                s=12,
                alpha=0.4,
                label=f"other ({np.sum(other_phys):,})",
                edgecolors="none",
            )

            for desc in top_descriptors:
                mask = phys_plot["dominant_odor"] == desc
                cnt = int(np.sum(mask))
                ax.scatter(
                    phys_plot.loc[mask, "umap_1"],
                    phys_plot.loc[mask, "umap_2"],
                    c=[color_map[desc]],
                    s=18,
                    alpha=0.75,
                    label=f"{desc} ({cnt:,})",
                    edgecolors="none",
                )

            ax.set_title(
                "Secondary Experiment — 2D UMAP of 9 Standardized Physicochemical Descriptors\n"
                "(Euclidean Distance, StandardScaler, random_state=42)",
                fontsize=12,
                fontweight="bold",
                pad=12,
            )
            ax.set_xlabel("Physicochemical UMAP Dimension 1", fontsize=10)
            ax.set_ylabel("Physicochemical UMAP Dimension 2", fontsize=10)
            ax.legend(title="Dominant Descriptor", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
            plt.tight_layout()

            p5_path = out_path / "umap_physicochemical_space.png"
            fig.savefig(p5_path, dpi=300)
            plt.close(fig)
            saved_plots["physicochemical_space"] = str(p5_path)

        return saved_plots

    def run_full_pipeline(
        self,
        dataset_dir: Optional[Union[str, Path]] = None,
        embeddings_dir: Optional[Union[str, Path]] = None,
        plots_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Execute end-to-end 2D, 10D, and physicochemical UMAP embedding generation and plotting."""
        ds_dir = Path(dataset_dir) if dataset_dir else Path("backend/datasets/processed")
        emb_dir = Path(embeddings_dir) if embeddings_dir else Path("backend/experiments/phase5/embeddings")
        plt_dir = Path(plots_dir) if plots_dir else Path("backend/experiments/phase5/plots")

        emb_dir.mkdir(parents=True, exist_ok=True)
        plt_dir.mkdir(parents=True, exist_ok=True)

        features_df = pd.read_csv(ds_dir / "molecular_features.csv")
        labels_df = pd.read_csv(ds_dir / "odor_labels.csv")
        molecules_df = pd.read_csv(ds_dir / "molecules.csv")

        # 1. Primary 2D UMAP (Morgan Fingerprints, Jaccard Distance)
        logger.info("Computing 2D Morgan Fingerprint UMAP embeddings...")
        umap_2d, meta_2d = self.fit_transform_fingerprints(features_df, n_components=2)
        umap_2d_path = emb_dir / "umap_2d_embeddings.csv"
        umap_2d.to_csv(umap_2d_path, index=False)

        # 2. Intermediate 10D UMAP (For Step 6 HDBSCAN comparison)
        logger.info("Computing 10D Morgan Fingerprint UMAP embeddings...")
        umap_10d, meta_10d = self.fit_transform_fingerprints(features_df, n_components=10)
        umap_10d_path = emb_dir / "umap_10d_embeddings.csv"
        umap_10d.to_csv(umap_10d_path, index=False)

        # 3. Secondary Experiment: 2D Physicochemical UMAP
        logger.info("Computing 2D Physicochemical UMAP embeddings (secondary experiment)...")
        phys_umap_2d, meta_phys = self.fit_transform_physicochemical(features_df, n_components=2)
        phys_umap_path = emb_dir / "umap_physicochemical_2d_embeddings.csv"
        phys_umap_2d.to_csv(phys_umap_path, index=False)

        # Save metadata JSON documenting all experiments
        metadata_combined = {
            "umap_2d_fingerprints": asdict(meta_2d),
            "umap_10d_fingerprints": asdict(meta_10d),
            "umap_2d_physicochemical": asdict(meta_phys),
        }
        meta_path = emb_dir / "embedding_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_combined, f, indent=2)

        # 4. Generate Visualizations
        logger.info("Generating publication-quality visualization figures...")
        plots = self.generate_all_visualizations(
            umap_2d=umap_2d,
            features_df=features_df,
            labels_df=labels_df,
            molecules_df=molecules_df,
            output_dir=plt_dir,
            phys_umap_2d=phys_umap_2d,
        )

        return {
            "umap_2d_path": str(umap_2d_path),
            "umap_10d_path": str(umap_10d_path),
            "phys_umap_path": str(phys_umap_path),
            "metadata_path": str(meta_path),
            "plots": plots,
            "metadata": metadata_combined,
        }

