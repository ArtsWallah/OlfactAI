"""Chemical space dimensionality reduction and clustering using UMAP and HDBSCAN."""

from typing import Any, Dict, List, Optional
import numpy as np
import hdbscan
import umap

from backend.app.core.logging import get_logger

logger = get_logger("clustering.service")


class ChemicalSpaceClusterer:
    """Projects molecular fingerprints into 2D/3D manifolds and discovers odor clusters."""

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        min_cluster_size: int = 5,
        random_state: int = 42,
    ):
        self.umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric="jaccard",
            random_state=random_state,
        )
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_epsilon=0.0,
        )

    def fit_transform(self, fingerprint_matrix: np.ndarray) -> Dict[str, Any]:
        """Project binary fingerprints to 2D coordinates and identify cluster labels.

        Args:
            fingerprint_matrix: Binary 2D array of molecular fingerprints (n_samples, n_bits).

        Returns:
            Dictionary containing 2D projections and cluster assignments.
        """
        if fingerprint_matrix.shape[0] < 5:
            raise ValueError("Clustering requires at least 5 molecules.")

        logger.info(f"Computing UMAP projection for {fingerprint_matrix.shape[0]} molecules...")
        coords_2d = self.umap_reducer.fit_transform(fingerprint_matrix)

        logger.info("Identifying clusters with HDBSCAN...")
        labels = self.clusterer.fit_predict(coords_2d)

        return {
            "coordinates": coords_2d.tolist(),
            "cluster_labels": labels.tolist(),
            "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
            "noise_count": int(np.sum(labels == -1)),
        }

