"""Chemical space clustering and dimensionality reduction package."""

from backend.app.clustering.hdbscan_clustering import (
    ClusterSummary,
    MolecularHDBSCAN,
    compare_representations,
    compute_jaccard_distance_matrix,
    estimate_distance_matrix_memory_mb,
    run_parameter_sensitivity_grid,
    validate_distance_matrix,
)
from backend.app.clustering.service import ChemicalSpaceClusterer
from backend.app.clustering.umap_embeddings import MolecularUMAP, UMAPMetadata

__all__ = [
    "ChemicalSpaceClusterer",
    "ClusterSummary",
    "MolecularHDBSCAN",
    "MolecularUMAP",
    "UMAPMetadata",
    "compare_representations",
    "compute_jaccard_distance_matrix",
    "estimate_distance_matrix_memory_mb",
    "run_parameter_sensitivity_grid",
    "validate_distance_matrix",
]
