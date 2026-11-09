"""Expose similarity models, engine, service, and neighborhood analyzer."""

from backend.app.similarity.engine import MolecularSimilarityEngine
from backend.app.similarity.models import (
    SimilarityBenchmarkResult,
    SimilarityMatch,
    SimilaritySearchResult,
)
from backend.app.similarity.neighborhood import (
    ClusterBootstrapSummary,
    GlobalPairwiseSummary,
    MolecularNeighborhoodAnalyzer,
    NeighborhoodResult,
    NeighborOdorMatch,
    TanimotoBinSummary,
    calculate_jaccard_similarity,
    calculate_label_overlap,
)
from backend.app.similarity.service import MolecularSimilarityService

__all__ = [
    "MolecularSimilarityEngine",
    "MolecularSimilarityService",
    "MolecularNeighborhoodAnalyzer",
    "SimilarityMatch",
    "SimilaritySearchResult",
    "SimilarityBenchmarkResult",
    "NeighborhoodResult",
    "NeighborOdorMatch",
    "TanimotoBinSummary",
    "GlobalPairwiseSummary",
    "ClusterBootstrapSummary",
    "calculate_jaccard_similarity",
    "calculate_label_overlap",
]
