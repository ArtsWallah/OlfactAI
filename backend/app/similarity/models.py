"""Data models and schemas for molecular similarity search."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SimilarityMatch:
    """Represents a single nearest-neighbor molecule match."""

    molecule_id: str
    canonical_smiles: str
    tanimoto_similarity: float
    rank: int
    known_odor_labels: List[str] = field(default_factory=list)
    physicochemical_descriptors: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimilarityBenchmarkResult:
    """Timing benchmark breakdown for similarity search operations."""

    fp_generation_ms: float
    bulk_tanimoto_ms: float
    ranking_packaging_ms: float
    end_to_end_ms: float
    n_repeats: int
    corpus_size: int


@dataclass
class SimilaritySearchResult:
    """Complete result returned from a similarity query."""

    query: str
    query_type: str  # "dataset_id" or "smiles"
    query_molecule_id: Optional[str]
    query_canonical_smiles: str
    matches: List[SimilarityMatch] = field(default_factory=list)
    total_candidates_searched: int = 0
    timing_ms: Optional[SimilarityBenchmarkResult] = None

