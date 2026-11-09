"""Chemical structure similarity calculation using Tanimoto and Morgan fingerprints."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
from rdkit import DataStructs  # type: ignore
from rdkit.Chem import rdMolDescriptors  # type: ignore

from backend.app.chemistry.parser import MoleculeParser
from backend.app.core.exceptions import InvalidSMILESError
from backend.app.similarity.engine import MolecularSimilarityEngine
from backend.app.similarity.models import SimilaritySearchResult


class MolecularSimilarityService:
    """Computes Tanimoto similarity coefficients between molecular fingerprints and provides corpus search."""

    def __init__(
        self,
        radius: int = 2,
        nbits: int = 2048,
        dataset_dir: Optional[Union[str, Path]] = None,
        lazy_load_engine: bool = True,
    ):
        self.radius = radius
        self.nbits = nbits
        self.dataset_dir = dataset_dir
        self.parser = MoleculeParser()
        self._engine: Optional[MolecularSimilarityEngine] = None
        if not lazy_load_engine:
            self._init_engine()

    def _init_engine(self) -> MolecularSimilarityEngine:
        """Initialize or return the cached MolecularSimilarityEngine singleton."""
        if self._engine is None:
            self._engine = MolecularSimilarityEngine(
                dataset_dir=self.dataset_dir,
                radius=self.radius,
                nbits=self.nbits,
                use_chirality=True,
                auto_load=True,
            )
        return self._engine

    @property
    def engine(self) -> MolecularSimilarityEngine:
        """Access the underlying MolecularSimilarityEngine singleton."""
        return self._init_engine()

    def _get_fp(self, smiles: str):
        mol = self.parser.parse_smiles(smiles)
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=self.radius, nBits=self.nbits, useChirality=True
        )

    def calculate_tanimoto(self, smiles_a: str, smiles_b: str) -> float:
        """Calculate pairwise Tanimoto similarity between two molecules."""
        fp_a = self._get_fp(smiles_a)
        fp_b = self._get_fp(smiles_b)
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))

    def rank_similar_molecules(
        self, query_smiles: str, candidate_smiles: List[str], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Rank a candidate pool of molecules by Tanimoto similarity to a query molecule."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        query_fp = self._get_fp(query_smiles)
        scored: List[Tuple[str, float]] = []

        for candidate in candidate_smiles:
            try:
                candidate_fp = self._get_fp(candidate)
                sim = float(DataStructs.TanimotoSimilarity(query_fp, candidate_fp))
                scored.append((candidate, round(sim, 4)))
            except InvalidSMILESError:
                continue

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_corpus(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        exclude_self: bool = True,
    ) -> SimilaritySearchResult:
        """Search the indexed 4,983-molecule corpus using MolecularSimilarityEngine."""
        engine = self._init_engine()
        return engine.search(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
            exclude_self=exclude_self,
        )
