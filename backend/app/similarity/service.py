"""Chemical structure similarity calculation using Tanimoto and Morgan fingerprints."""

from typing import List, Tuple
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

from backend.app.chemistry.parser import MoleculeParser
from backend.app.core.exceptions import InvalidSMILESError


class MolecularSimilarityService:
    """Computes Tanimoto similarity coefficients between molecular fingerprints."""

    def __init__(self, radius: int = 2, nbits: int = 2048):
        self.radius = radius
        self.nbits = nbits
        self.parser = MoleculeParser()

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
        query_fp = self._get_fp(query_smiles)
        scored = []

        for candidate in candidate_smiles:
            try:
                candidate_fp = self._get_fp(candidate)
                sim = float(DataStructs.TanimotoSimilarity(query_fp, candidate_fp))
                scored.append((candidate, round(sim, 4)))
            except InvalidSMILESError:
                continue

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

