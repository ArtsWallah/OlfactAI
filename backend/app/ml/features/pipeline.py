"""Feature extraction pipeline converting SMILES into numerical ML feature matrices."""

from typing import List, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from backend.app.chemistry.parser import MoleculeParser
from backend.app.core.logging import get_logger

logger = get_logger("ml.features.pipeline")


class MolecularFeaturePipeline:
    """Extracts Morgan fingerprint bits and scaled physicochemical descriptors from SMILES."""

    def __init__(self, morgan_radius: int = 2, morgan_nbits: int = 2048, include_descriptors: bool = True):
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
        self.include_descriptors = include_descriptors
        self.parser = MoleculeParser()

    def transform_single(self, smiles: str) -> np.ndarray:
        """Extract feature vector for a single SMILES string.

        Args:
            smiles: Input SMILES string.

        Returns:
            1D numpy float32 array.
        """
        mol = self.parser.parse_smiles(smiles)

        # 1. Morgan Fingerprint
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.morgan_radius,
            nBits=self.morgan_nbits,
            useChirality=True,
        )
        fp_bits = np.array([fp.GetBit(i) for i in range(self.morgan_nbits)], dtype=np.float32)

        if not self.include_descriptors:
            return fp_bits

        # 2. Physicochemical Descriptors
        desc = np.array(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.RingCount(mol),
                mol.GetNumHeavyAtoms(),
                Descriptors.FractionCSP3(mol),
            ],
            dtype=np.float32,
        )

        return np.concatenate([fp_bits, desc])

    def transform_batch(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[int]]:
        """Transform a batch of SMILES strings, skipping unparseable structures.

        Returns:
            Tuple of (features_matrix: np.ndarray, valid_indices: List[int])
        """
        features = []
        valid_indices = []

        for idx, sm in enumerate(smiles_list):
            try:
                feat = self.transform_single(sm)
                features.append(feat)
                valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Skipping invalid SMILES at index {idx} ('{sm}'): {e}")

        if not features:
            return np.empty((0, self.morgan_nbits)), []

        return np.vstack(features), valid_indices

