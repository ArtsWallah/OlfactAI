"""Generation of circular Morgan fingerprints (ECFP equivalent) using RDKit."""

from typing import Optional
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from backend.app.core.exceptions import MolecularProcessingError
from backend.app.schemas.chemistry import FingerprintMetadata


class FingerprintGenerator:
    """Generates Morgan / circular bit vectors and compact sparse representations."""

    @staticmethod
    def generate_morgan_fingerprint(
        mol: Chem.Mol,
        radius: int = 2,
        nbits: int = 2048,
        include_dense_vector: bool = False,
    ) -> FingerprintMetadata:
        """Generate Morgan circular fingerprint for a molecule.

        Args:
            mol: Parsed and sanitized RDKit Mol object.
            radius: Radius of atomic environment (e.g. 2 for ECFP4-like).
            nbits: Total bit length of the folded vector (e.g. 2048).
            include_dense_vector: Whether to include the full binary list.

        Returns:
            FingerprintMetadata schema instance containing sparse on-bits and optional dense array.

        Raises:
            MolecularProcessingError: If fingerprint calculation fails.
        """
        if mol is None:
            raise MolecularProcessingError("Cannot generate fingerprint from a null molecule.")

        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol,
                radius=radius,
                nBits=nbits,
                useChirality=True,
            )

            on_bits = list(fp.GetOnBits())
            dense_vector: Optional[list] = None

            if include_dense_vector:
                # Convert ExplicitBitVect to list of ints (0 or 1)
                dense_vector = [int(fp.GetBit(i)) for i in range(nbits)]

            return FingerprintMetadata(
                radius=radius,
                nbits=nbits,
                on_bits_count=len(on_bits),
                on_bit_indices=on_bits,
                bit_vector=dense_vector,
            )
        except Exception as e:
            raise MolecularProcessingError(
                message=f"Error generating Morgan fingerprint: {str(e)}",
                details={"radius": radius, "nbits": nbits, "error": str(e)},
            ) from e

