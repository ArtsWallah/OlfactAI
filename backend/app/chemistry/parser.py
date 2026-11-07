"""Molecular parser and validation utilities using RDKit."""

from typing import Optional, Tuple
from rdkit import Chem
from rdkit import RDLogger

from backend.app.core.exceptions import InvalidSMILESError
from backend.app.core.logging import get_logger

logger = get_logger("chemistry.parser")


class MoleculeParser:
    """Handles SMILES string sanitization, parsing, validation and canonicalization."""

    @staticmethod
    def parse_smiles(
        smiles: str,
        sanitize: bool = True,
    ) -> Chem.Mol:
        """Parse a SMILES string into an RDKit Mol object.

        Args:
            smiles: Input SMILES string.
            sanitize: Whether to sanitize the molecule upon parsing.

        Returns:
            RDKit Mol instance.

        Raises:
            InvalidSMILESError: If the SMILES string cannot be parsed or sanitized.
        """
        if not smiles or not isinstance(smiles, str) or not smiles.strip():
            raise InvalidSMILESError(
                message="SMILES string cannot be empty or whitespace.",
                smiles=smiles,
            )

        trimmed = smiles.strip()

        # Suppress noisy C++ stdout errors during trial parse
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)

        try:
            mol = Chem.MolFromSmiles(trimmed, sanitize=sanitize)
        except Exception as e:
            raise InvalidSMILESError(
                message=f"Chemistry parsing exception: {str(e)}",
                smiles=trimmed,
            ) from e

        if mol is None:
            raise InvalidSMILESError(
                message="Structure rejected by RDKit validator. Ensure valence and aromaticity rules are satisfied.",
                smiles=trimmed,
            )

        return mol

    @staticmethod
    def validate_smiles(smiles: str) -> Tuple[bool, Optional[str], str]:
        """Validate a SMILES string and return canonical representation if valid.

        Args:
            smiles: Input SMILES string.

        Returns:
            Tuple of (is_valid: bool, canonical_smiles: Optional[str], message: str)
        """
        if not smiles or not isinstance(smiles, str) or not smiles.strip():
            return False, None, "SMILES string cannot be empty or whitespace."

        trimmed = smiles.strip()
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)

        mol = Chem.MolFromSmiles(trimmed, sanitize=True)
        if mol is None:
            return False, None, "Invalid chemical structure or valence violation."

        canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return True, canonical, "SMILES structure is chemically valid."

    @staticmethod
    def canonicalize(smiles: str) -> str:
        """Convert a SMILES string to its canonical isomeric representation.

        Args:
            smiles: Input valid SMILES string.

        Returns:
            Canonical SMILES string.

        Raises:
            InvalidSMILESError: If SMILES is invalid.
        """
        mol = MoleculeParser.parse_smiles(smiles)
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

