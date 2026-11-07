"""High-level chemical intelligence service."""

from typing import Tuple, Optional
from backend.app.chemistry.parser import MoleculeParser
from backend.app.chemistry.descriptors import DescriptorExtractor
from backend.app.chemistry.fingerprints import FingerprintGenerator
from backend.app.schemas.chemistry import (
    MolecularAnalysisRequest,
    MolecularAnalysisResponse,
    SMILESValidationResponse,
)
from backend.app.core.logging import get_logger

logger = get_logger("chemistry.service")


class ChemistryService:
    """Service handling high-level molecular analysis workflows."""

    def __init__(self):
        self.parser = MoleculeParser()
        self.descriptor_extractor = DescriptorExtractor()
        self.fingerprint_generator = FingerprintGenerator()

    def validate_smiles(self, smiles: str) -> SMILESValidationResponse:
        """Validate input SMILES string and return validation details.

        Args:
            smiles: Input SMILES string.

        Returns:
            SMILESValidationResponse containing validation status and canonical form.
        """
        is_valid, canonical, msg = self.parser.validate_smiles(smiles)
        error_details = None if is_valid else msg

        return SMILESValidationResponse(
            is_valid=is_valid,
            input_smiles=smiles,
            canonical_smiles=canonical,
            message=msg,
            error_details=error_details,
        )

    def analyze_molecule(
        self, request: MolecularAnalysisRequest
    ) -> MolecularAnalysisResponse:
        """Parse molecule, extract physicochemical descriptors, and compute Morgan fingerprint.

        Args:
            request: MolecularAnalysisRequest containing SMILES and options.

        Returns:
            MolecularAnalysisResponse with full structure analysis.

        Raises:
            InvalidSMILESError: If SMILES string is invalid.
            MolecularProcessingError: If processing fails.
        """
        logger.info(f"Analyzing molecule with SMILES: '{request.smiles}'")

        # Parse and sanitize
        mol = self.parser.parse_smiles(request.smiles)
        canonical = self.parser.canonicalize(request.smiles)

        # Extract descriptors
        descriptors = self.descriptor_extractor.extract_descriptors(mol)

        # Generate fingerprint
        fingerprint = self.fingerprint_generator.generate_morgan_fingerprint(
            mol,
            radius=request.morgan_radius,
            nbits=request.morgan_nbits,
            include_dense_vector=request.include_fingerprint,
        )

        return MolecularAnalysisResponse(
            input_smiles=request.smiles,
            canonical_smiles=canonical,
            is_valid=True,
            descriptors=descriptors,
            fingerprint=fingerprint,
        )

