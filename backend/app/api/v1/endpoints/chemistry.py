"""Chemistry API endpoints for molecular analysis, validation, and property extraction."""

from fastapi import APIRouter, Depends, status
from backend.app.chemistry.service import ChemistryService
from backend.app.schemas.chemistry import (
    MolecularAnalysisRequest,
    MolecularAnalysisResponse,
    SMILESValidationRequest,
    SMILESValidationResponse,
)

router = APIRouter(prefix="/chemistry", tags=["Chemistry"])


def get_chemistry_service() -> ChemistryService:
    """Dependency injection provider for ChemistryService."""
    return ChemistryService()


@router.post(
    "/analyze",
    response_model=MolecularAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze chemical molecular structure",
    description=(
        "Parses a SMILES string, verifies chemical validity, computes canonical representation, "
        "extracts 10 physicochemical descriptors, and computes a Morgan circular fingerprint."
    ),
)
def analyze_molecule(
    request: MolecularAnalysisRequest,
    service: ChemistryService = Depends(get_chemistry_service),
) -> MolecularAnalysisResponse:
    """Analyze a single molecule by SMILES representation."""
    return service.analyze_molecule(request)


@router.post(
    "/validate",
    response_model=SMILESValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate chemical SMILES string",
    description="Validates whether a SMILES string is syntactically and chemically legitimate.",
)
def validate_smiles(
    request: SMILESValidationRequest,
    service: ChemistryService = Depends(get_chemistry_service),
) -> SMILESValidationResponse:
    """Validate SMILES string validity and return canonical form if valid."""
    return service.validate_smiles(request.smiles)

