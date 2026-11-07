"""Pydantic schemas for chemistry operations, molecular analysis, and validation."""

from typing import List, Optional
from pydantic import BaseModel, Field


class SMILESValidationRequest(BaseModel):
    """Payload for validating a chemical SMILES string."""

    smiles: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="SMILES string to validate",
        examples=["CC(=O)Oc1ccccc1C(=O)O", "CCO"],
    )


class SMILESValidationResponse(BaseModel):
    """Response returned from SMILES validation."""

    is_valid: bool = Field(..., description="Whether the SMILES represents a chemically valid structure")
    input_smiles: str = Field(..., description="Original input SMILES string")
    canonical_smiles: Optional[str] = Field(None, description="Canonical SMILES if valid")
    message: str = Field(..., description="Validation summary or error explanation")
    error_details: Optional[str] = Field(None, description="Detailed parser diagnostic if invalid")


class MolecularDescriptors(BaseModel):
    """Physicochemical molecular descriptors calculated via RDKit."""

    molecular_weight: float = Field(..., description="Exact or average molecular weight (g/mol)", examples=[180.16])
    molecular_formula: str = Field(..., description="Hill-system molecular formula", examples=["C9H8O4"])
    logp: float = Field(..., description="Octanol-water partition coefficient (Wildman-Crippen LogP)", examples=[1.31])
    tpsa: float = Field(..., description="Topological Polar Surface Area in Å²", examples=[63.60])
    h_bond_donors: int = Field(..., description="Count of hydrogen bond donors", examples=[1])
    h_bond_acceptors: int = Field(..., description="Count of hydrogen bond acceptors", examples=[4])
    rotatable_bonds: int = Field(..., description="Count of rotatable single bonds", examples=[3])
    ring_count: int = Field(..., description="Total count of rings in the molecule", examples=[1])
    heavy_atom_count: int = Field(..., description="Count of non-hydrogen atoms", examples=[13])
    fraction_csp3: float = Field(..., description="Fraction of sp3 hybridized carbons", examples=[0.11])


class FingerprintMetadata(BaseModel):
    """Metadata and representations for Morgan circular fingerprints."""

    radius: int = Field(2, description="Radius for Morgan fingerprint circular neighborhood")
    nbits: int = Field(2048, description="Fixed length of fingerprint bit vector")
    on_bits_count: int = Field(..., description="Number of bits set to 1 (sparse count)")
    on_bit_indices: List[int] = Field(
        default_factory=list,
        description="Sparse list of bit indices that are active (non-zero)",
    )
    bit_vector: Optional[List[int]] = Field(
        None,
        description="Full dense binary array of length nbits (only returned if include_fingerprint=true)",
    )


class MolecularAnalysisRequest(BaseModel):
    """Payload for molecular structure analysis."""

    smiles: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="SMILES string to analyze",
        examples=["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"],
    )
    include_fingerprint: bool = Field(
        False,
        description="If true, returns the dense 2048-bit binary vector in the response",
    )
    morgan_radius: int = Field(
        2,
        ge=1,
        le=4,
        description="Circular radius for Morgan fingerprint",
    )
    morgan_nbits: int = Field(
        2048,
        ge=64,
        le=8192,
        description="Number of bits for Morgan fingerprint bit vector",
    )


class MolecularAnalysisResponse(BaseModel):
    """Comprehensive chemical and structural analysis response."""

    input_smiles: str = Field(..., description="Original input SMILES")
    canonical_smiles: str = Field(..., description="Standardized canonical SMILES")
    is_valid: bool = Field(True, description="Indicates successful parsing and validation")
    descriptors: MolecularDescriptors = Field(..., description="Extracted physicochemical descriptors")
    fingerprint: FingerprintMetadata = Field(..., description="Calculated Morgan fingerprint information")

