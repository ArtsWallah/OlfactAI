"""Unit tests for chemistry SMILES parsing, validation, and canonicalization."""

import pytest
from backend.app.chemistry.parser import MoleculeParser
from backend.app.core.exceptions import InvalidSMILESError


class TestMoleculeParser:
    """Test suite for MoleculeParser operations."""

    def test_parse_valid_smiles(self, sample_molecules):
        """Verify successful parsing of valid chemical structures."""
        for name, data in sample_molecules.items():
            mol = MoleculeParser.parse_smiles(data["smiles"])
            assert mol is not None, f"Failed to parse valid SMILES for {name}"
            assert mol.GetNumAtoms() > 0

    @pytest.mark.parametrize("bad_smiles", [
        "NotAValidSMILES",
        "C1CC",
        "C=C=C=C=",
        "12345",
        "",
        "   ",
    ])
    def test_parse_invalid_smiles_raises_error(self, bad_smiles):
        """Verify invalid SMILES strings raise InvalidSMILESError."""
        with pytest.raises(InvalidSMILESError) as exc_info:
            MoleculeParser.parse_smiles(bad_smiles)
        assert exc_info.value.code == "INVALID_SMILES"

    def test_canonicalize_smiles(self):
        """Verify different representations of the same molecule produce identical canonical SMILES."""
        ethanol_variants = ["CCO", "OCC", "C(O)C"]
        canonicals = {MoleculeParser.canonicalize(s) for s in ethanol_variants}
        assert len(canonicals) == 1
        assert "CCO" in canonicals

    def test_canonicalize_aspirin(self, sample_molecules):
        """Verify canonical representation of aspirin."""
        aspirin_smiles = sample_molecules["aspirin"]["smiles"]
        canonical = MoleculeParser.canonicalize(aspirin_smiles)
        assert canonical == "CC(=O)Oc1ccccc1C(=O)O"

    def test_validate_smiles_success(self, sample_molecules):
        """Verify validate_smiles returns True and canonical SMILES for valid inputs."""
        for _, data in sample_molecules.items():
            is_valid, canonical, msg = MoleculeParser.validate_smiles(data["smiles"])
            assert is_valid is True
            assert canonical is not None
            assert len(canonical) > 0

    def test_validate_smiles_failure(self, invalid_smiles_list):
        """Verify validate_smiles returns False without raising for invalid inputs."""
        for bad in invalid_smiles_list:
            is_valid, canonical, msg = MoleculeParser.validate_smiles(bad)
            assert is_valid is False
            assert canonical is None
            assert msg is not None

