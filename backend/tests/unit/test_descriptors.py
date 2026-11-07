"""Unit tests for physicochemical descriptor extraction."""

import pytest
from backend.app.chemistry.parser import MoleculeParser
from backend.app.chemistry.descriptors import DescriptorExtractor
from backend.app.core.exceptions import MolecularProcessingError


class TestDescriptorExtractor:
    """Test suite for RDKit molecular descriptor calculations."""

    def test_extract_descriptors_ethanol(self, sample_molecules):
        """Verify calculated descriptors for ethanol."""
        mol = MoleculeParser.parse_smiles(sample_molecules["ethanol"]["smiles"])
        desc = DescriptorExtractor.extract_descriptors(mol)

        assert desc.molecular_formula == "C2H6O"
        assert abs(desc.molecular_weight - 46.07) < 0.1
        assert desc.h_bond_donors == 1
        assert desc.h_bond_acceptors == 1
        assert desc.rotatable_bonds == 0
        assert desc.ring_count == 0
        assert desc.heavy_atom_count == 3
        assert desc.fraction_csp3 == 1.0

    def test_extract_descriptors_aspirin(self, sample_molecules):
        """Verify calculated descriptors for aspirin."""
        mol = MoleculeParser.parse_smiles(sample_molecules["aspirin"]["smiles"])
        desc = DescriptorExtractor.extract_descriptors(mol)

        assert desc.molecular_formula == "C9H8O4"
        assert abs(desc.molecular_weight - 180.16) < 0.1
        assert desc.ring_count == 1
        assert desc.h_bond_donors == 1
        assert desc.h_bond_acceptors == 3
        assert desc.rotatable_bonds == 2
        assert desc.heavy_atom_count == 13

    def test_extract_descriptors_null_mol_raises_error(self):
        """Verify passing null molecule raises MolecularProcessingError."""
        with pytest.raises(MolecularProcessingError):
            DescriptorExtractor.extract_descriptors(None)
