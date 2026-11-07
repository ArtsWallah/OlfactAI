"""Unit tests for circular Morgan fingerprint generation."""

import pytest
from backend.app.chemistry.parser import MoleculeParser
from backend.app.chemistry.fingerprints import FingerprintGenerator
from backend.app.core.exceptions import MolecularProcessingError


class TestFingerprintGenerator:
    """Test suite for Morgan fingerprint calculations."""

    def test_default_morgan_dimensions(self, sample_molecules):
        """Verify default 2048-bit Morgan fingerprint dimensions."""
        mol = MoleculeParser.parse_smiles(sample_molecules["vanillin"]["smiles"])
        fp_meta = FingerprintGenerator.generate_morgan_fingerprint(
            mol, radius=2, nbits=2048, include_dense_vector=False
        )

        assert fp_meta.radius == 2
        assert fp_meta.nbits == 2048
        assert fp_meta.on_bits_count > 0
        assert len(fp_meta.on_bit_indices) == fp_meta.on_bits_count
        # Full dense vector should be omitted by default
        assert fp_meta.bit_vector is None

    def test_include_dense_vector(self, sample_molecules):
        """Verify full dense bit vector is returned when explicitly requested."""
        mol = MoleculeParser.parse_smiles(sample_molecules["ethanol"]["smiles"])
        fp_meta = FingerprintGenerator.generate_morgan_fingerprint(
            mol, radius=2, nbits=2048, include_dense_vector=True
        )

        assert fp_meta.bit_vector is not None
        assert len(fp_meta.bit_vector) == 2048
        assert sum(fp_meta.bit_vector) == fp_meta.on_bits_count
        # Verify indices align with on_bit_indices
        for idx in fp_meta.on_bit_indices:
            assert fp_meta.bit_vector[idx] == 1

    def test_custom_nbits(self, sample_molecules):
        """Verify configurable nbits length."""
        mol = MoleculeParser.parse_smiles(sample_molecules["aspirin"]["smiles"])
        fp_meta = FingerprintGenerator.generate_morgan_fingerprint(
            mol, radius=3, nbits=1024, include_dense_vector=True
        )

        assert fp_meta.radius == 3
        assert fp_meta.nbits == 1024
        assert len(fp_meta.bit_vector) == 1024

    def test_null_molecule_raises_error(self):
        """Verify null molecule raises MolecularProcessingError."""
        with pytest.raises(MolecularProcessingError):
            FingerprintGenerator.generate_morgan_fingerprint(None)

