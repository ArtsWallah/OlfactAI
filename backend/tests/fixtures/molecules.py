"""Chemical molecule fixtures for unit and integration testing."""

# Known test molecules with verified chemical properties
SAMPLE_MOLECULES = {
    "ethanol": {
        "name": "Ethanol",
        "smiles": "CCO",
        "canonical": "CCO",
        "formula": "C2H6O",
        "approx_mw": 46.07,
        "ring_count": 0,
        "h_donors": 1,
        "h_acceptors": 1,
    },
    "aspirin": {
        "name": "Aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "canonical": "CC(=O)Oc1ccccc1C(=O)O",
        "formula": "C9H8O4",
        "approx_mw": 180.16,
        "ring_count": 1,
        "h_donors": 1,
        "h_acceptors": 3,
        "rotatable_bonds": 2,
    },
    "vanillin": {
        "name": "Vanillin (sweet/vanilla odor)",
        "smiles": "COc1cc(C=O)ccc1O",
        "canonical": "COc1cc(C=O)ccc1O",
        "formula": "C8H8O3",
        "approx_mw": 152.15,
        "ring_count": 1,
        "h_donors": 1,
        "h_acceptors": 3,
    },
    "isoamyl_acetate": {
        "name": "Isoamyl Acetate (banana odor)",
        "smiles": "CC(=O)OCCC(C)C",
        "canonical": "CC(=O)OCCC(C)C",
        "formula": "C7H14O2",
        "approx_mw": 130.18,
        "ring_count": 0,
        "h_donors": 0,
        "h_acceptors": 2,
    },
}

# Invalid SMILES strings for negative testing
INVALID_SMILES = [
    "NotAValidSMILES",
    "C1CC",          # Unclosed ring
    "C=C=C=C=",      # Incomplete valence
    "12345",         # Numeric nonsense
    "P[O](O)(O)(O)O",# Invalid pentavalent coordination if uncorrected
    "",              # Empty string
    "   ",           # Whitespace string
]
