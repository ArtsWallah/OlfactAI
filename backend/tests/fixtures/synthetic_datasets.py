"""Synthetic olfactory dataset fixtures for fast, isolated automated testing."""

import pandas as pd


def get_synthetic_raw_df() -> pd.DataFrame:
    """Return a 10-molecule synthetic raw olfactory dataframe."""
    data = [
        {"nonStereoSMILES": "CCO", "descriptors": "alcoholic, sweet", "sweet": 1, "fruity": 0, "floral": 0, "woody": 0},
        {"nonStereoSMILES": "CC(=O)OCCC(C)C", "descriptors": "fruity, sweet", "sweet": 1, "fruity": 1, "floral": 0, "woody": 0},
        {"nonStereoSMILES": "COc1cc(C=O)ccc1O", "descriptors": "sweet, vanilla", "sweet": 1, "fruity": 0, "floral": 0, "woody": 0},
        {"nonStereoSMILES": "c1ccccc1O", "descriptors": "medicinal", "sweet": 0, "fruity": 0, "floral": 0, "woody": 0},
        {"nonStereoSMILES": "CC(C)CC(=O)O", "descriptors": "fruity, herbal", "sweet": 0, "fruity": 1, "floral": 0, "woody": 0},
        {"nonStereoSMILES": "c1ccccc1", "descriptors": "aromatic", "sweet": 0, "fruity": 0, "floral": 0, "woody": 0},
        {"nonStereoSMILES": "CCCC(=O)OCC", "descriptors": "fruity", "sweet": 0, "fruity": 1, "floral": 0, "woody": 0},
        {"nonStereoSMILES": "CCc1ccccc1", "descriptors": "woody, floral", "sweet": 0, "fruity": 0, "floral": 1, "woody": 1},
        {"nonStereoSMILES": "CCCCCC", "descriptors": "gasoline", "sweet": 0, "fruity": 0, "floral": 0, "woody": 0},
        {"nonStereoSMILES": "CC(C)=CCCC(C)=CCO", "descriptors": "floral, rose", "sweet": 0, "fruity": 0, "floral": 1, "woody": 0},
    ]
    return pd.DataFrame(data)


def get_synthetic_with_invalid_smiles_df() -> pd.DataFrame:
    """Return synthetic dataset containing invalid SMILES strings."""
    data = [
        {"nonStereoSMILES": "CCO", "descriptors": "alcoholic", "sweet": 1, "fruity": 0},
        {"nonStereoSMILES": "C1CC", "descriptors": "broken_ring", "sweet": 0, "fruity": 0},  # Invalid
        {"nonStereoSMILES": "Invalid_SMILES_XYZ", "descriptors": "unknown", "sweet": 0, "fruity": 0},  # Invalid
        {"nonStereoSMILES": "c1ccccc1", "descriptors": "aromatic", "sweet": 0, "fruity": 0},
    ]
    return pd.DataFrame(data)


def get_synthetic_with_duplicates_df() -> pd.DataFrame:
    """Return synthetic dataset containing duplicate canonical SMILES."""
    data = [
        {"nonStereoSMILES": "CCO", "descriptors": "alcoholic", "sweet": 1, "fruity": 0},
        {"nonStereoSMILES": "OCC", "descriptors": "sweet", "sweet": 1, "fruity": 1},  # Identical canonical SMILES CCO
        {"nonStereoSMILES": "c1ccccc1", "descriptors": "aromatic", "sweet": 0, "fruity": 0},
    ]
    return pd.DataFrame(data)

