"""Software-simulated Electronic Nose (e-Nose) cross-reactive virtual sensor array."""

from typing import Dict, List
import numpy as np

from backend.app.chemistry.parser import MoleculeParser
from backend.app.chemistry.descriptors import DescriptorExtractor


class VirtualElectronicNose:
    """Simulates a virtual multi-channel electronic nose sensor array on CPU.

    Strictly software-simulated: Models cross-reactive sensor response curves
    based on molecular weight, volatility, polarity, and electronic properties.
    """

    def __init__(self, n_channels: int = 16, random_seed: int = 42):
        self.n_channels = n_channels
        self.parser = MoleculeParser()
        self.descriptor_extractor = DescriptorExtractor()
        # Initialize virtual sensor affinity matrix (n_channels, n_descriptors)
        rng = np.random.RandomState(random_seed)
        self.sensor_weights = rng.normal(loc=0.0, scale=1.0, size=(self.n_channels, 6))

    def simulate_response(self, smiles: str) -> Dict[str, float]:
        """Simulate virtual sensor channel response voltages/signals for a molecule."""
        mol = self.parser.parse_smiles(smiles)
        desc = self.descriptor_extractor.extract_descriptors(mol)

        # Normalized property vector: [MW, LogP, TPSA, HBD, HBA, RotB]
        raw_props = np.array([
            desc.molecular_weight / 300.0,
            (desc.logp + 2.0) / 6.0,
            desc.tpsa / 150.0,
            desc.h_bond_donors / 5.0,
            desc.h_bond_acceptors / 10.0,
            desc.rotatable_bonds / 10.0,
        ], dtype=np.float32)

        # Dot product with sensor affinity weights passed through non-linear response curve
        signals = np.tanh(np.dot(self.sensor_weights, raw_props))

        return {f"channel_{i+1:02d}": float(round(signals[i], 4)) for i in range(self.n_channels)}

