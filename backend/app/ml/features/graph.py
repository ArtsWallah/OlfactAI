"""Molecular graph representation and featurization for Graph Neural Networks.

Constructs rich, deterministic graph representations from chemical SMILES using RDKit:
- Atom features: atomic number, degree, formal charge, hybridization, aromaticity,
  hydrogens, ring membership, ring sizes, chirality.
- Bond features: bond type, conjugation, ring membership, stereochemistry.
- Bidirectional edge connectivity and disjoint-union graph batching.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
import torch

from backend.app.core.exceptions import InvalidSMILESError
from backend.app.core.logging import get_logger

logger = get_logger("ml.features.graph")

# One-hot encoding categories for deterministic mapping
ALLOWED_ATOMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H, B, C, N, O, F, Si, P, S, Cl, Br, I + other
ALLOWED_DEGREES = [0, 1, 2, 3, 4, 5]
ALLOWED_CHARGES = [-2, -1, 0, 1, 2]
ALLOWED_HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
ALLOWED_TOTAL_HS = [0, 1, 2, 3, 4]
ALLOWED_RING_SIZES = [3, 4, 5, 6, 7, 8]
ALLOWED_CHIRALITIES = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]

ALLOWED_BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]
ALLOWED_BOND_STEREO = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
]


def one_hot_encode(value: Any, allowable_set: List[Any], include_other: bool = True) -> List[float]:
    """Helper to one-hot encode a value against an allowable category list."""
    if include_other:
        vec = [1.0 if value == cat else 0.0 for cat in allowable_set]
        vec.append(1.0 if value not in allowable_set else 0.0)
        return vec
    return [1.0 if value == cat else 0.0 for cat in allowable_set]


@dataclass
class MolecularGraph:
    """Container for a single molecular graph."""

    atom_features: torch.Tensor  # Shape: (N_atoms, d_atom)
    edge_index: torch.Tensor     # Shape: (2, N_edges) - directed bidirectional
    bond_features: torch.Tensor  # Shape: (N_edges, d_bond)
    num_atoms: int
    num_bonds: int


@dataclass
class MolecularGraphBatch:
    """Disjoint union batch of multiple molecular graphs for parallel GNN execution."""

    atom_features: torch.Tensor  # Shape: (Total_atoms, d_atom)
    edge_index: torch.Tensor     # Shape: (2, Total_edges)
    bond_features: torch.Tensor  # Shape: (Total_edges, d_bond)
    batch_idx: torch.Tensor      # Shape: (Total_atoms,) - mapping atom to graph index
    num_graphs: int
    num_atoms_per_graph: torch.Tensor  # Shape: (num_graphs,)

    def to(self, device: torch.device) -> "MolecularGraphBatch":
        """Move batch tensors to device."""
        return MolecularGraphBatch(
            atom_features=self.atom_features.to(device),
            edge_index=self.edge_index.to(device),
            bond_features=self.bond_features.to(device),
            batch_idx=self.batch_idx.to(device),
            num_graphs=self.num_graphs,
            num_atoms_per_graph=self.num_atoms_per_graph.to(device),
        )


class MolecularGraphFeaturizer:
    """Converts chemical SMILES into deterministic atom and bond graph tensors."""

    # 50 atom features, 12 bond features
    ATOM_FEATURE_DIM: int = 50
    BOND_FEATURE_DIM: int = 12

    def __call__(self, smiles: str) -> MolecularGraph:
        return self.smiles_to_graph(smiles)

    @classmethod
    def get_atom_features(cls, atom: rdchem.Atom) -> List[float]:
        """Extract deterministic one-hot and binary atomic features."""
        features = []
        # 1. Atomic Number
        features.extend(one_hot_encode(atom.GetAtomicNum(), ALLOWED_ATOMS, include_other=True))
        # 2. Degree
        features.extend(one_hot_encode(atom.GetTotalDegree(), ALLOWED_DEGREES, include_other=True))
        # 3. Formal Charge
        features.extend(one_hot_encode(atom.GetFormalCharge(), ALLOWED_CHARGES, include_other=True))
        # 4. Hybridization
        features.extend(one_hot_encode(atom.GetHybridization(), ALLOWED_HYBRIDIZATIONS, include_other=True))
        # 5. Aromaticity
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        # 6. Total Hydrogens
        features.extend(one_hot_encode(atom.GetTotalNumHs(), ALLOWED_TOTAL_HS, include_other=True))
        # 7. In Ring
        features.append(1.0 if atom.IsInRing() else 0.0)
        # 8. Ring Size Indicators
        for size in ALLOWED_RING_SIZES:
            features.append(1.0 if atom.IsInRingSize(size) else 0.0)
        # 9. Chirality
        features.extend(one_hot_encode(atom.GetChiralTag(), ALLOWED_CHIRALITIES, include_other=True))

        return features

    @classmethod
    def get_bond_features(cls, bond: rdchem.Bond) -> List[float]:
        """Extract deterministic one-hot and binary bond features."""
        features = []
        # 1. Bond Type
        features.extend(one_hot_encode(bond.GetBondType(), ALLOWED_BOND_TYPES, include_other=True))
        # 2. Conjugation
        features.append(1.0 if bond.GetIsConjugated() else 0.0)
        # 3. In Ring
        features.append(1.0 if bond.IsInRing() else 0.0)
        # 4. Stereochemistry
        features.extend(one_hot_encode(bond.GetStereo(), ALLOWED_BOND_STEREO, include_other=True))

        return features

    @classmethod
    def smiles_to_graph(cls, smiles: str) -> MolecularGraph:
        """Parse SMILES string and return MolecularGraph container.

        Raises:
            InvalidSMILESError: If SMILES string cannot be parsed into a valid RDKit molecule.
        """
        if not smiles or not isinstance(smiles, str) or smiles.strip() == "":
            raise InvalidSMILESError(
                message="Empty or non-string SMILES supplied",
                smiles=str(smiles),
            )

        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            raise InvalidSMILESError(
                message="RDKit MolFromSmiles returned None (parsing failure)",
                smiles=smiles,
            )

        # Ensure atom orders are canonical and consistent
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            raise InvalidSMILESError(
                message="Molecule has 0 atoms",
                smiles=smiles,
            )

        # 1. Node/Atom features
        atom_feats = [cls.get_atom_features(atom) for atom in mol.GetAtoms()]
        atom_tensor = torch.tensor(atom_feats, dtype=torch.float32)

        # 2. Edge indices and Edge/Bond features
        edge_src: List[int] = []
        edge_dst: List[int] = []
        bond_feats: List[List[float]] = []

        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            b_feat = cls.get_bond_features(bond)

            # Bidirectional directed graph edges
            edge_src.extend([u, v])
            edge_dst.extend([v, u])
            bond_feats.extend([b_feat, b_feat])

        # Handle isolated mono-atomic molecules (e.g. single ion, noble gas, methane if hydrogens implicit)
        if len(edge_src) == 0:
            # Add self-loops for isolated atoms with zero bond features
            for i in range(num_atoms):
                edge_src.append(i)
                edge_dst.append(i)
                bond_feats.append([0.0] * cls.BOND_FEATURE_DIM)

        edge_index_tensor = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        bond_tensor = torch.tensor(bond_feats, dtype=torch.float32)

        return MolecularGraph(
            atom_features=atom_tensor,
            edge_index=edge_index_tensor,
            bond_features=bond_tensor,
            num_atoms=num_atoms,
            num_bonds=len(edge_src),
        )

    @classmethod
    def collate_graphs(cls, graph_list: List[MolecularGraph]) -> MolecularGraphBatch:
        """Batch a list of MolecularGraph objects into a single disjoint union graph."""
        if not graph_list:
            raise ValueError("Cannot collate empty list of molecular graphs")

        atom_features_list: List[torch.Tensor] = []
        edge_index_list: List[torch.Tensor] = []
        bond_features_list: List[torch.Tensor] = []
        batch_idx_list: List[torch.Tensor] = []
        num_atoms_list: List[int] = []

        atom_offset = 0

        for g_idx, g in enumerate(graph_list):
            atom_features_list.append(g.atom_features)
            # Shift edge indices by cumulative atom offset
            edge_index_list.append(g.edge_index + atom_offset)
            bond_features_list.append(g.bond_features)

            batch_idx_list.append(torch.full((g.num_atoms,), g_idx, dtype=torch.long))
            num_atoms_list.append(g.num_atoms)
            atom_offset += g.num_atoms

        return MolecularGraphBatch(
            atom_features=torch.cat(atom_features_list, dim=0),
            edge_index=torch.cat(edge_index_list, dim=1),
            bond_features=torch.cat(bond_features_list, dim=0),
            batch_idx=torch.cat(batch_idx_list, dim=0),
            num_graphs=len(graph_list),
            num_atoms_per_graph=torch.tensor(num_atoms_list, dtype=torch.long),
        )
