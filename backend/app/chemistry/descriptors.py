"""Extraction of physicochemical and structural descriptors from chemical molecules."""

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from backend.app.core.exceptions import MolecularProcessingError
from backend.app.schemas.chemistry import MolecularDescriptors


class DescriptorExtractor:
    """Extracts physicochemical and topological properties from RDKit Mol structures."""

    @staticmethod
    def extract_descriptors(mol: Chem.Mol) -> MolecularDescriptors:
        """Compute standard molecular descriptors for smell and fragrance modeling.

        Calculated descriptors include:
        - Molecular weight (MolWt)
        - Chemical formula (Hill system)
        - Octanol-water partition coefficient (MolLogP)
        - Topological Polar Surface Area (TPSA)
        - Hydrogen bond donors (NumHDonors)
        - Hydrogen bond acceptors (NumHAcceptors)
        - Rotatable bonds count (NumRotatableBonds)
        - Total ring count (RingCount)
        - Heavy atom count (NumHeavyAtoms)
        - Fraction of SP3 carbon atoms (FractionCSP3)

        Args:
            mol: Parsed and sanitized RDKit Mol object.

        Returns:
            MolecularDescriptors schema instance.

        Raises:
            MolecularProcessingError: If descriptor extraction fails.
        """
        if mol is None:
            raise MolecularProcessingError("Cannot extract descriptors from a null molecule.")

        try:
            mw = float(Descriptors.MolWt(mol))
            formula = rdMolDescriptors.CalcMolFormula(mol)
            logp = float(Descriptors.MolLogP(mol))
            tpsa = float(Descriptors.TPSA(mol))
            h_donors = int(Descriptors.NumHDonors(mol))
            h_acceptors = int(Descriptors.NumHAcceptors(mol))
            rotatable = int(Descriptors.NumRotatableBonds(mol))
            rings = int(Descriptors.RingCount(mol))
            heavy_atoms = int(mol.GetNumHeavyAtoms())
            fraction_csp3 = float(Descriptors.FractionCSP3(mol))

            return MolecularDescriptors(
                molecular_weight=round(mw, 4),
                molecular_formula=formula,
                logp=round(logp, 4),
                tpsa=round(tpsa, 4),
                h_bond_donors=h_donors,
                h_bond_acceptors=h_acceptors,
                rotatable_bonds=rotatable,
                ring_count=rings,
                heavy_atom_count=heavy_atoms,
                fraction_csp3=round(fraction_csp3, 4),
            )
        except Exception as e:
            raise MolecularProcessingError(
                message=f"Error extracting molecular descriptors: {str(e)}",
                details={"error": str(e)},
            ) from e

