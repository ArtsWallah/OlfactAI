"""Molecular preprocessing pipeline for OlfactAI.

Extracts canonical SMILES, physicochemical descriptors, and 2048-bit Morgan
circular fingerprints using the RDKit chemistry engine.
Separates molecular feature matrix from the multi-label odor matrix.
Saves structured artifacts in backend/datasets/processed/.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app.chemistry.parser import MoleculeParser
from backend.app.chemistry.descriptors import DescriptorExtractor
from backend.app.chemistry.fingerprints import FingerprintGenerator


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def preprocess_dataset(
    raw_csv_path: Path,
    processed_dir: Path,
    morgan_radius: int = 2,
    morgan_nbits: int = 2048,
) -> Dict[str, Path]:
    """Execute end-to-end molecular feature extraction and matrix separation.

    Args:
        raw_csv_path: Path to the raw input CSV.
        processed_dir: Output directory for processed datasets.
        morgan_radius: Radius for Morgan circular fingerprint (default 2 -> ECFP4-equivalent).
        morgan_nbits: Length of the folded bit vector (default 2048).

    Returns:
        Dictionary mapping artifact names to their file paths.
    """
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_csv_path}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Reading raw dataset from: {raw_csv_path}")
    raw_df = pd.read_csv(raw_csv_path)

    # Suppress verbose RDKit C++ warnings
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    smiles_col = raw_df.columns[0]
    desc_str_col = raw_df.columns[1]
    odor_cols = list(raw_df.columns[2:])

    print(f"[INFO] Processing {len(raw_df)} molecules...")

    parser = MoleculeParser()
    extractor = DescriptorExtractor()
    fp_gen = FingerprintGenerator()

    molecules_records = []
    features_records = []
    labels_records = []
    failed_records = []

    fp_col_names = [f"fp_{i}" for i in range(morgan_nbits)]

    for idx, row in raw_df.iterrows():
        raw_smiles = row[smiles_col]
        mol_id = f"MOL_{idx + 1:05d}"

        try:
            mol = parser.parse_smiles(raw_smiles)
            canonical_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            desc = extractor.extract_descriptors(mol)
            fp_meta = fp_gen.generate_morgan_fingerprint(
                mol,
                radius=morgan_radius,
                nbits=morgan_nbits,
                include_dense_vector=True,
            )

            # 1. Molecules Metadata Record
            molecules_records.append({
                "molecule_id": mol_id,
                "input_smiles": raw_smiles,
                "canonical_smiles": canonical_smi,
                "molecular_formula": desc.molecular_formula,
                "descriptors_text": row[desc_str_col],
            })

            # 2. Physicochemical & Fingerprint Feature Record
            feature_row = {
                "molecule_id": mol_id,
                "canonical_smiles": canonical_smi,
                "molecular_weight": desc.molecular_weight,
                "logp": desc.logp,
                "tpsa": desc.tpsa,
                "h_bond_donors": desc.h_bond_donors,
                "h_bond_acceptors": desc.h_bond_acceptors,
                "rotatable_bonds": desc.rotatable_bonds,
                "ring_count": desc.ring_count,
                "heavy_atom_count": desc.heavy_atom_count,
                "fraction_csp3": desc.fraction_csp3,
            }
            # Append 2048 fingerprint bits
            for bit_i, bit_val in enumerate(fp_meta.bit_vector):
                feature_row[f"fp_{bit_i}"] = bit_val
            features_records.append(feature_row)

            # 3. Odor Labels Record (pure binary multi-label matrix)
            label_row = {
                "molecule_id": mol_id,
                "canonical_smiles": canonical_smi,
            }
            for col in odor_cols:
                label_row[col] = int(row[col])
            labels_records.append(label_row)

        except Exception as exc:
            failed_records.append({
                "row_index": idx,
                "input_smiles": raw_smiles,
                "error": str(exc),
            })

    print(f"[INFO] Successfully processed: {len(molecules_records)} / {len(raw_df)}")
    if failed_records:
        print(f"[WARNING] Failed structures: {len(failed_records)}")

    # Convert to DataFrames
    molecules_df = pd.DataFrame(molecules_records)
    features_df = pd.DataFrame(features_records)
    labels_df = pd.DataFrame(labels_records)

    # Save to CSV
    molecules_csv = processed_dir / "molecules.csv"
    features_csv = processed_dir / "molecular_features.csv"
    labels_csv = processed_dir / "odor_labels.csv"
    meta_json = processed_dir / "dataset_metadata.json"

    molecules_df.to_csv(molecules_csv, index=False)
    features_df.to_csv(features_csv, index=False)
    labels_df.to_csv(labels_csv, index=False)

    print(f"[SUCCESS] Saved molecules metadata:  {molecules_csv} ({molecules_df.shape})")
    print(f"[SUCCESS] Saved molecular features:  {features_csv} ({features_df.shape})")
    print(f"[SUCCESS] Saved odor labels matrix:  {labels_csv} ({labels_df.shape})")

    # Generate metadata description
    desc_cols = [
        "molecular_weight", "logp", "tpsa", "h_bond_donors",
        "h_bond_acceptors", "rotatable_bonds", "ring_count",
        "heavy_atom_count", "fraction_csp3",
    ]

    metadata = {
        "dataset_name": "OlfactAI Curated Olfactory Dataset",
        "provenance": {
            "source_raw_file": raw_csv_path.name,
            "source_sha256": compute_sha256(raw_csv_path),
            "processed_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "dimensions": {
            "total_molecules": len(molecules_df),
            "failed_molecules": len(failed_records),
            "feature_matrix_shape": list(features_df.shape),
            "label_matrix_shape": list(labels_df.shape),
            "odor_descriptors_count": len(odor_cols),
        },
        "fingerprint_parameters": {
            "type": "Morgan Circular Fingerprint (ECFP-equivalent)",
            "radius": morgan_radius,
            "nbits": morgan_nbits,
            "use_chirality": True,
        },
        "feature_specification": {
            "physicochemical_descriptors": desc_cols,
            "fingerprint_columns": [f"fp_0 ... fp_{morgan_nbits - 1}"],
            "total_feature_columns": len(features_df.columns) - 2,  # minus ID & canonical SMILES
        },
        "odor_descriptors": odor_cols,
        "artifact_checksums": {
            "molecules_csv": compute_sha256(molecules_csv),
            "molecular_features_csv": compute_sha256(features_csv),
            "odor_labels_csv": compute_sha256(labels_csv),
        },
    }

    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SUCCESS] Saved metadata descriptor: {meta_json}")

    return {
        "molecules": molecules_csv,
        "features": features_csv,
        "labels": labels_csv,
        "metadata": meta_json,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess OlfactAI molecular dataset.")
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=root_dir / "backend" / "datasets" / "raw" / "curated_GS_LF_merged_4983.csv",
        help="Path to raw CSV file.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=root_dir / "backend" / "datasets" / "processed",
        help="Directory to save processed datasets.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint circular radius.",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=2048,
        help="Number of Morgan fingerprint bits.",
    )
    args = parser.parse_args()
    preprocess_dataset(args.raw_csv, args.processed_dir, args.radius, args.nbits)


if __name__ == "__main__":
    main()

