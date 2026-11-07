"""Comprehensive dataset inspection and validation script for OlfactAI.

Analyzes data quality, column schemas, missing values, duplicates,
invalid molecular structures, and conflicting odor annotations.
Generates an audit report without altering raw files.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app.chemistry.parser import MoleculeParser


def inspect_dataset(
    raw_csv_path: Path,
    output_report_path: Path = None,
) -> Dict[str, Any]:
    """Perform in-depth quality inspection of raw dataset."""
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_csv_path}")

    print(f"[INFO] Inspecting dataset: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)

    # Suppress verbose C++ logs from RDKit during bulk parse
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    total_records, total_columns = df.shape
    columns = list(df.columns)
    smiles_col = columns[0]
    desc_str_col = columns[1]
    odor_cols = columns[2:]

    # 1. Missing Values
    null_counts = df.isnull().sum()
    missing_by_col = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}
    total_missing_cells = int(null_counts.sum())

    # 2. Duplicate Records
    exact_duplicates_count = int(df.duplicated().sum())
    raw_smiles_duplicates_count = int(df[smiles_col].duplicated().sum())

    # 3. Structure Parsing & Canonicalization
    valid_count = 0
    invalid_records: List[Dict[str, Any]] = []
    canonical_smiles_list: List[str] = []

    parser = MoleculeParser()

    for idx, raw_smiles in enumerate(df[smiles_col]):
        if not isinstance(raw_smiles, str) or not raw_smiles.strip():
            invalid_records.append({
                "row_index": idx,
                "raw_smiles": raw_smiles,
                "reason": "Empty or non-string SMILES",
            })
            canonical_smiles_list.append(None)
            continue

        try:
            mol = parser.parse_smiles(raw_smiles)
            can_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            canonical_smiles_list.append(can_smi)
            valid_count += 1
        except Exception as exc:
            invalid_records.append({
                "row_index": idx,
                "raw_smiles": str(raw_smiles),
                "reason": str(exc),
            })
            canonical_smiles_list.append(None)

    df["_canonical_smiles"] = canonical_smiles_list

    # Valid structures stats
    valid_df = df[df["_canonical_smiles"].notnull()].copy()
    unique_canonical_count = int(valid_df["_canonical_smiles"].nunique())
    duplicate_canonical_count = int(len(valid_df) - unique_canonical_count)

    # 4. Conflicting Odor Annotations Analysis
    # Group by canonical SMILES and detect differences in odor columns
    canonical_groups = valid_df.groupby("_canonical_smiles")
    molecules_with_duplicates = 0
    molecules_with_conflicting_labels = 0
    conflict_details: List[Dict[str, Any]] = []

    for can_smi, group in canonical_groups:
        if len(group) > 1:
            molecules_with_duplicates += 1
            # Check if all binary label rows in this group are identical
            odor_sub_matrix = group[odor_cols].values
            is_identical = (odor_sub_matrix == odor_sub_matrix[0]).all()
            if not is_identical:
                molecules_with_conflicting_labels += 1
                # Find which descriptors differ
                diff_mask = odor_sub_matrix.max(axis=0) != odor_sub_matrix.min(axis=0)
                differing_descriptors = [odor_cols[i] for i, d in enumerate(diff_mask) if d]
                if len(conflict_details) < 10:  # store sample for reporting
                    conflict_details.append({
                        "canonical_smiles": can_smi,
                        "instance_count": len(group),
                        "row_indices": group.index.tolist(),
                        "differing_descriptors": differing_descriptors,
                    })

    # 5. Odor Descriptor Statistics
    label_frequencies = valid_df[odor_cols].sum().to_dict()
    # Sort by frequency
    sorted_labels = sorted(label_frequencies.items(), key=lambda x: x[1], reverse=True)
    top_10_labels = sorted_labels[:10]
    bottom_10_labels = sorted_labels[-10:]

    labels_per_mol = valid_df[odor_cols].sum(axis=1)
    zero_label_count = int((labels_per_mol == 0).sum())
    single_label_count = int((labels_per_mol == 1).sum())
    multi_label_count = int((labels_per_mol > 1).sum())

    report = {
        "dataset_file": raw_csv_path.name,
        "total_records": total_records,
        "total_columns": total_columns,
        "structure_column": smiles_col,
        "descriptor_string_column": desc_str_col,
        "odor_descriptor_count": len(odor_cols),
        "missing_values": {
            "total_missing_cells": total_missing_cells,
            "columns_with_missing_values": missing_by_col,
        },
        "duplicates": {
            "exact_duplicate_rows": exact_duplicates_count,
            "raw_smiles_duplicates": raw_smiles_duplicates_count,
            "canonical_smiles_duplicates": duplicate_canonical_count,
            "molecules_with_duplicate_records": molecules_with_duplicates,
            "molecules_with_conflicting_odor_labels": molecules_with_conflicting_labels,
            "sample_conflict_details": conflict_details,
        },
        "chemical_validation": {
            "valid_molecules": valid_count,
            "invalid_molecules": len(invalid_records),
            "unique_canonical_molecules": unique_canonical_count,
            "invalid_records_sample": invalid_records[:5],
        },
        "odor_annotations": {
            "zero_annotation_molecules": zero_label_count,
            "single_annotation_molecules": single_label_count,
            "multi_annotation_molecules": multi_label_count,
            "min_labels_per_molecule": int(labels_per_mol.min()),
            "max_labels_per_molecule": int(labels_per_mol.max()),
            "mean_labels_per_molecule": float(round(labels_per_mol.mean(), 3)),
            "median_labels_per_molecule": float(labels_per_mol.median()),
            "top_10_frequent_descriptors": top_10_labels,
            "bottom_10_rarest_descriptors": bottom_10_labels,
            "all_label_frequencies": {k: int(v) for k, v in label_frequencies.items()},
        },
    }

    # Print human-readable summary
    print("\n" + "=" * 60)
    print("DATASET QUALITY & VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Rows:                {total_records}")
    print(f"Total Columns:             {total_columns}")
    print(f"Odor Descriptors:          {len(odor_cols)}")
    print(f"Valid Molecules:           {valid_count} ({valid_count/total_records*100:.2f}%)")
    print(f"Invalid Molecules:         {len(invalid_records)}")
    print(f"Unique Canonical SMILES:   {unique_canonical_count}")
    print(f"Canonical Duplicates:      {duplicate_canonical_count}")
    print(f"Conflicting Odor Profiles: {molecules_with_conflicting_labels}")
    print(f"Molecules with 0 Odors:    {zero_label_count}")
    print(f"Molecules with >=1 Odors:  {valid_count - zero_label_count}")
    print(f"Avg Labels per Molecule:   {labels_per_mol.mean():.2f}")
    print(f"Missing Values in Data:    {total_missing_cells}")
    print("-" * 60)
    print("Top 5 Most Frequent Descriptors:")
    for lbl, count in top_10_labels[:5]:
        print(f"  - {lbl:<15}: {count} ({count/valid_count*100:.1f}%)")
    print("Top 5 Rarest Descriptors:")
    for lbl, count in bottom_10_labels[-5:]:
        print(f"  - {lbl:<15}: {count} ({count/valid_count*100:.2f}%)")
    print("=" * 60 + "\n")

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] Detailed inspection report saved to: {output_report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Inspect OlfactAI raw dataset.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=root_dir / "backend" / "datasets" / "raw" / "curated_GS_LF_merged_4983.csv",
        help="Path to raw CSV dataset.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=root_dir / "backend" / "datasets" / "processed" / "data_quality_report.json",
        help="Path to save JSON inspection report.",
    )
    args = parser.parse_args()
    inspect_dataset(args.input_csv, args.output_report)


if __name__ == "__main__":
    main()

