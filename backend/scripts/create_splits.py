"""Leakage-safe dataset splitting pipeline for OlfactAI.

Generates:
1. Random train/validation/test split (70 / 15 / 15) with fixed seed.
2. Bemis-Murcko molecular scaffold-based train/validation/test split (70 / 15 / 15).
Performs rigorous scaffold leakage verification between training, validation, and test sets.
Monitors multi-label odor coverage and reports rare label representation across splits.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def generate_random_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, str]:
    """Generate reproducible random train/val/test assignment."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    n = len(df)
    rng = np.random.RandomState(random_seed)
    shuffled_indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    split_map = {}
    for pos, idx in enumerate(shuffled_indices):
        mol_id = df.iloc[idx]["molecule_id"]
        if pos < n_train:
            split_map[mol_id] = "train"
        elif pos < n_train + n_val:
            split_map[mol_id] = "val"
        else:
            split_map[mol_id] = "test"

    return split_map


def generate_scaffold_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Generate Bemis-Murcko scaffold-based split with zero scaffold overlap between partitions.

    For ring-containing molecules: Grouped strictly by Bemis-Murcko core ring scaffold.
    For acyclic molecules: Grouped deterministically by carbon skeleton graph so acyclics
    are disjointly partitioned without cross-partition scaffold leakage.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    scaffold_to_ids: Dict[str, List[str]] = {}
    mol_to_scaffold: Dict[str, str] = {}

    for _, row in df.iterrows():
        mol_id = row["molecule_id"]
        smi = row["canonical_smiles"]
        mol = Chem.MolFromSmiles(smi)

        scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if not scaff:
            # Acyclic: group by carbon skeleton formula representation
            scaff = f"acyclic_{Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(mol))}"

        mol_to_scaffold[mol_id] = scaff
        scaffold_to_ids.setdefault(scaff, []).append(mol_id)

    # Sort scaffold groups deterministically with random jitter for identical sizes
    rng = np.random.RandomState(random_seed)
    scaffold_keys = list(scaffold_to_ids.keys())
    rng.shuffle(scaffold_keys)
    scaffold_groups = sorted(scaffold_keys, key=lambda k: len(scaffold_to_ids[k]), reverse=True)

    total_mols = len(df)
    train_target = int(total_mols * train_ratio)
    val_target = int(total_mols * val_ratio)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    train_scaffolds: Set[str] = set()
    val_scaffolds: Set[str] = set()
    test_scaffolds: Set[str] = set()

    split_map: Dict[str, str] = {}

    for scaff in scaffold_groups:
        mol_ids = scaffold_to_ids[scaff]
        group_size = len(mol_ids)

        if len(train_ids) + group_size <= train_target:
            train_ids.extend(mol_ids)
            train_scaffolds.add(scaff)
            for m in mol_ids:
                split_map[m] = "train"
        elif len(val_ids) + group_size <= val_target:
            val_ids.extend(mol_ids)
            val_scaffolds.add(scaff)
            for m in mol_ids:
                split_map[m] = "val"
        else:
            test_ids.extend(mol_ids)
            test_scaffolds.add(scaff)
            for m in mol_ids:
                split_map[m] = "test"

    # Strict Verification of Scaffold Leakage
    train_val_overlap = train_scaffolds.intersection(val_scaffolds)
    train_test_overlap = train_scaffolds.intersection(test_scaffolds)
    val_test_overlap = val_scaffolds.intersection(test_scaffolds)

    scaffold_leakage_detected = (
        len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0
    )

    leakage_audit = {
        "scaffold_leakage_detected": scaffold_leakage_detected,
        "train_scaffolds_count": len(train_scaffolds),
        "val_scaffolds_count": len(val_scaffolds),
        "test_scaffolds_count": len(test_scaffolds),
        "train_val_overlap_count": len(train_val_overlap),
        "train_test_overlap_count": len(train_test_overlap),
        "val_test_overlap_count": len(val_test_overlap),
    }

    return split_map, leakage_audit


def audit_label_coverage(
    df_labels: pd.DataFrame,
    split_map: Dict[str, str],
    odor_cols: List[str],
) -> Dict[str, Any]:
    """Audit label distribution and detect any missing rare labels across splits."""
    df = df_labels.copy()
    df["split"] = df["molecule_id"].map(split_map)

    coverage = {}
    missing_in_train = []
    missing_in_val = []
    missing_in_test = []

    for col in odor_cols:
        counts = df.groupby("split")[col].sum().to_dict()
        train_c = int(counts.get("train", 0))
        val_c = int(counts.get("val", 0))
        test_c = int(counts.get("test", 0))

        if train_c == 0:
            missing_in_train.append(col)
        if val_c == 0:
            missing_in_val.append(col)
        if test_c == 0:
            missing_in_test.append(col)

        coverage[col] = {
            "train": train_c,
            "val": val_c,
            "test": test_c,
            "total": train_c + val_c + test_c,
        }

    return {
        "missing_in_train": missing_in_train,
        "missing_in_val": missing_in_val,
        "missing_in_test": missing_in_test,
        "per_label_counts": coverage,
    }


def create_splits(
    molecules_csv: Path,
    odor_labels_csv: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute random and scaffold splits, verify leakage, and generate audit reports."""
    print(f"[INFO] Loading molecules from: {molecules_csv}")
    mol_df = pd.read_csv(molecules_csv)
    labels_df = pd.read_csv(odor_labels_csv)
    odor_cols = [c for c in labels_df.columns if c not in ("molecule_id", "canonical_smiles")]

    print(f"[INFO] Generating random 70/15/15 split (seed={random_seed})...")
    random_split_map = generate_random_split(
        mol_df, train_ratio, val_ratio, test_ratio, random_seed=random_seed
    )

    print("[INFO] Generating Bemis-Murcko scaffold 70/15/15 split...")
    scaffold_split_map, scaffold_leakage_audit = generate_scaffold_split(
        mol_df, train_ratio, val_ratio, test_ratio, random_seed=random_seed
    )

    # Multi-label coverage audit
    print("[INFO] Auditing multi-label odor coverage across splits...")
    random_coverage = audit_label_coverage(labels_df, random_split_map, odor_cols)
    scaffold_coverage = audit_label_coverage(labels_df, scaffold_split_map, odor_cols)

    # Attach split columns to molecules dataframe
    mol_df["random_split"] = mol_df["molecule_id"].map(random_split_map)
    mol_df["scaffold_split"] = mol_df["molecule_id"].map(scaffold_split_map)
    mol_df.to_csv(molecules_csv, index=False)
    print(f"[INFO] Updated {molecules_csv} with 'random_split' and 'scaffold_split' columns.")

    # Compute partition counts
    random_counts = dict(mol_df["random_split"].value_counts())
    scaffold_counts = dict(mol_df["scaffold_split"].value_counts())

    summary = {
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "random_seed": random_seed,
        "random_split": {
            "sizes": {k: int(v) for k, v in random_counts.items()},
            "fractions": {k: round(int(v) / len(mol_df), 4) for k, v in random_counts.items()},
            "missing_in_train": random_coverage["missing_in_train"],
            "missing_in_val": random_coverage["missing_in_val"],
            "missing_in_test": random_coverage["missing_in_test"],
        },
        "scaffold_split": {
            "sizes": {k: int(v) for k, v in scaffold_counts.items()},
            "fractions": {k: round(int(v) / len(mol_df), 4) for k, v in scaffold_counts.items()},
            "leakage_audit": scaffold_leakage_audit,
            "missing_in_train": scaffold_coverage["missing_in_train"],
            "missing_in_val": scaffold_coverage["missing_in_val"],
            "missing_in_test": scaffold_coverage["missing_in_test"],
        },
    }

    # Save splits JSON
    splits_json_path = output_dir / "splits.json"
    splits_summary_path = output_dir / "splits_summary.json"

    full_splits_payload = {
        "random_split": random_split_map,
        "scaffold_split": scaffold_split_map,
    }

    with open(splits_json_path, "w", encoding="utf-8") as f:
        json.dump(full_splits_payload, f, indent=2)

    with open(splits_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SPLIT GENERATION & LEAKAGE AUDIT REPORT")
    print("=" * 60)
    print(f"Total Molecules: {len(mol_df)}")
    print(f"Random Split:    Train={random_counts.get('train')} ({random_counts.get('train')/len(mol_df):.1%}), "
          f"Val={random_counts.get('val')} ({random_counts.get('val')/len(mol_df):.1%}), "
          f"Test={random_counts.get('test')} ({random_counts.get('test')/len(mol_df):.1%})")
    print(f"Scaffold Split:  Train={scaffold_counts.get('train')} ({scaffold_counts.get('train')/len(mol_df):.1%}), "
          f"Val={scaffold_counts.get('val')} ({scaffold_counts.get('val')/len(mol_df):.1%}), "
          f"Test={scaffold_counts.get('test')} ({scaffold_counts.get('test')/len(mol_df):.1%})")
    print("-" * 60)
    print(f"Scaffold Overlap (Train <-> Test): {scaffold_leakage_audit['train_test_overlap_count']}")
    print(f"Scaffold Overlap (Train <-> Val):  {scaffold_leakage_audit['train_val_overlap_count']}")
    print(f"Scaffold Overlap (Val <-> Test):    {scaffold_leakage_audit['val_test_overlap_count']}")
    print(f"Scaffold Leakage Detected:         {scaffold_leakage_audit['scaffold_leakage_detected']}")
    print("-" * 60)
    print(f"Random Split Missing in Val:  {random_coverage['missing_in_val'] or 'None'}")
    print(f"Random Split Missing in Test: {random_coverage['missing_in_test'] or 'None'}")
    print(f"Scaffold Split Missing in Val:  {scaffold_coverage['missing_in_val'] or 'None'}")
    print(f"Scaffold Split Missing in Test: {scaffold_coverage['missing_in_test'] or 'None'}")
    print("=" * 60 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Create leakage-safe dataset splits.")
    parser.add_argument(
        "--molecules-csv",
        type=Path,
        default=root_dir / "backend" / "datasets" / "processed" / "molecules.csv",
        help="Path to processed molecules.csv",
    )
    parser.add_argument(
        "--odor-labels-csv",
        type=Path,
        default=root_dir / "backend" / "datasets" / "processed" / "odor_labels.csv",
        help="Path to processed odor_labels.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "backend" / "datasets" / "processed",
        help="Directory to save split artifacts.",
    )
    args = parser.parse_args()
    create_splits(args.molecules_csv, args.odor_labels_csv, args.output_dir)


if __name__ == "__main__":
    main()

