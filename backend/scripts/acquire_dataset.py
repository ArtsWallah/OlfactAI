"""Dataset acquisition script for OlfactAI.

Downloads the official curated GoodScents & Leffingwell merged dataset
from the BioMachineLearning/openpom repository (Lee et al. 2023 replication).
Ensures atomic download, checksum verification, provenance tracking,
and prevents accidental overwriting.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import httpx

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

DATASET_CONFIG = {
    "name": "curated_GS_LF_merged_4983.csv",
    "primary_url": (
        "https://raw.githubusercontent.com/BioMachineLearning/openpom/"
        "main/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
    ),
    "repository_url": "https://github.com/BioMachineLearning/openpom",
    "commit_sha": "5434c4b7badebca8e7d180aa34d2f787402121ee",
    "expected_sha256": "0c18b6e9f0f99b772203ba4da30a96b3d60db700349376cbafe23512a9baa704",
    "expected_bytes": 1647971,
    "license": "MIT License (Aryan Amit Barsainyan 2023)",
    "scientific_reference": (
        "Brian K. Lee et al., 'A principal odor map unifies diverse tasks in "
        "human olfactory perception', Science 381, 999-1006 (2023). "
        "DOI: 10.1126/science.ade4401"
    ),
}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a local file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def acquire_dataset(
    output_dir: Path,
    force: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Acquire the raw olfactory benchmark dataset.

    Args:
        output_dir: Directory where raw dataset will be saved.
        force: Overwrite existing file if True.
        timeout: HTTP request timeout in seconds.

    Returns:
        Path to the saved raw dataset file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / DATASET_CONFIG["name"]
    meta_path = output_dir / "raw_dataset_metadata.json"

    if target_path.exists() and not force:
        existing_sha = compute_sha256(target_path)
        print(f"[INFO] Dataset already exists at: {target_path}")
        print(f"[INFO] Existing file size: {target_path.stat().st_size} bytes")
        print(f"[INFO] Existing SHA-256:   {existing_sha}")

        if existing_sha == DATASET_CONFIG["expected_sha256"]:
            print("[INFO] Integrity verified against expected SHA-256. Skipping download.")
            return target_path
        else:
            print("[WARNING] Existing file SHA-256 mismatch! Use --force to re-download.")
            return target_path

    print(f"[INFO] Downloading official dataset from: {DATASET_CONFIG['primary_url']}")
    temp_target = target_path.with_suffix(".tmp")

    try:
        headers = {"User-Agent": "OlfactAI-Data-Acquisition/1.0"}
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            with client.stream("GET", DATASET_CONFIG["primary_url"], headers=headers) as response:
                response.raise_for_status()
                with open(temp_target, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)

        # Verify checksum before atomic rename
        actual_sha = compute_sha256(temp_target)
        actual_bytes = temp_target.stat().st_size

        if actual_sha != DATASET_CONFIG["expected_sha256"]:
            raise ValueError(
                f"Checksum verification failed!\nExpected: {DATASET_CONFIG['expected_sha256']}\n"
                f"Actual:   {actual_sha}"
            )

        # Atomic replacement
        if target_path.exists():
            target_path.unlink()
        temp_target.rename(target_path)
        print(f"[SUCCESS] Download completed and verified: {target_path}")

        # Record provenance metadata
        metadata = {
            "dataset_name": DATASET_CONFIG["name"],
            "source_url": DATASET_CONFIG["primary_url"],
            "repository": DATASET_CONFIG["repository_url"],
            "commit_sha": DATASET_CONFIG["commit_sha"],
            "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "file_size_bytes": actual_bytes,
            "sha256_checksum": actual_sha,
            "license": DATASET_CONFIG["license"],
            "scientific_reference": DATASET_CONFIG["scientific_reference"],
            "status": "verified_authentic",
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[INFO] Provenance metadata saved to: {meta_path}")
        return target_path

    except Exception as e:
        if temp_target.exists():
            temp_target.unlink()
        print(f"[ERROR] Failed to acquire dataset: {e}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description="Acquire raw OpenPOM olfactory dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "backend" / "datasets" / "raw",
        help="Directory to store raw dataset files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset already exists.",
    )
    args = parser.parse_args()

    acquire_dataset(output_dir=args.output_dir, force=args.force)


if __name__ == "__main__":
    main()

