"""Script to generate reproducible Phase 5 UMAP embeddings and visualizations."""

import time
from pathlib import Path

from backend.app.clustering.umap_embeddings import MolecularUMAP


def main():
    print("=" * 70)
    print("OlfactAI Phase 5 — UMAP Chemical Space Dimensionality Reduction")
    print("=" * 70)

    t0 = time.time()
    umap_runner = MolecularUMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="jaccard",
        random_state=42,
    )

    dataset_dir = Path("backend/datasets/processed")
    embeddings_dir = Path("backend/experiments/phase5/embeddings")
    plots_dir = Path("backend/experiments/phase5/plots")

    print("\nRunning complete UMAP embedding and visualization pipeline...")
    results = umap_runner.run_full_pipeline(
        dataset_dir=dataset_dir,
        embeddings_dir=embeddings_dir,
        plots_dir=plots_dir,
    )

    duration = time.time() - t0
    meta = results["metadata"]

    print("\n" + "=" * 70)
    print(f"Pipeline Completed in {duration:.2f}s")
    print("=" * 70)

    print("\n--- Primary 2D Morgan Fingerprint UMAP ---")
    m2 = meta["umap_2d_fingerprints"]
    print(f"Dimensions:      {m2['corpus_size']} molecules x {m2['n_components']} components")
    print(f"Metric / Seed:   {m2['metric']} / random_state={m2['random_state']}")
    print(f"Neighbors / Dist:{m2['n_neighbors']} / min_dist={m2['min_dist']}")
    print(f"Execution Time:  {m2['execution_time_seconds']:.2f}s")
    print(f"File Saved:      {results['umap_2d_path']}")

    print("\n--- Intermediate 10D Morgan Fingerprint UMAP (Step 6 Clustering Input) ---")
    m10 = meta["umap_10d_fingerprints"]
    print(f"Dimensions:      {m10['corpus_size']} molecules x {m10['n_components']} components")
    print(f"Execution Time:  {m10['execution_time_seconds']:.2f}s")
    print(f"File Saved:      {results['umap_10d_path']}")

    print("\n--- Secondary Experiment: 2D Physicochemical UMAP ---")
    mp = meta["umap_2d_physicochemical"]
    print(f"Dimensions:      {mp['corpus_size']} molecules x {mp['n_components']} components")
    print(f"Features:        9 Standardized Descriptors (StandardScaler)")
    print(f"Execution Time:  {mp['execution_time_seconds']:.2f}s")
    print(f"File Saved:      {results['phys_umap_path']}")

    print("\n--- Generated Publication Figures ---")
    for name, p in results["plots"].items():
        print(f"  [{name:<26}] -> {p}")

    print("\nMetadata File:")
    print(f"  {results['metadata_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

