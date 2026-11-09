"""Script to run Phase 5 Molecular Neighborhood Analysis on CPU and export artifacts."""

import time
from pathlib import Path

from backend.app.similarity.engine import MolecularSimilarityEngine
from backend.app.similarity.neighborhood import MolecularNeighborhoodAnalyzer


def main():
    print("=" * 70)
    print("OlfactAI Phase 5 — Molecular Neighborhood & Pairwise Analysis")
    print("=" * 70)

    t_start = time.time()
    print("Initializing MolecularSimilarityEngine...")
    engine = MolecularSimilarityEngine(auto_load=True)
    print(f"Engine indexed {engine.corpus_size} molecules in {time.time() - t_start:.2f}s")

    analyzer = MolecularNeighborhoodAnalyzer(engine=engine)
    output_dir = Path("backend/experiments/phase5/neighborhoods")

    print("\nExecuting comprehensive pairwise and neighborhood study...")
    t_study = time.time()
    results = analyzer.run_full_neighborhood_study(
        output_dir=output_dir,
        n_bootstrap=100,
        seed=42,
    )
    study_duration = time.time() - t_study
    print(f"Study completed in {study_duration:.2f}s")

    glob_sum = results["global_summary"]
    boot_sum = results["bootstrap_summary"]

    print("\n--- Global Pairwise Findings (12,412,653 Unique Pairs) ---")
    print(f"Total Unique Molecule Pairs: {glob_sum['total_unique_pairs']:,}")
    print(f"Mean Structural Tanimoto:    {glob_sum['mean_tanimoto']:.4f}")
    print(f"Median Structural Tanimoto:  {glob_sum['median_tanimoto']:.4f}")
    print(f"Mean Odor Jaccard:           {glob_sum['mean_odor_jaccard']:.4f}")
    print(f"Median Odor Jaccard:         {glob_sum['median_odor_jaccard']:.4f}")
    print(f"Zero Odor Overlap Pairs:     {glob_sum['zero_odor_overlap_total_pairs']:,} ({glob_sum['zero_odor_overlap_overall_proportion']*100:.2f}%)")
    print(f"Descriptive Pearson r:       {glob_sum['pearson_correlation']:.4f}")
    print(f"Descriptive Spearman rho:    {glob_sum['spearman_correlation']:.4f}")

    print("\n--- Dependence-Aware Molecule-Level Cluster Bootstrap (B=100) ---")
    print(f"Resampling Unit:             {boot_sum['resampling_unit']}")
    print(f"Pearson r Bootstrap Mean:    {boot_sum['pearson_r_bootstrap_mean']:.4f} (SE = {boot_sum['pearson_r_bootstrap_se']:.4f})")
    print(f"Pearson r 95% CI:            [{boot_sum['pearson_r_ci_95'][0]:.4f}, {boot_sum['pearson_r_ci_95'][1]:.4f}]")
    print(f"Spearman rho Bootstrap Mean: {boot_sum['spearman_rho_bootstrap_mean']:.4f} (SE = {boot_sum['spearman_rho_bootstrap_se']:.4f})")
    print(f"Spearman rho 95% CI:         [{boot_sum['spearman_rho_ci_95'][0]:.4f}, {boot_sum['spearman_rho_ci_95'][1]:.4f}]")

    print("\n--- Binned Tanimoto Similarity vs Odor Jaccard Agreement ---")
    print(f"{'Tanimoto Bin':<18} | {'Pair Count':<12} | {'Mean Jaccard':<14} | {'Median Jacc':<12} | {'Zero Overlap %':<14}")
    print("-" * 80)
    for b in glob_sum["binned_analysis"]:
        print(f"{b['bin_label']:<18} | {b['pair_count']:<12,} | {b['mean_odor_jaccard']:<14.4f} | {b['median_odor_jaccard']:<12.4f} | {b['zero_overlap_proportion']*100:<14.2f}%")

    print(f"\nArtifacts successfully written to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

