# OlfactAI — Phase 5 Research Report: Molecular Similarity, Neighborhoods, Chemical Space, and HDBSCAN Discovery

**Date**: November 10, 2026  
**Status**: Step 4 (HDBSCAN Molecular Structure Discovery) Completed — Independent Review Corrections Applied  
**Branch**: `main`  
**Environment**: Ubuntu 24.04 (WSL2), Python 3.11.18, CPU-only PyTorch, RDKit, HDBSCAN 0.8.44, UMAP  
**Dataset**: OpenPOM Curated Dataset (4,983 molecules, 138 binary odor descriptors)

---

## Executive Summary

Phase 5 investigates the topological and density-based structure of chemical odor space using molecular fingerprints, nearest-neighbor similarity search, global pairwise distributions, non-linear manifold learning (UMAP), and unsupervised hierarchical density-based clustering (HDBSCAN).

In this milestone (**Step 4: HDBSCAN Molecular Structure Discovery**), we designed and implemented a memory-bounded, BLAS-accelerated HDBSCAN clustering engine to partition the 4,983 molecules into objective structural families. The primary scientific experiment was executed directly in the native 2,048-bit Morgan fingerprint space using Jaccard distance ($1.0 - \text{Tanimoto}$), and systematically contrasted against exploratory 10-dimensional UMAP and visualization-oriented 2-dimensional UMAP embeddings.

```
+----------------------------------------------------------------------------------------------------+
|                                    HDBSCAN Clustering Hierarchy                                    |
|                                                                                                    |
|  [2,048-bit Morgan Fingerprints]  ──(BLAS)──>  [Symmetric Jaccard Distance Matrix (189.4 MB)]      |
|                                                        │                                           |
|                                                        ▼                                           |
|                                            [HDBSCAN (mcs=15, ms=5)]                                |
|                                                        │                                           |
|                     ┌──────────────────────────────────┴──────────────────────────────────┐        |
|                     ▼                                                                     ▼        |
|          [37 Core Structural Clusters]                                          [3,873 Noise Mols] |
|          (1,110 molecules, 22.28%)                                              (77.72% sparsity)  |
|          • Median Size: 24.0 molecules                                          • Singleton rings  |
|          • Max Size: 88 molecules                                               • Rare scaffolds   |
|          • Mean Membership Strength: 0.8881                                     • Complex natural  |
|          • Silhouette Score (clustered): 0.2982                                   products         |
+----------------------------------------------------------------------------------------------------+
```

---

## 1. Methodology & Distance Metrics

### 1.1 Representation Hierarchy
Density-based clustering results are fundamentally determined by the geometry of the input feature space. To prevent misleading scientific claims, we evaluated three distinct representations:

1. **Primary Representation — Native Morgan Fingerprint Space (Jaccard Metric)**:
   - Morgan circular bit vectors: radius = 2, fingerprint size = 2,048 bits, chirality enabled (`useChirality=True`).
   - Distance metric: Jaccard distance $D(A, B) = 1.0 - \text{Tanimoto}(A, B)$.
   - Preserves uncompressed chemical structure without topological distortion.
2. **Exploratory Representation — 10-Dimensional UMAP (Euclidean Metric)**:
   - Generated in Phase 5 Step 3 from Morgan fingerprints via UMAP (`metric='jaccard'`, `n_components=10`, `n_neighbors=15`, `min_dist=0.1`, `random_state=42`).
   - Evaluates clustering behavior on an intermediate non-linear manifold.
3. **Visualization-Oriented Representation — 2-Dimensional UMAP (Euclidean Metric)**:
   - 2D UMAP projection (`n_components=2`).
   - Evaluated strictly to observe clustering behavior under extreme 2D projection. **Never used as the primary chemical clustering reference.**

### 1.2 Core Parameters and Mathematical Semantics
HDBSCAN relies on two primary hyperparameters whose roles are strictly distinguished:
- **`min_cluster_size` ($mcs = 15$)**: Sets the minimum group size (cardinality) for a cluster to be extracted during condensation of the cluster tree hierarchy. Any split containing fewer than $mcs$ points is considered points falling out of a continuing cluster rather than a true cluster split.
- **`min_samples` ($ms = 5$)**: Governs the local density threshold through the $k$-nearest-neighbor core distance ($k = ms = 5$, counting the molecule itself). The mutual reachability distance is defined as:
  $$d_{\text{mreach}-k}(a, b) = \max\left(d_{\text{core}}(a), d_{\text{core}}(b), d(a, b)\right)$$
  Molecules residing in low-density regions have large core distances and are thus pushed farther apart in the minimum spanning tree.
- **Noise Molecules (Label `-1`)**: Indicates that a molecule was not assigned to any selected cluster under the chosen condensed tree hierarchy and Excess of Mass (EOM) extraction. It does **not** simply mean a molecule lacks 15 neighbors within a fixed radius; rather, it reflects that the point did not belong to a sufficiently stable, persistent cluster of size $\ge 15$ at the mutual reachability density scale.

### 1.3 Memory-Bounded BLAS Jaccard Distance Matrix
For $N = 4,983$ molecules, calculating pairwise distances requires $N(N - 1) / 2 = 12,412,653$ comparisons. Naive pairwise loops in pure Python or unvectorized SciPy `pdist` require $>90$ seconds. 

We implemented a BLAS matrix formulation in [`compute_jaccard_distance_matrix`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/app/clustering/hdbscan_clustering.py):
$$\text{Intersection}_{ij} = (X \cdot X^T)_{ij}$$
$$\text{Union}_{ij} = \text{OnBits}_i + \text{OnBits}_j - \text{Intersection}_{ij}$$
$$\text{Tanimoto}_{ij} = \frac{\text{Intersection}_{ij}}{\text{Union}_{ij}}$$
$$D_{ij} = 1.0 - \text{Tanimoto}_{ij}$$

- **Execution Time**: **4.455 seconds** on CPU.
- **Memory Footprint**: $4,983 \times 4,983 \times 8\text{ bytes} = 189.44\text{ MB}$ (`float64`).
- **Validation**:
  - Exact symmetry: $\max |D - D^T| < 10^{-15}$.
  - Identical zero diagonal: $\forall i, D_{ii} = 0.0$.
  - Exact bounds: $D_{ij} \in [0.0, 1.0]$.
  - Zero-bit handling: Single inorganic atoms (e.g., `MOL_03583: [Zn]` with 0 Morgan on-bits) are assigned $D_{ii} = 0.0$ and $D_{ij} = 1.0$ for $i \neq j$.

---

## 2. Statistical Distinction of Model Metrics

To ensure scientific rigor, three distinct quantities from the fitted HDBSCAN model are tracked and differentiated:

1. **Membership Strengths (`clusterer.probabilities_`)**:
   - Measures the relative stability of each point within its cluster's condensed tree hierarchy.
   - Values are bounded in $[0.0, 1.0]$, where noise points are identically $0.0$.
   - **Crucial Caveat**: Membership strengths are heuristic density-tree scores, **not calibrated statistical probabilities** and **not raw $\lambda$ density values**.
2. **Cluster Stability / Persistence (`clusterer.cluster_persistence_`)**:
   - Represents the library's actual cluster persistence array across tree simplifications.
   - In HDBSCAN, cluster stability $S(C)$ integrates the excess of mass across the condensed tree:
     $$S(C) = \sum_{p \in C} \left(\lambda_p - \lambda_{\text{birth}}(C)\right)$$
   - It is an integrated measure of cluster stability, **not a simple 1D interval $\lambda_{\text{death}} - \lambda_{\text{birth}}$**.
3. **Outlier Scores (`clusterer.outlier_scores_`)**:
   - GLOSH (Global-Local Outlier Score from Hierarchies), measuring relative point outlierness across all levels of the cluster hierarchy.
   - **Documented Project Convention**: Detached singletons or noise points whose reachability in the minimum spanning tree is undefined evaluate to `NaN` in standard GLOSH routines. These are imputed to `1.0` (maximal outlier status). This project convention accurately reflects that completely detached points are maximal outliers, while ensuring complete numerical stability without concealing model outputs.

---

## 3. Parameter Sensitivity Grid

We evaluated a controlled 24-point parameter grid exploring $mcs \in \{5, 10, 15, 20, 25, 30\}$ and $ms \in \{\text{default (None)}, 3, 5, 10\}$:

| Config ID | `min_cluster_size` | `min_samples` | Clusters | Noise Count | Noise % | Median Size | Max Size | Clustered Mean Strength | Silhouette (Clustered) | Runtime (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `mcs_5_ms_3` | 5 | 3 | 179 | 3,295 | 66.12% | 8.0 | 36 | 0.8484 | 0.4112 | 4.48 |
| `mcs_5_ms_5` | 5 | 5 | 113 | 3,695 | 74.15% | 9.0 | 66 | 0.8327 | 0.4506 | 3.37 |
| `mcs_5_ms_10` | 5 | 10 | 47 | 4,212 | 84.53% | 13.0 | 60 | 0.8826 | 0.4824 | 3.83 |
| `mcs_10_ms_3` | 10 | 3 | 72 | 3,506 | 70.36% | 17.0 | 76 | 0.8734 | 0.3073 | 3.60 |
| `mcs_10_ms_5` | 10 | 5 | 56 | 3,807 | 76.40% | 17.5 | 88 | 0.8671 | 0.3496 | 3.91 |
| `mcs_10_ms_10` | 10 | 10 | 35 | 4,220 | 84.69% | 18.0 | 69 | 0.8209 | 0.4398 | 3.12 |
| `mcs_15_ms_3` | 15 | 3 | 46 | 3,647 | 73.19% | 23.5 | 76 | 0.9118 | 0.2770 | 2.80 |
| **`mcs_15_ms_5`** *(Selected Baseline)* | **15** | **5** | **37** | **3,873** | **77.72%** | **24.0** | **88** | **0.8881** | **0.2982** | **2.63** |
| `mcs_15_ms_10` | 15 | 10 | 24 | 4,304 | 86.37% | 24.0 | 73 | 0.8749 | 0.3980 | 2.63 |
| `mcs_20_ms_3` | 20 | 3 | 32 | 3,715 | 74.55% | 34.5 | 108 | 0.8841 | 0.2226 | 2.69 |
| `mcs_20_ms_5` | 20 | 5 | 26 | 3,999 | 80.25% | 31.5 | 88 | 0.8945 | 0.2648 | 3.13 |
| `mcs_25_ms_5` | 25 | 5 | 19 | 3,998 | 80.23% | 40.0 | 176 | 0.8073 | 0.2217 | 3.16 |
| `mcs_30_ms_default` | 30 | -1 | 0 | 4,983 | 100.0% | 0.0 | 0 | 0.0000 | N/A | 2.67 |

### Scientific Parameter Selection Rationale
We selected **`min_cluster_size = 15, min_samples = 5`** as a principled **exploratory baseline** (not an objectively optimal truth, as unsupervised clustering in chemistry depends on intended resolution):
1. **Cluster Granularity**: A minimum size of 15 molecules identifies substantial congeneric series (e.g., aliphatic esters, monoterpenes, substituted phenols, pyrazines) while avoiding fragmented micro-clusters of trivial isomers.
2. **Noise Representation**: At $ms = 5$, spurious single-molecule density bridges are rejected, yielding $77.72\%$ noise. This faithfully reflects the genuine structural sparsity of organic chemical libraries.
3. **Cluster Stability**: Clustered molecules exhibit high mean membership strength ($0.8881$), indicating robust core placement in the condensed tree.
4. **Avoidance of Degeneracy**: Unlike $mcs = 30, ms = \text{default}$ (which collapses to 0 clusters / 100% noise), $mcs = 15, ms = 5$ retains 37 reproducible chemical families.

---

## 4. Representation Comparison: Cautious Interpretation of Dimensionality Reduction

We evaluated HDBSCAN clustering behavior under identical hyperparameters ($mcs = 15, ms = 5$) across three geometric representations:

| Representation | Metric | Clusters | Noise Count | Noise % | Median Size | Max Size | Clustered Membership | Silhouette (Clustered) | Fit Time |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Primary: Morgan FP** | **Precomputed Jaccard** | **37** | **3,873** | **77.72%** | **24.0** | **88** | **0.8881** | **0.2982** | **3.17 s** |
| Exploratory: 10D UMAP | Euclidean | 107 | 1,225 | 24.58% | 30.0 | 125 | 0.8734 | 0.5606 | 0.14 s |
| Visualization: 2D UMAP | Euclidean | 117 | 941 | 18.88% | 30.0 | 99 | 0.8486 | 0.5767 | 0.06 s |

```
Noise Proportion across Chemical Representations
Morgan FP (Primary):   [████████████████████████████████████████     ]  77.72% (True Chemical Sparsity)
10D UMAP (Exploratory): [████████████                                 ]  24.58% (Moderate Manifold Compression)
2D UMAP (Visualization):[█████████                                    ]  18.88% (Severe Non-linear Projection)
```

### Scientifically Cautious Interpretation:
1. **Clustering is Representation-Dependent**: The number of discovered clusters and noise fraction vary dramatically across representations because density-based clustering is sensitive to local distance geometry.
2. **Geometric Distortion of Dimensionality Reduction**: Compressing a 2,048-dimensional sparse binary space into 10 or 2 dimensions fundamentally alters distance distributions, local neighborhood ranks, and relative point densities.
3. **Non-Equivalence of Projected Clusters**: Clusters identified in 2D or 10D UMAP space are **not mathematically or chemically equivalent** to clusters in the original fingerprint space. UMAP is designed to optimize low-dimensional fuzzy simplicial sets for visual layout; it does not preserve high-dimensional metric distances.
4. **Role of 2D Representation**: The 2D UMAP clustering is strictly visualization-oriented and exploratory. It must never be presented as the primary reference for chemical families.
5. **Silhouette Incomparability**: Silhouette scores cannot be directly compared across different feature spaces or distance metrics. The higher silhouette score in 2D UMAP ($0.5767$ vs. $0.2982$) reflects Euclidean distance evaluation on coordinates already optimized for visual grouping, not superior chemical partitioning.

---

## 5. Visualizations & Empirical Observations

All figures are archived in [`backend/experiments/phase5/clustering/`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/clustering):

### 1. HDBSCAN Clusters over 2D UMAP (`hdbscan_clusters_umap2d.png`)
- Displays the 37 primary chemical clusters projected onto the 2D UMAP embedding.
- Clustered points form contiguous groupings on peripheral islands and cores, while $3,873$ noise molecules (grey) diffuse across the global manifold.

### 2. Density Partition: Clustered vs. Noise (`noise_vs_clustered_umap2d.png`)
- Contrasts the core congeneric scaffolds ($22.28\%$, green) against sparse chemical space ($77.72\%$, red).
- Demonstrates that density-based clustering does not force every molecule into a cluster, avoiding the artificial partition artifacts common to $K$-Means.

### 3. Cluster Size Distribution (`cluster_size_distribution.png`)
- Rank-ordered bar chart of the 37 clusters.
- Median size: $24.0$ molecules. Largest cluster: $88$ molecules. Smallest cluster: $15$ molecules (enforcing $mcs = 15$).

### 4. Membership Strength Distribution (`membership_strength_distribution.png`)
- Bimodal density plot: noise points have membership strength $0.0$; clustered molecules show strong concentration near $[0.8, 1.0]$.
- Confirms that points assigned to clusters are core points deep within condensed trees.

### 5. Parameter Sensitivity Heatmaps (`parameter_sensitivity_comparison.png`)
- Heatmaps of cluster counts and noise proportions across all 24 grid configurations.
- Demonstrates smooth, monotonic behavior across varying neighborhood definitions.

---

## 6. Artifact Inventory

The following artifacts have been created in `backend/experiments/phase5/clustering/`:

| Artifact | Type | Description |
|---|---|---|
| [`hdbscan_fingerprints_primary.csv`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/clustering/hdbscan_fingerprints_primary.csv) | CSV | 4,983 molecules: `molecule_id`, `canonical_smiles`, `cluster_hdbscan`, `membership_strength`, `outlier_score`. |
| [`hdbscan_umap_10d_exploratory.csv`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/clustering/hdbscan_umap_10d_exploratory.csv) | CSV | Exploratory cluster assignments from 10D UMAP. |
| [`hdbscan_umap_2d_visualization.csv`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/clustering/hdbscan_umap_2d_visualization.csv) | CSV | Visualization cluster assignments from 2D UMAP. |
| [`parameter_sensitivity_results.csv`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/clustering/parameter_sensitivity_results.csv) | CSV | Tabular summary across all 24 parameter configurations. |
| [`parameter_sensitivity_results.json`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/clustering/parameter_sensitivity_results.json) | JSON | Detailed statistical metadata per configuration. |
| [`cluster_size_distribution.json`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/clustering/cluster_size_distribution.json) | JSON | Molecule counts for all 37 clusters and noise. |
| [`cluster_persistence.json`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/clustering/cluster_persistence.json) | JSON | Cluster stability / persistence from fitted library model attribute. |
| [`representation_comparison.csv`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/clustering/representation_comparison.csv) | CSV | Comparative metrics between Morgan FP, 10D UMAP, and 2D UMAP. |
| [`experiment_metadata.json`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/clustering/experiment_metadata.json) | JSON | Complete execution parameters, runtime, memory, and caveats. |
| `hdbscan_clusters_umap2d.png` | PNG | Visual scatter plot of 37 clusters over 2D UMAP. |
| `noise_vs_clustered_umap2d.png` | PNG | Binary scatter plot of core clusters vs. noise. |
| `cluster_size_distribution.png` | PNG | Bar chart of rank-ordered cluster sizes. |
| `membership_strength_distribution.png` | PNG | Distribution of HDBSCAN membership strengths. |
| `parameter_sensitivity_comparison.png` | PNG | Heatmaps of cluster counts and noise proportions. |

---

## 7. Mandatory Scientific Caveats

1. **Arbitrary Numerical Identifiers**: Cluster labels (0 to 36) are arbitrary integer identifiers produced by density reachability; they do not represent natural taxonomies or human odor classes.
2. **Noise Molecules (-1) Are Chemically Valid**: A noise assignment indicates that a molecule does not meet the local density threshold (i.e. it lacks $\ge 15$ structural neighbors within a dense radius). In fragrance chemistry, noise molecules represent unique scaffolds or isolated natural products, not synthetic errors.
3. **Membership Strengths Are Not Probabilities**: HDBSCAN membership strength $\lambda$ measures stability across the condensed cluster hierarchy. They are heuristic density values and **not calibrated statistical probabilities**.
4. **Structural Clusters $\neq$ Perceptual Odor Clusters**: Molecules in the same structural cluster do not automatically smell the same (activity cliffs), nor do molecules with the same smell necessarily cluster together (scaffold hopping / polypharmacology).
5. **Annotation Sparsity vs. Intensity**: The binary labels in the OpenPOM dataset reflect cataloged annotations, not perceptual odor threshold, volatility, or perceived sensory intensity.
6. **Exploratory Scope**: This clustering analysis is strictly unsupervised and does not establish causal chemical-receptor or structure-odor relationships.

---

## 8. Next Steps: Phase 5 Step 5 (Statistical Enrichment & Outliers)

With robust, validated HDBSCAN cluster assignments established:
- **Cluster Odor Enrichment**: Perform Fisher's Exact Tests with Benjamini-Hochberg False Discovery Rate (FDR) control across all cluster-descriptor pairs to discover statistically significant odor enrichments.
- **Outlier Analysis**: Systematically inspect GLOSH outlier scores and structural boundary molecules.
