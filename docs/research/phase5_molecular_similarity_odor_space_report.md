# OlfactAI — Phase 5 Research Report: Molecular Similarity, Neighborhoods, Chemical Space, HDBSCAN, and Statistical Odor Enrichment

**Date**: November 10, 2026  
**Status**: Step 5 (Statistical Odor-Descriptor Enrichment Analysis) Completed  
**Branch**: `main`  
**Environment**: Ubuntu 24.04 (WSL2), Python 3.11.18, CPU-only PyTorch, RDKit, HDBSCAN 0.8.44, UMAP, SciPy  
**Dataset**: OpenPOM Curated Dataset (4,983 molecules, 138 binary odor descriptors)

---

## Executive Summary

Phase 5 investigates the topological and density-based structure of chemical odor space using molecular fingerprints, nearest-neighbor similarity search, global pairwise distributions, non-linear manifold learning (UMAP), unsupervised hierarchical density-based clustering (HDBSCAN), and rigorous observational statistical association testing.

In this milestone (**Step 5: Statistical Odor-Descriptor Enrichment Analysis**), we evaluated whether specific odor descriptors from the 138 binary OpenPOM annotations are statistically significantly overrepresented or underrepresented in the 37 primary structural clusters discovered by HDBSCAN.

```
+----------------------------------------------------------------------------------------------------+
|                               Statistical Odor Enrichment Architecture                             |
|                                                                                                    |
|  [37 HDBSCAN Clusters] ──────┐                                                                     |
|  (1,110 clustered molecules) │                                                                     |
|                              ├──> [2x2 Contingency Tables] ──> [Two-Sided Fisher's Exact Test]     |
|  [138 Binary Odor Labels] ───┘    (37 clusters x 138 descs)    (5,106 hypothesis tests)            |
|                                                                           │                        |
|                                                                           ▼                        |
|                                                               [Benjamini-Hochberg FDR]             |
|                                                               (family-wide q < 0.05)               |
|                                                                           │                        |
|                     ┌─────────────────────────────────────────────────────┴──────────────────┐     |
|                     ▼                                                                        ▼     |
|         [220 Significant Enrichments]                                            [28 Significant   |
|         • q < 0.05, Odds Ratio > 1.0                                               Depletions]     |
|         • 37/37 clusters exhibit >= 1 enrichment                                 • q < 0.05, OR < 1|
|         • Max: Cluster 4 (Musk, OR=400.4, q=5.7e-38)                             • Max: Cluster 6  |
|         • Max: Cluster 13 (Onion/Garlic, OR~740, q~10^-32)                         (Fruity, OR=0.02|
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

## 6. Statistical Odor-Descriptor Enrichment Analysis (Step 5)

### 6.1 Statistical Methodology & Contingency Tables

To examine whether specific odor descriptors are over- or underrepresented in the 37 HDBSCAN clusters, we constructed a 2x2 contingency table for every cluster $k \in \{0, \dots, 36\}$ and descriptor $j \in \{1, \dots, 138\}$:

| | In Cluster $k$ | Outside Cluster $k$ (Other Non-Noise Clusters) |
|---|---|---|
| **Descriptor $j = 1$ (Annotated)** | $a$ | $c$ |
| **Descriptor $j = 0$ (Not Annotated)** | $b$ | $d$ |

- **Sample Universe ($N_{\text{clustered}} = 1,110$)**: Unclustered noise molecules (`cluster = -1`, $N_{\text{noise}} = 3,873$) were strictly excluded from the background of individual cluster tests, preventing diffuse background distortion and artificial deflation of cluster prevalence differences.
- **Inferential Testing**: Two-sided Fisher's Exact Test (`scipy.stats.fisher_exact`) to evaluate exact hypergeometric probabilities without large-sample asymptotic cell-count approximations.
- **Odds Ratio Metrics & Confidence Intervals**:
  - **Raw Sample Odds Ratio**: $\text{OR}_{\text{raw}} = (a \cdot d) / (b \cdot c)$. Where $b \cdot c = 0$ and $a \cdot d > 0$, $\text{OR} = \infty$; where $a \cdot d = 0$, $\text{OR} = 0.0$.
  - **Exact Conditional 95% Confidence Intervals**: Computed using `scipy.stats.contingency.odds_ratio(..., kind="conditional")` based on the non-central hypergeometric distribution. This provides valid confidence bounds even for zero-cell and small-count contingency tables (e.g. $[0, \text{high}]$ when $a=0$, or $[\text{low}, \infty]$ when $b=0$).
  - **Approximate Woolf 95% Log-Linear CI**: $\exp(\ln(\text{OR}) \pm 1.96 \cdot \text{SE})$ where $\text{SE} = \sqrt{1/a + 1/b + 1/c + 1/d}$. Recorded as an exploratory diagnostic; undefined and explicitly flagged as NaN whenever any cell count is zero.
  - **Haldane-Anscombe Corrected Effect Size**: To stabilize variance across zero-count cells for visualization and ranking, we computed:
    $$\log_2(\text{OR}_{\text{Haldane}}) = \log_2\left(\frac{(a + 0.5)(d + 0.5)}{(b + 0.5)(c + 0.5)}\right)$$
- **Distinct Multiple Testing FDR Families**:
  - **Family A (Cluster Enrichment)**: $37 \times 138 = 5,106$ simultaneous hypothesis tests adjusted via Benjamini-Hochberg (BH) FDR ($q < 0.05$). No cluster-wise subsetting is presented as global control.
  - **Family B (Clustered vs. Noise Populations)**: 138 descriptor comparisons adjusted via BH FDR independently across its own 138 tests ($q < 0.05$).

---

### 6.2 Key Statistical Associations and Observed Structural Motifs

Across Family A (5,106 hypothesis tests), **220 associations** demonstrated statistically significant enrichment ($q < 0.05, \text{OR} > 1.0$), and all 37 structural clusters exhibited at least one significantly enriched odor descriptor:

1. **Cluster 4 (Size 40)**:
   - **Enriched descriptor**: `musk` ($a=29/40$, cluster prevalence = 72.5% vs. background prevalence = 0.65%).
   - **Odds Ratio & CIs**: Raw $\text{OR} = 400.4$, Exact Conditional 95% CI $[93.7, 2406.8]$, Woolf 95% CI $[93.7, 1710.8]$, $\log_2(\text{OR}_{\text{Haldane}}) = 8.51$, $q = 5.70 \times 10^{-38}$.
   - **Structural inspection**: All 40 molecules possess cyclic oxygenated scaffolds (heavy atom range 6–19, 1–2 rings), including macrocyclic lactones (e.g. oxacyclohexadecan-2-one) and substituted cyclohexyl/cyclopentyl ketones and ethers. *Hypothesis for future structural investigation: Macrocyclic and multi-ring oxygenated packing density correlates strongly with musk descriptor cataloging in OpenPOM.*
2. **Cluster 13 (Size 26)**:
   - **Enriched descriptors**: `onion` ($a=23/26$, prev = 88.5% vs. 1.0%, raw $\text{OR} = 747.8$, exact CI $[149.2, 5459.7]$, $q = 5.44 \times 10^{-33}$), `garlic` ($a=22/26$, prev = 84.6% vs. 0.74%, raw $\text{OR} = 739.8$, exact CI $[144.3, 5458.1]$, $q = 2.02 \times 10^{-32}$), and `alliaceous` ($a=20/26$, prev = 76.9% vs. 0.83%, raw $\text{OR} = 398.1$, exact CI $[88.8, 2451.9]$, $q = 6.71 \times 10^{-28}$).
   - **Structural inspection**: 26/26 molecules (100%) are acyclic organosulfur compounds (heavy atoms 6–12, 0 rings, 0 nitrogens) containing dialkyl sulfides, disulfides, and trisulfides (e.g. dibutyl sulfide, dipropyl disulfide). *Hypothesis: Open-chain polysulfide linkages represent an isolated structural cluster with nearly uniform allium/alliaceous catalog tags.*
3. **Cluster 6 (Size 66)**:
   - **Enriched descriptors**: `nutty` ($a=44/66$, prev = 66.7% vs. 5.8%, raw $\text{OR} = 32.2$, exact CI $[17.7, 58.7]$, $q = 3.31 \times 10^{-30}$), `roasted` ($a=35/66$, prev = 53.0% vs. 3.2%, raw $\text{OR} = 34.6$, exact CI $[17.7, 68.3]$, $q = 2.30 \times 10^{-26}$).
   - **Structural inspection**: 66/66 molecules (100%) are nitrogenous aromatic heterocycles (heavy atoms 7–20, 1–2 rings) primarily comprising alkylpyrazines (e.g. 2-ethylpyridine, dimethylpyrazines, ethylmethylpyrazines) and pyridines.
4. **Cluster 17 (Size 88)**:
   - **Enriched descriptors**: `alcoholic` ($a=34/88$, prev = 38.6% vs. 0.88%, raw $\text{OR} = 70.9$, exact CI $[30.8, 172.9]$, $q = 1.41 \times 10^{-29}$), `green` ($a=79/88$, prev = 89.8% vs. 30.6%, raw $\text{OR} = 19.9$, exact CI $[10.3, 40.7]$, $q = 1.05 \times 10^{-25}$), `cognac` ($a=39/88$, prev = 44.3% vs. 3.6%, raw $\text{OR} = 21.2$, exact CI $[11.5, 39.4]$, $q = 6.31 \times 10^{-24}$).
   - **Structural inspection**: All 88 molecules contain multiple ether/acetal linkages and dialkoxy moieties (e.g. diethyl acetals, alkyl ethoxy ethers). Note: While frequently tagged as `alcoholic` and `cognac` in flavor catalogs due to their presence in distilled spirit bouquets, structural inspection reveals they are largely acetals and ethers rather than free monoterpene alcohols.
5. **Cluster 3 (Size 32)**:
   - **Enriched descriptor**: `woody` ($a=30/32$, prev = 93.8% vs. 9.56%, raw $\text{OR} = 142.0$, exact CI $[34.9, 1073.4]$, $q = 2.26 \times 10^{-24}$).

---

### 6.3 Statistically Significant Depletions

In contrast to enrichments, **28 associations** showed statistically significant depletion ($q < 0.05, \text{OR} < 1.0$):
- **Cluster 6 (Pyrazines/Heterocycles)**: Strongly depleted in common pleasant volatile descriptors: `fruity` (prev = 1.5% vs. background 45.9%, raw $\text{OR} = 0.018$, exact CI $[0.001, 0.106]$, $q = 1.87 \times 10^{-13}$), `waxy` (prev = 0.0% vs. 24.7%, raw $\text{OR} = 0.0$, exact CI $[0.0, 0.063]$, $q = 1.74 \times 10^{-6}$), `sweet` (prev = 4.5% vs. 32.4%, raw $\text{OR} = 0.099$, exact CI $[0.024, 0.301]$, $q = 7.34 \times 10^{-6}$), `floral` (prev = 1.5% vs. 24.7%, raw $\text{OR} = 0.047$, exact CI $[0.003, 0.281]$, $q = 2.87 \times 10^{-5}$).
- **Cluster 4 (Oxygenated Macrocycles/Rings)**: Zero `green` notes ($0/40$ vs. 36.6% background, raw $\text{OR} = 0.0$, exact CI $[0.0, 0.098]$, $q = 1.55 \times 10^{-6}$).
- **Cluster 13 (Acyclic Sulfides)**: Zero `fruity` annotations ($0/26$ vs. 44.3% background, raw $\text{OR} = 0.0$, exact CI $[0.0, 0.096]$, $q = 2.54 \times 10^{-5}$).

---

### 6.4 Global Comparison: Clustered vs. Noise Populations

In Family B (138 tests comparing $N_{\text{clustered}} = 1,110$ vs. $N_{\text{noise}} = 3,873$):
- **Odor-Annotation Prevalence Differences**: Molecules located in dense clusters exhibit significantly higher catalogued prevalence for common volatile and flavor descriptor tags:
  - `fatty` (27.7% in clustered vs. 7.5% in noise, raw $\text{OR} = 4.71$, exact CI $[3.89, 5.70]$, $q = 4.36 \times 10^{-61}$)
  - `waxy` (23.2% in clustered vs. 5.9% in noise, raw $\text{OR} = 4.80$, exact CI $[3.90, 5.90]$, $q = 1.63 \times 10^{-53}$)
  - `oily` (17.4% in clustered vs. 5.0% in noise, raw $\text{OR} = 3.97$, exact CI $[3.16, 4.97]$, $q = 7.24 \times 10^{-34}$)
  - `cognac` (6.8% in clustered vs. 0.93% in noise, raw $\text{OR} = 7.83$, exact CI $[5.01, 12.35]$, $q = 7.49 \times 10^{-24}$)
- **Confounder and Selection Caveat**: These differences reflect **dataset selection and annotation coverage biases** rather than an intrinsic division between "synthetic" and "natural" molecules. Molecules in dense structural clusters belong to heavily sampled chemical series (e.g. series of homologous aliphatic esters, acetals, pyrazines, and aldehydes) that have been systematically assayed and documented in flavor catalogs. Conversely, HDBSCAN noise molecules represent isolated scaffolds, complex polycycles, or singleton ring systems that have fewer analogues in the OpenPOM catalog and more sparse descriptive labeling. Noise status must not be interpreted as indicating natural origin.

---

## 7. Artifact Inventory

### HDBSCAN Clustering (`backend/experiments/phase5/clustering/`):
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

### Odor Enrichment (`backend/experiments/phase5/enrichment/` & `plots/`):
| Artifact | Type | Description |
|---|---|---|
| [`cluster_odor_enrichment.csv`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/enrichment/cluster_odor_enrichment.csv) | CSV | All 5,106 tests with p-values, q-values, ORs, CIs, and directions. |
| [`cluster_odor_enrichment.json`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/enrichment/cluster_odor_enrichment.json) | JSON | Full enrichment records in JSON. |
| [`significant_enrichments.csv`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/enrichment/significant_enrichments.csv) | CSV | 220 significant enrichments ($q < 0.05, \text{OR} > 1.0$). |
| [`significant_depletions.csv`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/enrichment/significant_depletions.csv) | CSV | 28 significant depletions ($q < 0.05, \text{OR} < 1.0$). |
| [`cluster_odor_prevalence.csv`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/enrichment/cluster_odor_prevalence.csv) | CSV | Full prevalence matrix for 37 clusters, background, and noise. |
| [`global_clustered_vs_noise.csv`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/enrichment/global_clustered_vs_noise.csv) | CSV | Comparative analysis between 1,110 clustered and 3,873 noise molecules. |
| [`enrichment_metadata.json`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/experiments/phase5/enrichment/enrichment_metadata.json) | JSON | Pipeline metadata, execution time (12.26s), and summary counts. |
| `cluster_odor_log2_odds_ratio_heatmap.png` | PNG | Heatmap of Haldane log2 OR across clusters and top descriptors. |
| `significant_enrichment_overview.png` | PNG | Horizontal bar chart of top 25 significant odor enrichments. |
| `significant_depletion_overview.png` | PNG | Horizontal bar chart of top significant odor depletions. |
| `global_clustered_vs_noise_prevalence.png` | PNG | Comparative bar chart of descriptor frequencies in clustered vs noise. |
| `cluster_level_prevalence_summary.png` | PNG | Most enriched descriptor per cluster vs background prevalence. |

---

## 8. Mandatory Scientific Caveats

1. **Association, Not Causation**: Statistically significant enrichment demonstrates that a binary odor label is observed more frequently among molecules of a given chemical cluster than expected by chance. It **does not prove that the cluster scaffold causes the odor perception** or that all molecules possessing that scaffold will smell identical.
2. **Binary Annotation vs. Sensory Intensity**: OpenPOM odor labels are binary tags curated from catalog descriptions. An enrichment of `garlic` at 84.6% indicates label prevalence, **not odor detection threshold, volatility, or perceived intensity**.
3. **Arbitrary Numerical Identifiers**: Clusters are labeled numerically (0 to 36). They are **not named after sensory notes** (e.g., Cluster 4 is not called the "Musk Cluster"), avoiding reification of unsupervised density groupings into biological categories.
4. **Noise Molecules (-1) Are Chemically Valid**: Noise molecules represent singleton scaffolds or dispersed chemotypes that lack $\ge 15$ close topological neighbors; they are chemically valid and must not be discarded or treated as errors.
5. **Multiple Hypothesis Correction**: Because 5,106 individual tests were conducted, raw unadjusted $p$-values are statistically uninterpretable. All claims are strictly based on Benjamini-Hochberg FDR-adjusted $q$-values.

---

## 9. Next Steps: Phase 5 Step 6 (Dedicated Molecular Outlier Analysis)

With both HDBSCAN density clustering and statistical odor enrichment fully mapped:
- **Dedicated Outlier Analysis**: Inspect the 3,873 noise molecules and quantify boundary vs. isolated structures using GLOSH outlier scores.
- **Activity Cliffs & Structural Anomaly Mining**: Identify molecules with high structural similarity to enriched clusters that lack the corresponding odor annotation, or conversely, molecules that exhibit distinctive odor notes despite being structural outliers.
