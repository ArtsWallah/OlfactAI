# OlfactAI — Phase 2 Research Report: Dataset Acquisition, Molecular Preprocessing & Exploratory Analysis

## 1. Executive Summary

This report documents the empirical investigation, acquisition, chemical preprocessing, quality validation, exploratory data analysis, and leakage-safe partitioning of the benchmark human olfactory dataset for **OlfactAI**.

The primary dataset acquired is the curated GoodScents & Leffingwell merged benchmark from the official OpenPOM initiative (Lee et al., *Science* 2023). Every metric, dimension, distribution, and split size in this document derives from deterministic, reproducible code execution on CPU.

---

## 2. Dataset Source & Provenance

- **Primary Source Repository:** `https://github.com/BioMachineLearning/openpom`
- **Source File Path:** `openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv`
- **Direct Raw Download URL:** `https://raw.githubusercontent.com/BioMachineLearning/openpom/main/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv`
- **Repository Commit Identifier:** `5434c4b7badebca8e7d180aa34d2f787402121ee`
- **Local Raw Storage Location:** `backend/datasets/raw/curated_GS_LF_merged_4983.csv`
- **File Size:** `1,647,971` bytes (1.65 MB)
- **SHA-256 Checksum:** `0c18b6e9f0f99b772203ba4da30a96b3d60db700349376cbafe23512a9baa704`
- **Retrieval Date:** November 7, 2026 (UTC)
- **Underlying Scientific Publication:**
  Brian K. Lee, Emily J. Mayhew, Benjamin Sanchez-Lengeling, et al., *"A principal odor map unifies diverse tasks in olfactory perception"*, **Science** 381, 999–1006 (2023). DOI: `10.1126/science.ade4401`.

---

## 3. License & Attribution

- **Dataset License:** MIT License (Copyright © 2023 Aryan Amit Barsainyan / BioMachineLearning)
- **Constituent Datasets:**
  - **The Good Scents Company (TGSC):** Curated multi-label perceptual database of fragrance and flavor materials.
  - **Leffingwell & Associates:** Flavor-Base database of odor character notes.
- **Attribution Requirement:**
  Any modeling, evaluation, or downstream publications must cite both Lee et al. (Science 2023) and the OpenPOM repository.

---

## 4. Verification of Referenced Benchmark Figures

Before acquisition, we investigated the widely cited figures of **4,983 molecules** and **138 odor descriptors**:
- **Observed Row Count:** Exactly **4,983** records.
- **Observed Column Count:** **140** total columns.
  - Column 0: `nonStereoSMILES` (chemical structure string)
  - Column 1: `descriptors` (comma-separated string of sensory notes)
  - Columns 2 to 139: **138** distinct binary odor descriptor attributes (`alcoholic`, `aldehydic`, ..., `woody`).

**Finding:** The referenced figures of 4,983 molecules and 138 odor descriptors are **empirically confirmed** for `curated_GS_LF_merged_4983.csv`.

### Alternative Datasets in the Repository
The OpenPOM repository also provides two constituent unmerged datasets:
1. `curated_goodcents.csv`: 4,392 molecules $\times$ 140 columns (GoodScents alone).
2. `curated_leffingwell.csv`: 3,510 molecules $\times$ 139 columns (Leffingwell alone).

**Decision:** We selected `curated_GS_LF_merged_4983.csv` as our sole primary dataset because it represents the unified, consensus benchmark established by Lee et al. (2023). No supplementary datasets were merged into this file.

---

## 5. Dataset Inspection & Quality Audit

The raw dataset was analyzed using `backend/scripts/inspect_dataset.py`.

| Dimension / Metric | Observed Value | Interpretation |
|---|---|---|
| **Total Rows** | 4,983 | Full cohort |
| **Total Columns** | 140 | 1 SMILES + 1 text descriptors + 138 binary targets |
| **Missing Values (Cells)** | 0 | 100% complete across all 697,620 matrix cells |
| **Exact Duplicate Rows** | 0 | No redundant records |
| **Raw SMILES Duplicates** | 0 | All input SMILES strings are distinct |
| **Valid SMILES (RDKit)** | 4,983 (100.0%) | Zero unparseable structures; valence rules satisfied |
| **Invalid SMILES** | 0 (0.0%) | No broken rings, valence violations, or empty rows |
| **Unique Canonical SMILES** | 4,983 | 1:1 mapping between canonical structures and records |
| **Duplicate Canonical Structures** | 0 | No tautomeric or stereoisomeric collapse to identical canonical SMILES |
| **Conflicting Odor Annotations** | 0 | No contradictory perceptual annotations exist |
| **Molecules with 0 Odors** | 0 | Every compound possesses $\ge 1$ verified olfactory label |
| **Minimum Labels per Molecule** | 1 | Minimum annotation depth |
| **Maximum Labels per Molecule** | 19 | Highly rich perceptual profiles |
| **Mean Labels per Molecule** | 4.88 | Multi-label density |
| **Median Labels per Molecule** | 4.00 | Robust central tendency |

---

## 6. Molecular Physicochemical Descriptor Statistics

Using the RDKit chemistry engine (`DescriptorExtractor`), 10 physicochemical descriptors were computed for all 4,983 molecules:

| Descriptor | Mean | Std Dev | Min | Median | Max | Olfactory Relevance |
|---|---|---|---|---|---|---|
| **Molecular Weight (g/mol)** | 180.46 | 70.78 | 4.00 | 170.30 | 1347.38 | Volatility cut-off (most odorants $<300$ g/mol) |
| **Wildman-Crippen LogP** | 2.44 | 2.02 | -17.41 | 2.52 | 17.71 | Mucus membrane partition / hydrophobicity |
| **TPSA (Å²)** | 29.34 | 30.49 | 0.00 | 26.30 | 633.20 | Polarity & olfactory receptor binding |
| **H-Bond Donors (HBD)** | 0.40 | 0.98 | 0 | 0 | 24 | Moderate donor capacity in odorants |
| **H-Bond Acceptors (HBA)** | 2.07 | 1.61 | 0 | 2 | 40 | Common carbonyl / ester / ether oxygens |
| **Rotatable Bonds** | 4.07 | 3.66 | 0 | 3 | 54 | Conformational flexibility |
| **Total Ring Count** | 0.85 | 4.39 | 0 | 1 | 264 | Aromatic & alicyclic ring prevalence |
| **Heavy Atom Count** | 12.56 | 5.07 | 1 | 12 | 92 | Molecular size indicator |
| **Fraction SP3 Carbons** | 0.65 | 0.28 | 0.00 | 0.70 | 1.00 | Three-dimensional structural complexity |

---

## 7. Odor Descriptor Frequency & Imbalance

The 138 odor descriptors exhibit a classic long-tail power law distribution typical of human sensory psychophysics.

### Top 10 Most Frequent Descriptors
1. **fruity:** 1,902 molecules (38.17%)
2. **green:** 1,446 molecules (29.02%)
3. **sweet:** 1,429 molecules (28.68%)
4. **floral:** 1,100 molecules (22.07%)
5. **herbal:** 756 molecules (15.17%)
6. **woody:** 643 molecules (12.90%)
7. **spicy:** 536 molecules (10.76%)
8. **fresh:** 497 molecules (9.97%)
9. **citrus:** 494 molecules (9.91%)
10. **fatty:** 488 molecules (9.79%)

### Top 10 Rarest Descriptors
1. **chamomile:** 31 molecules (0.62%)
2. **bergamot:** 32 molecules (0.64%)
3. **orangeflower:** 33 molecules (0.66%)
4. **malty:** 33 molecules (0.66%)
5. **hawthorn:** 34 molecules (0.68%)
6. **plum:** 34 molecules (0.68%)
7. **hyacinth:** 35 molecules (0.70%)
8. **narcissus:** 35 molecules (0.70%)
9. **radish:** 36 molecules (0.72%)
10. **cortex:** 36 molecules (0.72%)

**Imbalance Ratio:** The most frequent descriptor (`fruity`, 1,902) occurs **61.35 times more frequently** than the rarest descriptor (`chamomile`, 31). Multi-label loss functions in Phase 3 must incorporate class-frequency weighting or focal loss.

### Descriptor Co-occurrence Findings
Pearson correlation analysis of binary co-occurrence reveals distinct sensory clusters:
- **Fruit & Sweet Cluster:** Strong positive co-occurrence between `fruity`, `sweet`, `apple`, and `berry`.
- **Green & Herbal Cluster:** Strong co-occurrence between `green`, `herbal`, `leafy`, and `vegetable`.
- **Balsamic & Woody Cluster:** High correlation between `woody`, `cedar`, `amber`, and `balsamic`.
- **Negative Correlations:** Very rare co-occurrence between `citrus` and `sulfurous` or `fishy` and `floral`.

---

## 8. Dataset Splitting & Scaffold Leakage Prevention

Two distinct 70 / 15 / 15 train/validation/test partitions were created (`backend/scripts/create_splits.py`):

### A. Random Split (Fixed Seed: 42)
- **Train Set:** 3,488 molecules (70.00%)
- **Validation Set:** 747 molecules (14.99%)
- **Test Set:** 748 molecules (15.01%)
- **Label Coverage:** All 138 odor descriptors have $\ge 1$ positive instance in Train, Val, and Test. Zero missing labels.

### B. Bemis-Murcko Molecular Scaffold Split
To evaluate model generalization to truly novel chemical space, molecules were partitioned based on their Bemis-Murcko molecular scaffolds:
- **Total Unique Scaffolds:** 556
- **Acyclic Compounds:** 2,236 molecules (acyclic aliphatic structures partitioned by carbon graph generic representations)
- **Train Set:** 3,488 molecules (70.00%)
- **Validation Set:** 747 molecules (14.99%)
- **Test Set:** 748 molecules (15.01%)

### Scaffold Leakage Audit
An automated set-intersection check was executed across all scaffold partitions:

$$\text{Scaffolds}(\text{Train}) \cap \text{Scaffolds}(\text{Test}) = \emptyset \quad (\text{Count} = 0)$$
$$\text{Scaffolds}(\text{Train}) \cap \text{Scaffolds}(\text{Val}) = \emptyset \quad (\text{Count} = 0)$$
$$\text{Scaffolds}(\text{Val}) \cap \text{Scaffolds}(\text{Test}) = \emptyset \quad (\text{Count} = 0)$$

- **Scaffold Leakage Detected:** **False** (proven strictly disjoint).
- **Rare Label Audit under Scaffold Split:**
  - Missing in Train: **0**
  - Missing in Val: **0**
  - Missing in Test: **3** (`hawthorn`, `juicy`, `tomato`)
  - *Scientific Note:* Because scaffold splitting groups entire structural families into single partitions, three extremely rare descriptors whose carriers share unique scaffolds were naturally held in Train or Val. This empirical observation confirms the authenticity of scaffold partitioning and must be factored into Phase 3 model evaluation.

---

## 9. Generated Artifacts & Verification

All artifacts are persisted in `backend/datasets/`:

```
backend/datasets/
├── raw/
│   ├── curated_GS_LF_merged_4983.csv       (1.65 MB, SHA-256 verified)
│   └── raw_dataset_metadata.json           (Provenance, URL, Commit SHA)
│
└── processed/
    ├── molecules.csv                       (4,983 rows, IDs, SMILES, Formula, Splits)
    ├── molecular_features.csv              (4,983 rows x 2,059 columns: 9 desc + 2048 fp)
    ├── odor_labels.csv                     (4,983 rows x 140 columns: IDs + 138 odor targets)
    ├── dataset_metadata.json               (Full feature & label metadata, checksums)
    ├── data_quality_report.json            (Quality inspection metrics)
    ├── splits.json                         (Random & Scaffold split maps)
    ├── splits_summary.json                 (Split distributions & leakage audit results)
    └── eda/
        ├── label_frequency.png             (Top 30 descriptors horizontal bar chart)
        ├── label_cooccurrence.png          (Top 25 descriptor correlation heatmap)
        ├── molecular_weight_distribution.png (MW histogram and KDE)
        ├── logp_distribution.png           (LogP histogram and KDE)
        ├── tpsa_distribution.png           (TPSA histogram and KDE)
        ├── missing_values.png              (100% completeness verification plot)
        └── split_label_distribution.png    (Random vs Scaffold label distribution comparison)
```

---

## 10. Reproduction Pipeline Instructions

To reproduce the entire Phase 2 data pipeline on a clean Windows machine:

```powershell
# 1. Activate Python 3.11 virtual environment
backend\.venv\Scripts\Activate.ps1

# 2. Acquire raw dataset from official OpenPOM source
python backend/scripts/acquire_dataset.py

# 3. Inspect dataset and produce data quality report
python backend/scripts/inspect_dataset.py

# 4. Preprocess molecules, extract descriptors and 2048-bit Morgan fingerprints
python backend/scripts/preprocess_dataset.py

# 5. Generate leakage-safe random and Bemis-Murcko scaffold splits
python backend/scripts/create_splits.py

# 6. Generate the 7 Exploratory Data Analysis visualizations
python backend/scripts/generate_eda.py

# 7. Run full automated test suite (52 tests)
pytest backend/tests -v
```

---

## 11. Known Limitations & Recommendations for Phase 3

1. **Extreme Label Imbalance:** Descriptors such as `chamomile` (31) and `orangeflower` (33) have very few positive instances. Evaluator metrics should compute Macro-F1 alongside Micro-F1 and omit unrepresented classes on zero-instance folds.
2. **Subjectivity of Human Sensory Data:** Consensus odor notes aggregate human panel evaluations; perceptual variance is embedded within the ground truth labels.
3. **Strict Feature Separation:** The pipeline enforces zero leakage between odor labels and molecular features. All models must be evaluated against the scaffold test split to ensure genuine out-of-distribution chemical generalization.

