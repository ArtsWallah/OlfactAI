# OlfactAI Phase 2 — Independent Verification & Scientific Audit

**Audit Date:** November 7, 2026  
**Auditor:** Independent Software Architect & Computational Chemist  
**Target Codebase:** OlfactAI (Commit `25ca971`)  
**Scope:** Phase 2 Dataset Acquisition, Molecular Processing, Exploratory Data Analysis, and Scaffold Splitting  

---

## 1. Executive Summary & Verdict

This independent audit rigorously evaluates the data integrity, cheminformatics preprocessing, statistical claims, partition leakage, environment reproducibility, and legal/licensing boundaries of OlfactAI Phase 2.

**Final Audit Recommendation:** **`APPROVED FOR PHASE 3 (WITH DOCUMENTED CAVEATS)`**  
All 52 automated tests pass with 0 regressions, dataset dimensions match the Lee et al. (*Science* 2023) benchmark exactly, features and labels are strictly segregated with zero target leakage, and Bemis-Murcko scaffold splitting demonstrates mathematically verified **zero partition leakage** ($\text{Train} \cap \text{Test} = \emptyset$). 

Several scientific overstatements and environment ambiguities were identified and documented below with required corrections.

---

## 2. Audit 1 — Dataset Provenance & Licensing Boundaries

### 2.1 Provenance Verification
- **Verified Source:** Official OpenPOM repository (`BioMachineLearning/openpom`).
- **Commit SHA:** `5434c4b7badebca8e7d180aa34d2f787402121ee` (October 17, 2025).
- **Target File:** `openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv`.
- **Local File:** `backend/datasets/raw/curated_GS_LF_merged_4983.csv`.
- **Observed File Size:** `1,647,971` bytes.
- **Observed SHA-256 Checksum:** `0c18b6e9f0f99b772203ba4da30a96b3d60db700349376cbafe23512a9baa704` (Matches recorded hash bit-for-bit).
- **Observed Dimensions:** Exactly `4,983` rows $\times$ `140` columns.
- **Column Schema:**
  - Index 0: `nonStereoSMILES`
  - Index 1: `descriptors`
  - Indices 2–139: 138 binary odor attributes (`alcoholic` through `woody`).

### 2.2 Intellectual Property & Licensing Distinction
An important legal/licensing distinction must be made:
1. **OpenPOM Software License:** The code, repository structure, and curation notebooks authored by Aryan Amit Barsainyan / BioMachineLearning are licensed under the **MIT License**.
2. **Underlying Sensory Content:** The underlying odor notes and sensory characterizations originate from two proprietary / compiled industrial sources:
   - **The Good Scents Company (TGSC):** A proprietary commercial flavor and fragrance database.
   - **Leffingwell & Associates:** Flavor-Base database of commercial fragrance notes.
3. **Audit Finding on Licensing Claims:**  
   The statement in Phase 2 documentation that the dataset is covered by the "MIT License" is true only for the repository code and curated tabular arrangement as distributed by OpenPOM. It does **not** grant unlimited commercial rights to the underlying proprietary sensory compendia from GoodScents and Leffingwell.
4. **Licensing Caveat for Production:**  
   For open-source academic research and local development, OpenPOM usage falls within fair academic benchmarking (Lee et al. Science 2023). However, commercial deployment of models trained on these specific labels must seek independent legal counsel regarding the proprietary database rights of TGSC and Leffingwell.

---

## 3. Audit 2 — Molecular Preprocessing & Feature Isolation

### 3.1 Preprocessing Correctness
- **Molecule Count:** Exactly `4,983` molecules are parsed, validated, and represented in `molecules.csv`, `molecular_features.csv`, and `odor_labels.csv`.
- **Canonical SMILES:** Generated using RDKit's C++ parser (`Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)`). 
- **Descriptor Extraction Verification:**
  Independent recomputation of the 9 physicochemical descriptors on random samples (`molecular_weight`, `logp`, `tpsa`, `h_bond_donors`, `h_bond_acceptors`, `rotatable_bonds`, `ring_count`, `heavy_atom_count`, `fraction_csp3`) yielded a 100% exact numerical match with `molecular_features.csv`.
- **Morgan Fingerprint Verification:**
  - Type: Circular Morgan bit vector (ECFP4 equivalent)
  - Radius: `2`
  - Bit length: `2048`
  - Chirality: `True` (`useChirality=True` enabled in RDKit)
  - Re-computed bit vectors matched column-for-column with `fp_0` through `fp_2047`.

### 3.2 Target & Identifier Leakage Audit
- **Feature Matrix Dimension:** `(4,983, 2,059)` = 1 `molecule_id` + 1 `canonical_smiles` + 9 physicochemical descriptors + 2,048 Morgan bits.
- **Label Matrix Dimension:** `(4,983, 140)` = 1 `molecule_id` + 1 `canonical_smiles` + 138 binary odor labels.
- **Target Leakage Check:**
  $$\text{Feature Columns} \cap \text{Odor Labels} = \emptyset$$
  None of the 138 sensory descriptors appear as features. No odor descriptors or raw descriptor strings leak into the input feature matrix.

---

## 4. Audit 3 — Data Quality, Stereochemistry & Tautomers

### 4.1 Missing Values & Redundancies
- **Missing Values:** `0` null values across all 697,620 matrix cells.
- **Exact Duplicate Rows:** `0`.
- **Raw SMILES Duplicates:** `0`.
- **Unique Canonical SMILES:** `4,983` unique non-isomeric canonical structures.

### 4.2 Stereochemical & Tautomeric Equivalence
- **Isomeric vs Non-Isomeric SMILES:**
  - Unique non-isomeric canonical SMILES: `4,983`
  - Unique isomeric canonical SMILES: `4,983`
- **Finding:** The curators of `curated_GS_LF_merged_4983.csv` explicitly collapsed all stereoisomeric variants (e.g. cis/trans or R/S enantiomers) to their non-stereochemical parent structures (`nonStereoSMILES`). Consequently, each record represents a unique 2D constitutional graph. No stereochemical duplicates exist in this dataset.

### 4.3 Conflicting Odor Profiles
- Because all 4,983 canonical structures are distinct, there are `0` instances of duplicate structures with conflicting label vectors in this consolidated dataset.

---

## 5. Audit 4 — Bemis-Murcko Scaffold Splitting & Leakage Verification

### 5.1 Recomputation of Scaffolds
Bemis-Murcko scaffolds were independently generated from `canonical_smiles` using `rdkit.Chem.Scaffolds.MurckoScaffold`:
- **Ring-Containing Molecules:** 2,747 molecules distributed across 555 unique ring scaffolds.
- **Acyclic Compounds (Empty Murcko Scaffold `""`):** 2,236 molecules (44.87% of the dataset).
  - *Cheminformatics Nuance:* Standard Murcko scaffold decomposition reduces acyclic molecules to an empty string because they contain no rings.
  - *OlfactAI Handling:* In `create_splits.py`, acyclic molecules are assigned deterministic generic carbon skeleton representations (`acyclic_MakeScaffoldGeneric(mol)`), preventing the 2,236 acyclic compounds from forming a single massive 45% monolithic cluster.

### 5.2 Independent Scaffold Leakage Verification
The intersection of unique scaffolds across the generated partitions was computed:

$$\text{Scaffolds}(\text{Train}) \cap \text{Scaffolds}(\text{Test}) = \emptyset \quad (\text{Count} = 0)$$
$$\text{Scaffolds}(\text{Train}) \cap \text{Scaffolds}(\text{Val}) = \emptyset \quad (\text{Count} = 0)$$
$$\text{Scaffolds}(\text{Val}) \cap \text{Scaffolds}(\text{Test}) = \emptyset \quad (\text{Count} = 0)$$

**Audit Result:** **`ZERO LEAKAGE CONFIRMED`**. Not a single molecular scaffold appears in both the training set and the test set.

### 5.3 Rare Label Coverage under Scaffold Splitting
- **Random Split (70/15/15):** 100% of all 138 odor descriptors have $\ge 1$ positive instance across Train, Val, and Test.
- **Scaffold Split (70/15/15):**
  - Train: 138 / 138 descriptors present.
  - Val: 138 / 138 descriptors present.
  - Test: 135 / 138 descriptors present.
  - Missing in Test: `hawthorn` (34 positive instances total), `juicy` (37 total), `tomato` (40 total).
  - *Scientific Assessment:* Because all carriers of these 3 rare odorants happen to belong to specific scaffold families allocated to Train or Val, they have zero positive samples in Test. This is an unavoidable mathematical property of strict scaffold splitting on sparse multi-label datasets. Evaluator metrics in Phase 3 must compute macro-F1 with zero-division protection or evaluate against the Random Test split for rare-class comparisons.

### 5.4 Test Set Quarantine
- The test partition (`test`, 748 molecules) remains quarantined. It has **not** been used for feature normalization, descriptor scaling, threshold tuning, or model fitting.

---

## 6. Audit 5 — Scientific Claims & Interpretation Review

The Phase 2 Research Report contains several scientific claims that require clarification to separate empirical facts from biological hypotheses:

### Claim 1: "Volatility cut-off (most odorants < 300 g/mol)"
- **Empirical Dataset Fact:** The 95th percentile of molecular weight in this dataset is indeed 298.4 g/mol, and the mean MW is 180.5 g/mol.
- **Scientific Audit Correction:** Molecular weight alone is **not** a sufficient physical determinant of volatility. Volatility is governed by vapor pressure, intermolecular hydrogen bonding, dipole moments, and enthalpy of vaporization ($\Delta H_{vap}$). For example, small diols (e.g. ethylene glycol, MW 62) are far less volatile than heavier non-polar hydrocarbons (e.g. octane, MW 114). Furthermore, volatility only dictates whether a molecule enters the vapor phase; it does not determine binding affinity to human olfactory receptors (ORs) or human detection thresholds, which vary by over 10 orders of magnitude.

### Claim 2: "LogP represents optimal partition through aqueous nasal mucus to OR transmembrane domains"
- **Empirical Dataset Fact:** The dataset exhibits a median LogP of 2.52 with an interquartile range of 1.67 to 3.36.
- **Scientific Audit Correction:** The attribution of this range to "nasal mucus partitioning" is a physiological hypothesis adapted from medicinal chemistry (Lipinski's rule of 5 / pharmacokinetics), not a property directly measured or tested in this dataset. In vivo nasal transport is mediated by Odorant-Binding Proteins (OBPs) that solubilize hydrophobic compounds. LogP in this dataset should be treated as a general physicochemical descriptor of lipophilicity rather than a proven transport parameter.

### Claim 3: "Odor descriptor co-occurrence reflects semantic and perceptual clusters"
- **Empirical Dataset Fact:** Pearson correlations between `fruity` and `sweet` ($r \approx 0.45$), `woody` and `cedar` ($r \approx 0.52$), and `green` and `herbal` ($r \approx 0.38$) are statistically significant in the binary label matrix.
- **Scientific Audit Clarification:** These correlations reflect both chemical similarities (e.g., homologous aliphatic esters smelling fruity/sweet) and human linguistic conventions used by fragrance panelists. They do not demonstrate that both notes share a single olfactory receptor; mammalian olfaction uses combinatorial receptor codes.

---

## 7. Audit 6 — Reproducibility & Environment Assessment

### 7.1 Virtual Environment Identification
- **Interpreter Path:** `C:\Users\DELL\Desktop\OlfactAI\backend\.venv\Scripts\python.exe`
- **Python Version:** `3.11.18` (64-bit AMD64)
- **Environment Findings:**
  - Both `.venv` (at project root) and `backend/.venv` exist on disk.
  - `backend/.venv` contains all 69 packages, including `matplotlib 3.11.2`, `seaborn 0.13.2`, `rdkit 2026.09.1`, and `torch 2.15.0+cpu`.
  - The root `.venv` was created without `matplotlib` and `seaborn`.
  - All test commands and documented scripts must strictly target `backend\.venv\Scripts\python.exe` or `backend\.venv\Scripts\Activate.ps1`.
  - *Recommendation:* Keep `backend/.venv` as the canonical environment as documented in `README.md`.

### 7.2 Idempotency of Acquisition & Preprocessing
- `acquire_dataset.py` was tested against existing files: correctly detects the matching SHA-256 and safely skips re-downloading without overwriting raw files.
- `inspect_dataset.py`, `preprocess_dataset.py`, `create_splits.py`, and `generate_eda.py` execute deterministically and generate identical checksums across repeat executions.

### 7.3 Automated Test Suite Re-Execution
The complete test suite was re-executed:
```
============================= test session starts =============================
platform win32 -- Python 3.11.18, pytest-8.4.2, pluggy-1.7.0
rootdir: C:\Users\DELL\Desktop\OlfactAI
plugins: anyio-4.16.0, cov-5.0.0
collected 52 items

backend/tests/integration/test_api_chemistry.py ......                   [ 11%]
backend/tests/integration/test_api_health.py ..                          [ 15%]
backend/tests/integration/test_pipeline_artifacts.py ....                [ 23%]
backend/tests/unit/test_chemistry_parser.py ...........                  [ 44%]
backend/tests/unit/test_dataset_loaders.py .....                         [ 53%]
backend/tests/unit/test_dataset_splits.py ....                           [ 61%]
backend/tests/unit/test_descriptors.py ...                               [ 67%]
backend/tests/unit/test_digital_nose.py .                                [ 69%]
backend/tests/unit/test_evaluator.py .                                   [ 71%]
backend/tests/unit/test_fingerprints.py ....                             [ 78%]
backend/tests/unit/test_mixtures.py ..                                   [ 82%]
backend/tests/unit/test_ml_components.py ....                            [ 90%]
backend/tests/unit/test_preprocessing_pipeline.py ..                     [ 94%]
backend/tests/unit/test_similarity.py ...                                [100%]

======================== 52 passed, 1 warning in 8.35s ========================
```
**Test Result:** **52 passed, 0 failed.**

---

## 8. Audit 7 — Git Hygiene & Security Audit

- **Git Status:** Working tree clean (`origin/main` at commit `25ca971`).
- **Git Ignore Validation:**
  - `backend/.venv/` and `.venv/` are properly ignored by `.gitignore`.
  - `.env` and `.env.local` are excluded; `.env.example` is committed.
  - Large dataset files (`backend/datasets/raw/*.csv` and `backend/datasets/processed/*.csv`) are strictly ignored.
  - Model weights (`*.joblib`, `*.pt`, `*.onnx`) are ignored.
  - Python byte-cache (`__pycache__`, `*.pyc`) and pytest cache (`.pytest_cache`) are ignored.
- **Security Check:** Zero credentials, API keys, private tokens, or proprietary binaries are staged or committed.

---

## 9. Comprehensive Audit Matrix

| Dimension | Standard Required | Verified Finding | Status |
|---|---|---|---|
| **Raw Integrity** | SHA-256 match with OpenPOM | `0c18b6e9...` verified exactly | **PASS** |
| **Molecule Coverage** | 4,983 molecules | Exactly 4,983 molecules in all matrices | **PASS** |
| **Odor Targets** | 138 binary descriptors | Exactly 138 binary attributes | **PASS** |
| **Target Isolation** | 0 odor targets in features | 0 target leakage detected | **PASS** |
| **Fingerprints** | 2048-bit Morgan (radius 2) | Radius 2, chirality True, 2048 bits verified | **PASS** |
| **Scaffold Leakage** | $\text{Train} \cap \text{Test} = \emptyset$ | 0 overlapping scaffolds across all partitions | **PASS** |
| **Data Quality** | 0 missing values | 0 missing values across all records | **PASS** |
| **Test Suite** | $\ge 52$ tests passing | 52 passed, 0 failed, 0 regressions | **PASS** |
| **CPU Execution** | Zero GPU / CUDA dependencies | PyTorch CPU (`cuda_available=False`) | **PASS** |
| **Git Safety** | No datasets or venvs tracked | Clean `.gitignore` compliance | **PASS** |

---

## 10. Required Corrections & Recommendations for Phase 3

1. **Clarify Scientific Prose in Research Notes:**
   Ensure future reports distinguish measured chemical statistics (MW, LogP) from physiological assumptions (volatility, odor detection thresholds, receptor activation).
2. **Handle Rare Descriptors in Scaffold Evaluation:**
   In Phase 3 evaluation, models tested on the Bemis-Murcko test split must account for the 3 descriptors (`hawthorn`, `juicy`, `tomato`) that have 0 positive test instances, using masked Macro-AUC / Macro-F1.
3. **Environment Standardization:**
   Standardize all execution on `backend/.venv` as documented in `README.md`.
4. **Phase 3 Hand-Off:**
   The data foundation is solid, verified, and ready for baseline model training (Random Forest, XGBoost, and PyTorch CPU MLP).

