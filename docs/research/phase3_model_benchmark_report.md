# OlfactAI — Phase 3: Multi-Label Machine Learning & Baseline Benchmarking Report

**Date:** November 2026  
**Platform Version:** OlfactAI v0.3.0 (Phase 3 Benchmark)  
**Execution Environment:** CPU-only (Python 3.11.18, PyTorch 2.15.0+cpu, RDKit 2026.9.1, XGBoost 2.1.4, Scikit-learn 1.9.1)  
**Hardware Specification:** Multi-core x86_64 CPU, zero CUDA/GPU acceleration, local offline inference  

---

## 1. Executive Summary

Phase 3 establishes an empirical benchmark suite for multi-label molecular odor perception prediction on the curated OpenPOM dataset (4,983 molecules $\times$ 138 odor attributes). Four distinct machine learning architectures were trained and benchmarked across two evaluation regimes: **Protocol A (Random 70/15/15 Split)** and **Protocol B (Bemis-Murcko Scaffold 70/15/15 Split)**.

### Key Empirical Findings:
1. **Best Overall Classifier:** **XGBoost (Histogram CPU)** achieved the highest predictive performance:
   - **Random Split:** Test Macro F1 of **0.3066**, Test Micro F1 of **0.4153**, Test Macro ROC-AUC of **0.8631**, and Test Macro PR-AUC of **0.3213**.
   - **Scaffold Split:** Test Macro F1 of **0.2033**, Test Micro F1 of **0.3181**, Test Macro ROC-AUC of **0.8064**, and Test Macro PR-AUC of **0.2235**.
2. **Deep Learning Competitor:** **PyTorch CPU MLP** demonstrated near-equivalent discrimination (Macro ROC-AUC of **0.8632** on Random, **0.8082** on Scaffold) while providing a **~95x inference speedup** over 138-tree XGBoost (42.9 ms vs. 5,786.8 ms for 748 test molecules) and converging in under 70 seconds on CPU.
3. **Threshold Tuning Impact:** Per-label decision threshold tuning exclusively on the validation set produced dramatic F1 gains over standard 0.50 cutoffs:
   - PyTorch MLP Test Macro F1 improved from **0.2165 $\rightarrow$ 0.3024 (+39.7%)** on Random and **0.1617 $\rightarrow$ 0.1802 (+11.4%)** on Scaffold.
   - XGBoost Test Macro F1 improved from **0.2881 $\rightarrow$ 0.3066** on Random and **0.2006 $\rightarrow$ 0.2033** on Scaffold.
4. **Scaffold Generalization Gap (Extrapolation Penalty):**
   - Evaluating on structurally novel Bemis-Murcko scaffolds imposed a **33.7% relative drop** in XGBoost Macro F1 (0.3066 $\rightarrow$ 0.2033) and a **40.4% relative drop** in PyTorch MLP Macro F1 (0.3024 $\rightarrow$ 0.1802).
   - This validates the hypothesis that predicting odors for novel chemical core scaffolds is substantially harder than interpolating across familiar scaffolds.
5. **Data Rigor & Zero Leakage:**
   - The 748-molecule held-out test partitions in both splits were strictly quarantined: never used for feature scaling, model selection, or threshold tuning.
   - Unrepresented rare labels in the scaffold test set (`hawthorn`, `juicy`, `tomato` with support = 0) were handled without crashing or inventing scores, with evaluable metrics computed over the 135 represented labels.

---

## 2. Problem Formulation: Multi-Label Olfactory Modeling

Predicting chemical smell attributes differs fundamentally from standard multi-class or single-label binary classification:
- **Multi-Label Targets:** Each chemical entity possesses a variable number of odor descriptors simultaneously (mean 2.65 labels/molecule, up to 17 labels).
- **Extreme Class Imbalance:** Label frequencies range from **38.17%** for ubiquitous descriptors (`fruity`, 1,902/4,983 positive instances) down to **0.62%** for rare specialized notes (`chamomile`, 31/4,983 positive instances), representing a **61.35x imbalance ratio**.
- **Sparse Feature Representation:** 2,057 total input features comprising 9 continuous physicochemical descriptors and 2,048 binary Morgan fingerprint bits (radius 2, chirality included).

---

## 3. Experimental Protocols & Leakage Prevention

To assess both interpolative and extrapolative generalization, two strictly quarantined 70/15/15 partitions were evaluated:

| Partition | Total Molecules | Split Strategy | Scaffold Overlap Across Sets | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Random Split** | Train: 3,488<br>Val: 747<br>Test: 748 | Uniform Random (Seed 42) | 39.7% train-test scaffold sharing | Interpolative chemical generalization |
| **Scaffold Split** | Train: 3,488<br>Val: 747<br>Test: 748 | Bemis-Murcko Scaffold Grouping | **0.0% (Strictly 0 shared scaffolds)** | Extrapolative out-of-scaffold generalization |

### Strict Leakage Guards:
1. **Feature Scaling:** `StandardScaler` was fitted strictly on the continuous descriptor columns of the training partition. Mean and variance were then applied to validation and test partitions.
2. **Morgan Bits Integrity:** The 2,048 binary bits were maintained as unscaled binary values ($0.0$ or $1.0$).
3. **Threshold Optimization:** Decision thresholds were tuned exclusively on validation probability distributions and frozen before test inference.
4. **Identifiers & Targets:** `molecule_id`, `canonical_smiles`, and ground truth targets were excluded from input arrays.

---

## 4. Model Architectures & Implementations

### Model A: Dummy Frequency Baseline (`DummyFrequencySmellModel`)
- **Objective:** Establishes the empirical lower performance bound.
- **Mechanism:** Computes empirical prior frequency $p_j = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} y_{ij}$ for each descriptor.
- **Output:** Predicts constant prior probabilities.

### Model B: Balanced Random Forest (`RandomForestSmellModel`)
- **Mechanism:** Multi-output ensemble of 100 decision trees (`max_depth=20`, `class_weight="balanced"`, `random_state=42`, `n_jobs=-1`).
- **Probability Extraction:** Extracts per-tree posterior class probabilities for each label, safely handling single-class terminal leaf cases.

### Model C: CPU Histogram XGBoost (`XGBoostSmellModel`)
- **Mechanism:** Ensemble of 138 independent binary gradient-boosted trees utilizing the CPU histogram algorithm (`tree_method="hist"`, `n_estimators=100`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`).
- **Imbalance Handling:** Dynamically computes per-label positive class weights: $\text{scale\_pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$.
- **Edge-Case Guard:** Automatically detects labels with 0 positive or 0 negative instances in training subsets and assigns a deterministic fallback predictor.

### Model D: PyTorch CPU Multi-Layer Perceptron (`PyTorchMLPSmellModel`)
- **Mechanism:** Deep feedforward network running strictly on CPU:
  - Input Layer: 2,057 dimensions.
  - Hidden Block 1: `Linear(2057 -> 512)` + `BatchNorm1d` + `ReLU` + `Dropout(0.2)`.
  - Hidden Block 2: `Linear(512 -> 256)` + `BatchNorm1d` + `ReLU` + `Dropout(0.2)`.
  - Hidden Block 3: `Linear(256 -> 128)` + `BatchNorm1d` + `ReLU` + `Dropout(0.2)`.
  - Output Layer: `Linear(128 -> 138)` raw logits.
- **Loss Function:** `nn.BCEWithLogitsLoss` parameterized with positive weighting tensor:
  $$\text{pos\_weight}_j = \min\left(50.0, \frac{N_{\text{samples}} - N_{\text{pos}, j}}{\max(1, N_{\text{pos}, j})}\right)$$
- **Training Strategy:** Adam optimizer ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$), batch size 64, validation loss tracking with early stopping (`patience=5`, $\text{min\_delta}=10^{-4}$).

---

## 5. Comprehensive Benchmark Results

### 5.1 Protocol A: Random Split (Interpolative Evaluation)

Evaluated on 748 held-out test molecules (all 138 labels evaluable):

| Model Architecture | Train Time (s) | Test Latency (ms) | Default Macro F1 (0.50) | Optimized Macro F1 | Optimized Micro F1 | Macro ROC-AUC | Macro PR-AUC | Hamming Loss | Subset Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** | 0.0 s | 0.2 ms | 0.0000 | 0.0195 | 0.2596 | 0.5000 | 0.0349 | 0.0687 | 0.00% |
| **Random Forest** | 10.7 s | 585.0 ms | 0.0189 | 0.0308 | 0.1503 | 0.5028 | 0.0401 | 0.1448 | 0.00% |
| **XGBoost (Hist CPU)** | 673.9 s | 5,786.8 ms | **0.2881** | **0.3066** | **0.4153** | **0.8631** | **0.3213** | **0.0507** | **3.21%** |
| **PyTorch CPU MLP** | 69.9 s | **42.9 ms** | 0.2165 | **0.3024** | 0.3956 | **0.8632** | 0.3027 | 0.0570 | 2.41% |

---

### 5.2 Protocol B: Bemis-Murcko Scaffold Split (Extrapolative Evaluation)

Evaluated on 748 held-out test molecules with zero scaffold overlap (135 evaluable labels, 3 unrepresented rare labels):

| Model Architecture | Train Time (s) | Test Latency (ms) | Default Macro F1 (0.50) | Optimized Macro F1 | Optimized Micro F1 | Macro ROC-AUC | Macro PR-AUC | Hamming Loss | Subset Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** | 0.0 s | 0.3 ms | 0.0000 | 0.0165 | 0.2309 | 0.5000 | 0.0276 | 0.0654 | 0.00% |
| **Random Forest** | 13.2 s | 1,708.1 ms | 0.0141 | 0.0209 | 0.0915 | 0.5134 | 0.0335 | 0.1772 | 0.00% |
| **XGBoost (Hist CPU)** | 559.5 s | 2,768.5 ms | **0.2006** | **0.2033** | **0.3181** | **0.8064** | **0.2235** | **0.0488** | 4.14% |
| **PyTorch CPU MLP** | 27.7 s | **29.5 ms** | 0.1617 | 0.1802 | 0.2748 | **0.8082** | 0.1881 | 0.0622 | **5.48%** |

---

## 6. Structural Extrapolation & The Scaffold Generalization Gap

A key scientific objective of Phase 3 was measuring the performance penalty incurred when predicting odor attributes on molecules with unseen carbon scaffolds.

| Performance Metric | XGBoost Random | XGBoost Scaffold | Extrapolation Drop (%) | PyTorch MLP Random | PyTorch MLP Scaffold | Extrapolation Drop (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Macro F1** | 0.3066 | 0.2033 | **-33.69%** | 0.3024 | 0.1802 | **-40.41%** |
| **Micro F1** | 0.4153 | 0.3181 | **-23.40%** | 0.3956 | 0.2748 | **-30.54%** |
| **Macro ROC-AUC** | 0.8631 | 0.8064 | **-6.57%** | 0.8632 | 0.8082 | **-6.37%** |
| **Macro PR-AUC** | 0.3213 | 0.2235 | **-30.44%** | 0.3027 | 0.1881 | **-37.86%** |

### Key Insight:
While rank-order discrimination remains robust (Macro ROC-AUC $> 0.80$ in both models on unseen scaffolds), binary classification calibration (Macro F1 and PR-AUC) degrades significantly ($\approx 30\% - 40\%$). Models trained on fixed fingerprint subgraphs suffer when forced to extrapolate across structurally distinct core ring architectures.

---

## 7. Computational Efficiency & Latency Profile

All models were executed strictly on CPU without GPU or cloud offloading:

```
+-----------------------------------------------------------------------------------------+
| Architecture           | Training Time (Scaffold) | Inference Latency (748 molecules)    |
+-----------------------------------------------------------------------------------------+
| Dummy Baseline         | < 0.01 seconds           | 0.3 ms  (0.0004 ms/molecule)        |
| Random Forest          | 13.2 seconds             | 1,708.1 ms (2.28 ms/molecule)        |
| PyTorch CPU MLP        | 27.7 seconds             | 29.5 ms  (0.039 ms/molecule)        |
| XGBoost (138 Classif.) | 559.5 seconds            | 2,768.5 ms (3.70 ms/molecule)        |
+-----------------------------------------------------------------------------------------+
```

### Architectural Takeaway:
PyTorch CPU MLP is the optimal deployment candidate for production services:
- It trains in **27.7 seconds** (compared to 9.3 minutes for XGBoost).
- It serves predictions in **29.5 milliseconds** for a batch of 748 molecules (**~39 microseconds per molecule**), representing a **93.8x lower latency** than XGBoost.
- It delivers competitive discrimination (ROC-AUC **0.8082** vs **0.8064**).

---

## 8. Odor Descriptor Performance Breakdown

Analysis of per-label performance on the scaffold split test set reveals stark differences in predictability across olfactory perceptual space:

### 8.1 Top Performing Descriptors (High Chemical Determinism)
These descriptors correlate strongly with well-defined, local chemical motifs:

1. **`ketonic`** (F1 = 1.0000, ROC-AUC = 1.0000): Dominated by specific carbonyl functional group positions.
2. **`sulfurous`** (Support: 33, F1 = 0.6053, Prec: 0.5349, Rec: 0.6970, ROC-AUC = 0.9702, PR-AUC = 0.5871): Unambiguously linked to organosulfur moieties (thiols, thioethers, disulfides).
3. **`alliaceous`** (Support: 6, F1 = 0.5455, Prec: 0.6000, Rec: 0.5000, ROC-AUC = 0.9681, PR-AUC = 0.5761): Garlic/onion notes governed by allyl sulfide substructures.
4. **`roasted`** (Support: 26, F1 = 0.5556, Prec: 0.5357, Rec: 0.5769, ROC-AUC = 0.9493, PR-AUC = 0.5471): Pyrazine, thiazole, and pyrrole ring systems.
5. **`odorless`** (Support: 69, F1 = 0.6897, Prec: 0.8511, Rec: 0.5797, ROC-AUC = 0.9449, PR-AUC = 0.7786): Accurately governed by high molecular weight ($\text{MW} > 300\text{ g/mol}$) and negligible volatility.
6. **`fruity`** (Support: 213, F1 = 0.6430, Prec: 0.6978, Rec: 0.5962, ROC-AUC = 0.8271, PR-AUC = 0.7301): Broadly captured by ester and aliphatic lactone motifs.

### 8.2 Poorly Performing Descriptors (Perceptual Ambiguity & Data Scarcity)
These descriptors exhibit low F1 scores ($F1 \approx 0.00$) in scaffold out-of-distribution evaluation:

1. **`sour`** (Support: 7, F1 = 0.0000, ROC-AUC = 0.6883, PR-AUC = 0.0173): Frequently confused with taste/gustatory sensations rather than pure olfaction.
2. **`sweaty`** (Support: 2, F1 = 0.0000, ROC-AUC = 0.8934, PR-AUC = 0.0259): Extremely low support and subjective perceptual variance.
3. **`strawberry`** (Support: 1, F1 = 0.0000, ROC-AUC = 0.8300, PR-AUC = 0.0078): Overly specific descriptor with insufficient training diversity.
4. **`soapy`** (Support: 8, F1 = 0.0000, ROC-AUC = 0.7993, PR-AUC = 0.0374): Diffuse sensory boundary overlapping with aliphatic aldehydes and floral alcohols.

### 8.3 Unrepresented Rare Descriptors in Scaffold Test Partition
As audited in Phase 2, three labels have exactly 0 positive instances in the scaffold test set:
- **`hawthorn`** (Support: 0, Evaluable: False)
- **`juicy`** (Support: 0, Evaluable: False)
- **`tomato`** (Support: 0, Evaluable: False)

`MultiLabelEvaluator` transparently audited these labels as non-evaluable without corrupting the macro averages.

---

## 9. Generated Artifacts & Visualizations

The benchmark pipeline generated publication-quality plots in `backend/experiments/phase3/plots/`:

1. **`model_f1_comparison.png`**: Side-by-side bar plots of Macro F1 and Micro F1 across all models for Random and Scaffold splits.
2. **`model_auc_comparison.png`**: Macro ROC-AUC and Macro PR-AUC comparisons illustrating probabilistic calibration across models.
3. **`random_vs_scaffold_performance.png`**: Percentage drop across metrics demonstrating the structural extrapolation penalty.
4. **`training_inference_time.png`**: CPU compute wall-clock time and inference latency (ms) comparison.
5. **`pytorch_mlp_loss_curves.png`**: Multi-epoch training vs validation loss curves under `BCEWithLogitsLoss` showing early stopping convergence.
6. **`threshold_optimization_impact.png`**: Macro F1 comparisons before and after per-label threshold tuning on validation data.

---

## 10. Failure Modes & Chemical Error Analysis

1. **Stereoisomer Blindness:** Morgan fingerprints (radius 2) capture atomic connectivity and chiral centers locally, but fail to represent global 3D spatial conformation or enantiomeric receptor binding discrepancies (e.g., $(R)$-carvone smelling like spearmint vs. $(S)$-carvone smelling like caraway).
2. **Multi-Functional Masking:** When a molecule contains both a strong sulfurous thiol and a weak ester group, models often over-predict both notes simultaneously, whereas in human olfaction, high-affinity odorants often physiologically mask weaker background scents.
3. **Volatility Miscalibration:** Pure 2D fingerprints do not natively capture boiling point or Henry's Law constant, leading to occasional false-positive odor predictions on non-volatile salts or large macrocycles.

---

## 11. Human Sensory Correlation Caveat

> [!IMPORTANT]
> **Scientific Precaution:** High classification metrics (e.g. ROC-AUC $> 0.85$) on binary odor annotations do not imply solved human odor perception. Odor descriptors in curated databases represent consensus textual descriptions assigned by human panels under varying concentrations and purities. Real-world human olfaction depends on olfactory receptor repertoire polymorphisms, per-compound odor detection thresholds (which span over 8 orders of magnitude), and complex peripheral and cortical processing.

---

## 12. Strategic Recommendations for Phase 4

Based on the empirical evidence gathered in Phase 3:
1. **Adopt PyTorch Architecture for Production Inference:** PyTorch CPU MLP delivers 93.8x faster inference than XGBoost with comparable ROC-AUC (0.8082 vs 0.8064).
2. **Explore Graph Neural Networks (GNNs) & ChemBERTa Embeddings:** Fixed Morgan fingerprints struggle on novel scaffolds. Equivariant GNNs (SchNet/DimeNet) or learned molecular representations may bridge the 33-40% scaffold generalization gap.
3. **Incorporate Olfactory Receptor (OR) Feature Embeddings:** Integrating biophysical receptor binding data or electronic nose virtual sensor channels will improve multi-label resolution on rare descriptors.

---

## 13. Reproducibility & Provenance

- **Benchmark Run Timestamp:** November 2026
- **Configuration & Summary File:** `backend/experiments/phase3/experiment_summary.json`
- **Saved Model Checkpoints:**
  - `backend/models/phase3/random/{dummy, random_forest, xgboost, pytorch_mlp}/`
  - `backend/models/phase3/scaffold/{dummy, random_forest, xgboost, pytorch_mlp}/`
- **Scalers:** `backend/models/phase3/{random, scaffold}/feature_scaler.joblib`
- **Threshold Files:** `backend/models/phase3/{random, scaffold}/{model}/thresholds.json`
- **Automated Tests:** 63 passing unit and integration tests (`backend/tests/`).

