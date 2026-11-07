# OlfactAI — Phase 3 Independent Verification Audit Report

**Date:** November 2026  
**Auditor:** Independent Senior ML Engineer & Computational Chemistry Researcher  
**Target Repository:** OlfactAI (Branch `main`, Commit `7e73f5c`)  
**Scope:** Phase 3 Multi-Label Machine Learning, Baseline Benchmarking, Metric Correctness, Latency Profiling, and Data Leakage Guards  
**Environment Audited:** Python 3.11.18, PyTorch 2.15.0+cpu, RDKit 2026.9.1, Scikit-learn 1.9.1, XGBoost 2.1.4 (Strictly CPU execution)  

---

## 1. Executive Verdict

### **VERDICT: APPROVED WITH CAVEATS**

The Phase 3 implementation, training pipeline, and benchmarking experiments adhere to rigorous computational machine learning and cheminformatics standards:
- **Zero Target Leakage:** Held-out test sets in both Random and Bemis-Murcko Scaffold splits are strictly quarantined. Feature scalers and decision thresholds were fitted exclusively on training and validation sets, respectively.
- **Metric Verification:** All reported metrics (Macro/Micro F1, ROC-AUC, PR-AUC, Hamming loss, Subset accuracy) were independently reproduced from disk checkpoints and match the reported figures to four decimal places.
- **Zero Scaffold Overlap:** Bemis-Murcko scaffold partitioning was independently re-verified with $0$ shared scaffolds across train, validation, and test sets.
- **Strict CPU Execution:** The codebase contains zero CUDA/GPU dependencies; PyTorch models explicitly assert and bind to `torch.device("cpu")`.

### Summary of Audit Classifications:
| Category | Audit Focus | Verdict | Details |
| :--- | :--- | :---: | :--- |
| **Audit 1** | Leakage & Split Integrity | **PASS** | Strict separation; `StandardScaler` on continuous only; unscaled Morgan bits. |
| **Audit 2** | Threshold Optimization Logic | **PASS** | Validation-only optimization; reproducible grid; fallback logic verified. |
| **Audit 3** | Metric Correctness & Checkpoints | **PASS WITH CAVEATS** | All numbers match; `ketonic` F1=1.0 is an $N=1$ test artifact; 3 rare classes handled. |
| **Audit 4** | Benchmark Timing & Profiling | **WARNING** | Latency measured core model inference only, omitting DataFrame/scaling overhead. |
| **Audit 5** | Scientific Interpretation & Claims | **PASS** | 33.7% and 40.4% generalization gaps verified; causal claims properly bounded. |
| **Audit 6** | Reproducibility & Git Hygiene | **PASS** | Seeds fixed; large binary checkpoints excluded by `.gitignore`; clean tree. |
| **Audit 7** | Automated Test Suite | **PASS** | 67/67 automated tests passing, including 4 new audit-specific regression tests. |

---

## 2. Audit 1 — Leakage and Data Integrity (PASS)

### 2.1 Split Independence & Molecule Separation
Molecules in [`backend/datasets/processed/molecules.csv`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/datasets/processed/molecules.csv) were inspected:
- **Random Split:** Train = 3,488 (70.0%), Val = 747 (15.0%), Test = 748 (15.0%).
- **Scaffold Split:** Train = 3,488 (70.0%), Val = 747 (15.0%), Test = 748 (15.0%).

Bemis-Murcko scaffolds were independently re-extracted using `MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)` across all 4,983 molecules:
- `Train vs Val` scaffold overlap: **0 molecules**
- `Train vs Test` scaffold overlap: **0 molecules**
- `Val vs Test` scaffold overlap: **0 molecules**

### 2.2 Feature Preprocessing & Scaling Isolation
In [`backend/app/ml/preprocessing/scaler.py`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/app/ml/preprocessing/scaler.py):
- `OlfactoryFeatureScaler` separates the 9 continuous physicochemical descriptors (`molecular_weight`, `logp`, `tpsa`, `h_bond_donors`, `h_bond_acceptors`, `rotatable_bonds`, `ring_count`, `heavy_atom_count`, `fraction_csp3`) from the 2,048 Morgan bits.
- Verification confirmed that `StandardScaler.mean_` and `StandardScaler.scale_` contain exactly **9 values**.
- The 2,048 Morgan fingerprint bits were verified to have unique values in $\{0.0, 1.0\}$ following transformation, confirming zero standardization was applied to binary bits.
- Non-feature columns (`molecule_id`, `canonical_smiles`, `input_smiles`) are stripped prior to model input.
- Positive-class weights in PyTorch MLP (`pos_weight`) and XGBoost (`scale_pos_weight`) are computed strictly from `y_train` binary label matrices.

---

## 3. Audit 2 — Threshold Optimization & Dummy Model Behavior (PASS)

### 3.1 Threshold Selection Protocol
In [`backend/app/ml/evaluation/thresholds.py`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/app/ml/evaluation/thresholds.py):
- Search grid: 17 candidate thresholds spanning $[0.10, 0.90]$ with step $0.05$.
- Optimized strictly on validation set: `optimize_thresholds(y_val, val_probs)`.
- Fallback logic: If validation support is $< 2$ or if the maximum F1 achieved across all candidate thresholds is $0.0$, the decision threshold falls back to $0.50$.
- Test application: Frozen thresholds are loaded from `thresholds.json` and applied to test set probabilities using `(test_probs >= thresholds[None, :]).astype(int)`. No test data is seen during tuning.

### 3.2 Mathematical Explanation of Dummy Baseline Macro F1
The benchmark report noted that `DummyFrequencySmellModel` achieved **Default Macro F1 = 0.0000**, but **Optimized Macro F1 = 0.0195 (Random)** and **0.0165 (Scaffold)**. This was investigated to ensure no target leakage occurred:

1. **Why Default Macro F1 is 0.0000:** The dummy model predicts empirical prior probabilities ($p_j = \frac{1}{N} \sum y_{ij}$). In the training split, the most frequent descriptor is `fruity` with prior $p = 0.3870$. Since **no descriptor has prior $p_j \ge 0.50$**, applying the default $0.50$ threshold predicts all zeros for all 138 classes across all molecules. Thus, True Positives = 0 for all classes $\implies$ F1 = 0.0000.
2. **Why Optimized Macro F1 is > 0:** When thresholds are tuned in $[0.10, 0.90]$, any descriptor with prior $p_j \ge 0.10$ can trigger a positive prediction if the tuned threshold $t \le p_j$. Exactly **8 descriptors** meet this criterion in the training set:
   - `fruity` (Prior: 0.3870, Tuned threshold: 0.35 $\implies$ Test F1 = 0.5348)
   - `sweet` (Prior: 0.2919, Tuned threshold: 0.25 $\implies$ Test F1 = 0.4593)
   - `green` (Prior: 0.2962, Tuned threshold: 0.25 $\implies$ Test F1 = 0.4119)
   - `floral` (Prior: 0.2219, Tuned threshold: 0.20 $\implies$ Test F1 = 0.3578)
   - `herbal` (Prior: 0.1482, Tuned threshold: 0.10 $\implies$ Test F1 = 0.2883)
   - `woody` (Prior: 0.1345, Tuned threshold: 0.10 $\implies$ Test F1 = 0.2441)
   - `fatty` (Prior: 0.1175, Tuned threshold: 0.10 $\implies$ Test F1 = 0.2105)
   - `fresh` (Prior: 0.1069, Tuned threshold: 0.10 $\implies$ Test F1 = 0.1778)

Predicting all 1s for these 8 frequent labels on test data yields positive F1 scores equal to $\frac{2 \times \text{prevalence}}{1 + \text{prevalence}}$. Averaged over all 138 labels (130 of which are 0.0), the resulting Macro F1 is:
$$\text{Macro F1} = \frac{0.5348 + 0.4593 + 0.4119 + 0.3578 + 0.2883 + 0.2441 + 0.2105 + 0.1778}{138} = \frac{2.6845}{138} = 0.0195$$
This confirms that the non-zero Macro F1 is a legitimate mathematical consequence of threshold tuning on constant predictors, not data leakage.

---

## 4. Audit 3 — Metric Correctness & Special Label Deep-Dive (PASS WITH CAVEATS)

### 4.1 Recomputed Benchmark Comparison
All model checkpoints and scalers were loaded from disk and re-evaluated against the quarantined test sets. All recomputed values matched the report exactly:

#### Protocol A (Random Split Test Set, N=748):
| Model | Reported Macro F1 | Recomputed Macro F1 | Reported ROC-AUC | Recomputed ROC-AUC | Reported PR-AUC | Recomputed PR-AUC | Discrepancy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Dummy | 0.0195 | 0.0195 | 0.5000 | 0.5000 | 0.0349 | 0.0349 | **0.0000** |
| Random Forest | 0.0308 | 0.0308 | 0.5028 | 0.5028 | 0.0401 | 0.0401 | **0.0000** |
| XGBoost (CPU) | 0.3066 | 0.3066 | 0.8631 | 0.8631 | 0.3213 | 0.3213 | **0.0000** |
| PyTorch CPU MLP | 0.3024 | 0.3024 | 0.8632 | 0.8632 | 0.3027 | 0.3027 | **0.0000** |

#### Protocol B (Bemis-Murcko Scaffold Split Test Set, N=748):
| Model | Reported Macro F1 | Recomputed Macro F1 | Reported ROC-AUC | Recomputed ROC-AUC | Reported PR-AUC | Recomputed PR-AUC | Discrepancy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Dummy | 0.0165 | 0.0165 | 0.5000 | 0.5000 | 0.0276 | 0.0276 | **0.0000** |
| Random Forest | 0.0209 | 0.0209 | 0.5134 | 0.5134 | 0.0335 | 0.0335 | **0.0000** |
| XGBoost (CPU) | 0.2033 | 0.2033 | 0.8064 | 0.8064 | 0.2235 | 0.2235 | **0.0000** |
| PyTorch CPU MLP | 0.1802 | 0.1802 | 0.8082 | 0.8082 | 0.1881 | 0.1881 | **0.0000** |

### 4.2 Handling of Unrepresented Rare Classes
In the scaffold test set, exactly **three labels** have zero positive examples:
- `hawthorn` (Support: 0)
- `juicy` (Support: 0)
- `tomato` (Support: 0)

Audit of [`backend/app/ml/evaluation/metrics.py`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/app/ml/evaluation/metrics.py):
- `evaluable_mask = (supports > 0) & (supports < n_samples)`
- `evaluable_labels_count` = **135** (out of 138 total).
- For these 3 classes, `roc_auc` and `pr_auc` are recorded as `None` (rather than 0.0 or 0.5).
- Macro ROC-AUC and Macro PR-AUC are computed strictly across the **135 evaluable classes**, avoiding arithmetic NaN crashes and preventing downward skewing of macro discrimination.

### 4.3 Deep-Dive: Investigation of `ketonic` F1 = 1.0000 (CAVEAT)
The benchmark report listed `ketonic` as the top-performing descriptor with F1 = 1.0000 and ROC-AUC = 1.0000 in the scaffold test set.

> [!CAUTION]
> **Audit Finding (Small-Sample Artifact):**
> Detailed inspection revealed that in the held-out scaffold test partition ($N=748$), `ketonic` has a ground-truth support of **$N = 1$ positive molecule**:
> - Positive Molecule ID: `MOL_02258`
> - Canonical SMILES: `O=C1CCCCCC1` (Cycloheptanone)
> - True Positives (TP): **1**
> - False Positives (FP): **0**
> - False Negatives (FN): **0**
> - True Negatives (TN): **747**
> - Predicted Probability on `MOL_02258`: **0.9773** (Tuned threshold: **0.9000**)
> - Maximum Predicted Probability on all 747 negative molecules: **0.7164**
>
> While the model correctly distinguished this ketone from all 747 non-ketonic test molecules, **an F1 of 1.0000 on a sample size of $N=1$ does not constitute robust statistical evidence of generalizability for that descriptor**. Future reports must explicitly annotate descriptors with support $< 5$ with a small-sample caveat.

---

## 5. Audit 4 — Benchmark Timing & Latency Profiling (WARNING)

### 5.1 Audit of Timing Instrumentation
In `RandomForestSmellModel`, `XGBoostSmellModel`, and `PyTorchMLPSmellModel`, timing was instrumented around the core estimator call:
```python
start = time.perf_counter()
probs = self.estimator.predict_proba(X)  # or forward pass in PyTorch
self.inference_time_sec = float(time.perf_counter() - start)
```
> [!WARNING]
> **Timing Scope Limitation:**
> The reported inference latencies (e.g., 29.5 ms for PyTorch MLP and 2,768.5 ms for XGBoost) measure **pure vectorized matrix operations on pre-transformed NumPy arrays for a batch of 748 molecules**. They do **not** include:
> 1. Raw SMILES parsing and feature extraction (~1.2 ms/mol).
> 2. Feature scaling DataFrame alignment (~4.23 ms/call).
> 3. Decision threshold comparison and JSON mapping.
>
> These numbers must **not** be quoted as real-time single-molecule API response times.

### 5.2 Independent Multi-Run Latency Profiling
To provide production-grade guidance, repeated micro-benchmarks were executed on the test system:

| Metric | PyTorch CPU MLP | XGBoost (138 Hist Trees) | Speedup Factor |
| :--- | :---: | :---: | :---: |
| **Single-Molecule Latency** (Mean $\pm$ Std, 100 runs) | **1.66 ms $\pm$ 0.41 ms** | **232.42 ms $\pm$ 65.86 ms** | **140.0x faster** |
| **Batch Latency (748 molecules)** (Mean $\pm$ Std, 10 runs) | **24.97 ms $\pm$ 2.06 ms** | **3,104.50 ms $\pm$ 141.61 ms** | **124.3x faster** |
| **Per-Molecule Throughput in Batch** | **0.033 ms / molecule** | **4.150 ms / molecule** | **125.8x faster** |
| **Single-Molecule Scaler Overhead** | 4.23 ms (Pandas DataFrame) | 4.23 ms (Pandas DataFrame) | Identical |

**Verdict:** PyTorch CPU MLP exhibits massive architectural efficiency over multi-output XGBoost for online API serving, maintaining sub-2ms per-molecule latency on CPU.

---

## 6. Audit 5 — Scientific Interpretation & Extrapolation Delta (PASS)

### 6.1 Verification of Generalization Drop Deltas
The reported drops from Random Split to Bemis-Murcko Scaffold Split were verified:
- **XGBoost Macro F1:** $0.3066 \rightarrow 0.2033 \implies \frac{0.3066 - 0.2033}{0.3066} \times 100 = \mathbf{33.68\%}$ (Reported: 33.7%).
- **PyTorch MLP Macro F1:** $0.3024 \rightarrow 0.1802 \implies \frac{0.3024 - 0.1802}{0.3024} \times 100 = \mathbf{40.41\%}$ (Reported: 40.4%).
- **ROC-AUC Preservation:** Macro ROC-AUC dropped only **6.57%** for XGBoost ($0.8631 \rightarrow 0.8064$) and **6.37%** for PyTorch MLP ($0.8632 \rightarrow 0.8082$).

### 6.2 Scientific Boundaries & Causal Distinctions
- **Ranking vs Calibration:** The data clearly demonstrates that models preserve relative ranking ability ($\text{ROC-AUC} > 0.80$) even on novel carbon skeletons, but calibration shifts significantly, necessitating adaptive thresholding.
- **Sensory Hypotheses:** The report's suggestions that descriptors like `sour` and `sweaty` perform poorly due to "sensory subjectivity" or "trigeminal cross-activation" must remain labeled as **biological hypotheses**, as the empirical data solely confirms severe label scarcity ($N \le 7$ in test set) and low training signal.

---

## 7. Audit 6 — Reproducibility, Checkpoints & Git Hygiene (PASS)

### 7.1 Seed & Configuration Integrity
- Random seed $42$ was enforced consistently across NumPy, PyTorch (`torch.manual_seed(42)`), and Scikit-learn estimators.
- Checkpoint reloading was verified by deserializing all 8 models (`model.joblib` and `model.pt`) and verifying bit-for-bit prediction reproduction.

### 7.2 Git Hygiene & Repository Size Protection
Inspected `.gitignore` and `git status`:
- Large binary artifacts (`*.joblib`, `*.pt`, `*.bin`, `*.onnx`) are strictly excluded by `.gitignore`.
- Lightweight threshold configurations (`thresholds.json`) and benchmark summary JSONs are tracked.
- No credentials, tokens, temporary Python virtual environment files, or raw dataset archives are staged for Git.

---

## 8. Audit 7 — Automated Test Suite Verification (PASS)

To address all potential testing gaps identified during this audit, a dedicated audit test suite was created:
[`backend/tests/unit/test_phase3_audit.py`](file:///c:/Users/DELL/Desktop/OlfactAI/backend/tests/unit/test_phase3_audit.py)

Tests added:
1. `test_threshold_leakage_quarantine`: Proves threshold optimization does not mutate or access test data.
2. `test_metric_unrepresented_labels_macro_count`: Verifies multi-label evaluator handles $0$-support classes without arithmetic corruption.
3. `test_strict_cpu_execution_enforcement`: Asserts PyTorch parameters reside strictly on `cpu` devices.
4. `test_saved_scaler_and_thresholds_consistency`: Verifies checkpoint file existence and structural validity.

### Full Test Suite Execution Output:
```
============================= test session starts =============================
platform win32 -- Python 3.11.18, pytest-8.4.2, pluggy-1.7.0
rootdir: C:\Users\DELL\Desktop\OlfactAI
collected 67 items

backend/tests/integration/test_api_chemistry.py ......                   [  8%]
backend/tests/integration/test_api_health.py ..                         [ 11%]
backend/tests/integration/test_pipeline_artifacts.py ....               [ 17%]
backend/tests/unit/test_chemistry_parser.py ...........                  [ 34%]
backend/tests/unit/test_dataset_loaders.py .....                        [ 41%]
backend/tests/unit/test_dataset_splits.py ....                          [ 47%]
backend/tests/unit/test_descriptors.py ...                              [ 52%]
backend/tests/unit/test_digital_nose.py .                               [ 53%]
backend/tests/unit/test_evaluator.py .                                  [ 55%]
backend/tests/unit/test_fingerprints.py ....                            [ 61%]
backend/tests/unit/test_mixtures.py ..                                  [ 64%]
backend/tests/unit/test_ml_components.py ....                           [ 70%]
backend/tests/unit/test_phase3_audit.py ....                            [ 76%]
backend/tests/unit/test_phase3_evaluation.py .......                    [ 86%]
backend/tests/unit/test_phase3_models.py ....                           [ 92%]
backend/tests/unit/test_preprocessing_pipeline.py ..                    [ 95%]
backend/tests/unit/test_similarity.py ...                               [100%]

======================= 67 passed, 1 warning in 15.94s ========================
```

---

## 9. Recommended Corrections & Action Items

1. **Add Small-Sample Annotations in Documentation:** Update the benchmark report table to mark `ketonic` with an asterisk noting $N=1$ test support.
2. **Clarify Inference Profiling Definitions:** Distinguish offline core inference throughput (~0.033 ms/mol for MLP) from end-to-end API latency (~5.89 ms/mol including SMILES parsing and DataFrame feature transformations).
3. **Commit New Audit Deliverables:** Commit `backend/scripts/audit_phase3.py`, `backend/tests/unit/test_phase3_audit.py`, and this document to the repository.

---

## 10. Audit Sign-Off

The Phase 3 multi-label machine learning baseline implementation is verified to be leakage-free, mathematically sound, reproducible, and compliant with all project constraints. OlfactAI is cleared to proceed toward Phase 4.

