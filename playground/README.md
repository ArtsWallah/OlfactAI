# OlfactAI — Research Playground

An interactive testing interface for OlfactAI's computational chemistry, machine learning models, and chemical space discoveries (Phases 1–5).

> **Important:** This is an internal research validation interface for developers and scientific reviewers, not the final production frontend.

---

## Capabilities

1. **SMILES Input & Molecular Analysis:**
   - Real-time 2D molecular structure rendering via RDKit.
   - Comprehensive physicochemical descriptors: MW, LogP, TPSA, HBD, HBA, RotBonds, Rings, Heavy Atoms, Fraction CSP3.
   - Presets for benchmark olfactory molecules (Vanillin, Eugenol, Isoamyl acetate, Benzyl alcohol, Geosmin, Cubane).

2. **Multi-Model Odor Prediction:**
   - Evaluates Phase 4 GNN (MPNN) alongside Phase 3 XGBoost and PyTorch MLP models.
   - Supports both `Random` and `Scaffold` split models.
   - Applies validation-optimized thresholds (`thresholds.json`).
   - Displays Temperature-Calibrated probabilities alongside raw sigmoid outputs.
   - Multi-model consensus comparison.

3. **Molecular Similarity Engine:**
   - Nearest-neighbor search over the 4,983-molecule curated corpus.
   - Top-K selector ($K \in \{5, 10, 20\}$).
   - Reports Tanimoto scores, canonical structures, and known odor annotations.
   - Strict identity-based self-match exclusion.

4. **Structural Neighborhood Analysis:**
   - Neighborhood statistics: Mean & median structural Tanimoto, Mean & median odor Jaccard, Zero-odor overlap percentage.
   - Neighborhood descriptor prevalence bar chart.
   - Annotation sparsity caveats.

5. **2D Chemical Space Manifold (UMAP):**
   - Renders precomputed 4,983-molecule 2D UMAP space ($2,048$-bit Morgan fingerprints, Jaccard distance).
   - Highlights the query molecule with a prominent marker if present in the dataset.
   - For novel external compounds, explains that coordinates are not in the saved embedding (coordinates are never fabricated).
   - Dynamic coloring by Dominant Odor, MW, LogP, TPSA, and Scaffold partition.

6. **Predictive Uncertainty & Calibration:**
   - Calculates predictive Shannon entropy ($H(p)$) and minimum decision margin.
   - High/Low uncertainty categorization based on empirical median cutoff ($0.44\text{ bits}$).
   - Clear distinction between mathematical model entropy and human sensory ambiguity.

---

## Launch Instructions

From the project root within the WSL2 environment:

```bash
# 1. Activate the environment
source backend/.venv/bin/activate

# 2. Launch the Streamlit application
PYTHONPATH=. streamlit run playground/app.py --server.port 8501 --server.headless true
```

Access the application in your browser at:
`http://localhost:8501`

---

## System Requirements
- **Hardware:** Local CPU only (no GPU required).
- **RAM:** $< 1.5\text{ GB}$ peak memory consumption.
- **Python:** 3.11+

