# OlfactAI — Molecular Smell Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.9+-yellow.svg)](https://www.rdkit.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20Only-orange.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(52%2F52)-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

OlfactAI is an artificial intelligence platform designed to decode, characterize, and predict the smell profiles of chemical molecules directly from molecular structure representations (SMILES). The system functions as a software-simulated electronic nose and chemical smell intelligence suite running entirely on local CPU resources.

---

## 1. Vision & Core Philosophy

Chemical olfaction has traditionally relied on expensive physical sensor arrays, human panel psychophysics, and sensory panels. OlfactAI models molecular olfactory perception purely in software:

- **100% Local CPU Execution:** Zero GPU dependencies, enabling deployment on developer laptops and standard server nodes.
- **No Hardware / IoT Required:** The electronic nose is modeled virtually via mathematical cross-reactive sensor response curves.
- **Zero Paid APIs or Cloud Lock-in:** All molecular parsing, fingerprint calculations, and ML inferences run hermetically offline.
- **Cheminformatics-First:** Built upon RDKit's validated C++ chemistry engine for SMILES canonicalization, valence validation, descriptor calculation, and circular Morgan fingerprints (ECFP).

---

## 2. System Architecture

The project is structured with a modular, layered design:

```
OlfactAI/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── endpoints/
│   │   │       │   ├── chemistry.py   # /api/v1/chemistry/analyze, /validate
│   │   │       │   └── health.py      # /api/v1/health
│   │   │       └── router.py          # Master v1 API router
│   │   ├── core/                      # Configuration, logging, exception handlers
│   │   ├── chemistry/                 # RDKit parsing, descriptors, fingerprints, service
│   │   ├── data/                      # Dataset loaders and base schemas
│   │   ├── ml/                        # ML pipelines, models, training & evaluation
│   │   │   ├── preprocessing/         # Feature scalers and transforms
│   │   │   ├── features/              # SMILES -> numerical feature pipeline
│   │   │   ├── models/                # RandomForest, XGBoost, PyTorch MLP (CPU), Registry
│   │   │   ├── training/              # Training orchestration
│   │   │   └── evaluation/            # Multi-label F1, Hamming loss, ROC-AUC
│   │   ├── prediction/                # Smell prediction service interface
│   │   ├── similarity/                # Tanimoto similarity and candidate ranking
│   │   ├── clustering/                # UMAP + HDBSCAN chemical space clustering
│   │   ├── explainability/            # SHAP model explainability interface
│   │   ├── digital_nose/              # Software-simulated e-nose sensor array
│   │   ├── mixtures/                  # Odor mixture blending simulation
│   │   ├── schemas/                   # Pydantic request and response schemas
│   │   ├── services/                  # Application service orchestrators
│   │   └── main.py                    # FastAPI application initialization
│   │
│   ├── tests/
│   │   ├── unit/                      # Fast unit tests (parsers, descriptors, ML guards)
│   │   ├── integration/               # API endpoint tests and error response tests
│   │   └── fixtures/                  # Benchmark molecules and negative test cases
│   │
│   ├── experiments/                   # Experiment artifacts
│   ├── models/                        # Serialized model weights (.joblib, .pt)
│   ├── datasets/                      # Raw, processed, and external datasets
│   │   ├── raw/
│   │   ├── processed/
│   │   └── external/
│   ├── scripts/                       # Backend management scripts (run_dev.py)
│   ├── requirements.txt               # Pinned dependencies with PyTorch CPU index
│   └── .env.example                   # Environment template
│
├── frontend/                          # Future Next.js frontend application
│   └── README.md
│
├── notebooks/                         # Research and exploration notebooks
│   ├── exploratory_analysis/
│   ├── model_experiments/
│   └── visualization/
│
├── docs/                              # Project documentation
│   ├── architecture/
│   ├── api/
│   └── research/
│
├── scripts/                           # Root operational utility scripts
├── .gitignore
├── LICENSE
└── README.md
```

---

## 3. Technology Stack

### Backend
- **Language:** Python 3.11
- **Web Framework:** FastAPI, Uvicorn, Pydantic v2, Pydantic-Settings
- **Cheminformatics:** RDKit (v2023+)
- **Numerical & Data:** NumPy, Pandas, SciPy
- **Machine Learning (CPU):** Scikit-learn, XGBoost, PyTorch (CPU wheel)
- **Dimensionality & Clustering:** UMAP-learn, HDBSCAN
- **Explainability:** SHAP
- **Testing:** Pytest, Pytest-cov, HTTPX

### Frontend (Future Phase)
- Next.js (React 19), TypeScript, Tailwind CSS, Recharts / Plotly.js

---

## 4. Windows Installation & Setup Guide

### Prerequisites
- Windows 10/11 (64-bit)
- PowerShell 5.1+ or PowerShell 7+
- Python 3.11 (or [uv](https://astral.sh/uv) package manager)

### Step 1: Clone Repository
```powershell
git clone <repository_url>
cd OlfactAI
```

### Step 2: Set Up Virtual Environment with Python 3.11

Using `uv` (recommended for ultra-fast, deterministic setup):
```powershell
# Install uv if not already installed
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create Python 3.11 virtual environment in backend/.venv
$env:Path = "$HOME\.local\bin;$env:Path"
uv python install 3.11
uv venv backend/.venv --python 3.11
```

Or using standard Python 3.11 launcher:
```powershell
py -3.11 -m venv backend/.venv
```

### Step 3: Activate the Virtual Environment
```powershell
backend\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies
```powershell
uv pip install -r backend/requirements.txt
# Or via standard pip:
# pip install -r backend/requirements.txt
```

### Step 5: Environment Variables Setup
Copy the example environment file:
```powershell
Copy-Item backend\.env.example backend\.env
```

---

## 5. Running the Application

### Option A: Using the Convenience Script
From the project root:
```powershell
python backend/scripts/run_dev.py
```

### Option B: Using Uvicorn Directly
```powershell
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Accessing the API & Documentation
- **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 6. API Reference

### Health Endpoints

#### `GET /health` and `GET /api/v1/health`
Checks service readiness and validates CPU-only execution.

**Response:**
```json
{
  "status": "healthy",
  "app_name": "OlfactAI",
  "version": "0.1.0",
  "environment": "development",
  "cpu_only": true,
  "components": {
    "python": "3.11.18",
    "platform": "win32",
    "rdkit": "2026.09.1",
    "torch": "2.15.0+cpu (cuda_available=False)"
  }
}
```

---

### Chemistry Endpoints

#### `POST /api/v1/chemistry/validate`
Validates whether a SMILES string represents a chemically feasible molecule.

**Request Body:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O"
}
```

**Response Body:**
```json
{
  "is_valid": true,
  "input_smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "message": "SMILES structure is chemically valid.",
  "error_details": null
}
```

---

#### `POST /api/v1/chemistry/analyze`
Parses SMILES, calculates canonical representation, extracts 10 physicochemical descriptors, and computes circular Morgan fingerprints.

**Request Body:**
```json
{
  "smiles": "COc1cc(C=O)ccc1O",
  "include_fingerprint": false,
  "morgan_radius": 2,
  "morgan_nbits": 2048
}
```

**Response Body:**
```json
{
  "input_smiles": "COc1cc(C=O)ccc1O",
  "canonical_smiles": "COc1cc(C=O)ccc1O",
  "is_valid": true,
  "descriptors": {
    "molecular_weight": 152.149,
    "molecular_formula": "C8H8O3",
    "logp": 1.2065,
    "tpsa": 46.53,
    "h_bond_donors": 1,
    "h_bond_acceptors": 3,
    "rotatable_bonds": 2,
    "ring_count": 1,
    "heavy_atom_count": 11,
    "fraction_csp3": 0.125
  },
  "fingerprint": {
    "radius": 2,
    "nbits": 2048,
    "on_bits_count": 31,
    "on_bit_indices": [29, 39, 140, 249, 389, 561, 650, 695, 787, 807, 875, 935, 1017, 1057, 1088, 1148, 1152, 1264, 1380, 1480, 1509, 1512, 1603, 1750, 1844, 1873, 1917, 1928, 1970, 2041, 2047],
    "bit_vector": null
  }
}
```

> **Note:** To retrieve the full 2048 dense binary vector, set `"include_fingerprint": true` in the request.

---

### Error Handling Specification

All error responses return standardized, machine-readable JSON:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_SMILES",
    "message": "Structure rejected by RDKit validator. Ensure valence and aromaticity rules are satisfied.",
    "details": {
      "smiles": "C1CC"
    }
  }
}
```

---

## 7. Running Automated Tests

Run the complete test suite locally:
```powershell
# Run all tests with verbose output
backend\.venv\Scripts\pytest backend/tests -v

# Run with test coverage
backend\.venv\Scripts\pytest backend/tests --cov=backend/app --cov-report=term-missing
```

### Test Coverage Highlights:
- **Valid SMILES parsing & canonicalization:** Verified against ethanol, aspirin, vanillin, and isoamyl acetate.
- **Negative testing:** Unclosed rings, invalid valence, empty strings, and non-chemical inputs.
- **Descriptor accuracy:** Molecular weight, formula, LogP, TPSA, HBD, HBA, rotatable bonds, ring counts.
- **Fingerprint fidelity:** 2048-bit Morgan bit vector dimensions, sparse indexing, and dense array toggling.
- **Untrained model guardrails:** RandomForest, XGBoost, and PyTorch MLP models properly prevent inference without trained weights.
- **API integration:** Status codes, payloads, and error handlers.

---

## 8. Dataset Pipeline & Preprocessing (Phase 2)

OlfactAI utilizes the unified human olfactory benchmark dataset from the official OpenPOM repository (Lee et al., *Science* 2023), comprising **4,983 molecules** and **138 binary odor descriptors**.

### Pipeline Execution Commands

All commands are executed from the project root using the Python 3.11 virtual environment:

```powershell
# Step 1: Acquire raw dataset from official OpenPOM source (with SHA-256 verification)
python backend/scripts/acquire_dataset.py

# Step 2: Inspect raw dataset and generate data quality audit report
python backend/scripts/inspect_dataset.py

# Step 3: Preprocess molecules, extract physicochemical descriptors & 2048-bit Morgan fingerprints
python backend/scripts/preprocess_dataset.py

# Step 4: Generate leakage-free random and Bemis-Murcko scaffold splits (70/15/15)
python backend/scripts/create_splits.py

# Step 5: Render 7 publication-ready Exploratory Data Analysis (EDA) visualizations
python backend/scripts/generate_eda.py
```

### Dataset Artifact Structure
```
backend/datasets/
├── raw/
│   ├── curated_GS_LF_merged_4983.csv       # Original verified benchmark dataset
│   └── raw_dataset_metadata.json           # Checksum, provenance, and license info
│
└── processed/
    ├── molecules.csv                       # Canonical SMILES, formula, split tags
    ├── molecular_features.csv              # 9 physicochemical descriptors + 2048 Morgan bits
    ├── odor_labels.csv                     # Pure multi-label binary matrix (138 targets)
    ├── dataset_metadata.json               # Schema, dimensions, and artifact checksums
    ├── data_quality_report.json            # Missing values and duplicate analysis
    ├── splits.json                         # Full molecule-level split assignments
    ├── splits_summary.json                 # Partition sizes & scaffold leakage audit
    └── eda/                                # Generated analytical figures
        ├── label_frequency.png
        ├── label_cooccurrence.png
        ├── molecular_weight_distribution.png
        ├── logp_distribution.png
        ├── tpsa_distribution.png
        ├── missing_values.png
        └── split_label_distribution.png
```

---

## 9. CPU-Only Execution Guarantees

1. **PyTorch Wheel:** Configured with CPU-only binaries (`--extra-index-url https://download.pytorch.org/whl/cpu`).
2. **Device Enforcement:** PyTorch models explicitly bind to `torch.device("cpu")`.
3. **XGBoost Configuration:** Configured with `tree_method="hist"` for high-throughput multi-core CPU training.
4. **Scikit-learn:** Multi-threaded parallel processing enabled via `n_jobs=-1`.

---

## 10. Development Roadmap

- [x] **Phase 1: Project Initialization & Chemistry Engine (Completed)**
  - Modular architecture, FastAPI foundation, RDKit processing, automated tests.
- [x] **Phase 2: Dataset Acquisition, Preprocessing & EDA (Completed)**
  - OpenPOM benchmark dataset acquisition, validation, 2048-bit Morgan FP & descriptor matrices, 7 EDA charts, leakage-free Bemis-Murcko scaffold splitting.
- [ ] **Phase 3: Model Training & Benchmark Evaluation**
  - Multi-label smell classification benchmarks comparing Random Forest, XGBoost, and Deep MLP.
- [ ] **Phase 4: Software-Simulated Electronic Nose & Mixture Blending**
  - Virtual sensor array response profiles and non-linear blend interaction modeling.
- [ ] **Phase 5: Explainability & Chemical Space Exploration**
  - Sub-structural SHAP visualizations and interactive UMAP chemical space atlas.
- [ ] **Phase 6: Next.js Frontend Development**
  - Interactive UI with 3D molecular canvas, radar charts, and sensor simulation dashboard.

---

## 11. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

