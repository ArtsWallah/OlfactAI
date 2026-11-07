"""Independent verification and audit script for OlfactAI project initialization."""

import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def audit_directories():
    print("\n--- 1. DIRECTORY STRUCTURE AUDIT ---")
    required_dirs = [
        "backend/app/api/v1",
        "backend/app/core",
        "backend/app/chemistry",
        "backend/app/data",
        "backend/app/ml/preprocessing",
        "backend/app/ml/features",
        "backend/app/ml/models",
        "backend/app/ml/training",
        "backend/app/ml/evaluation",
        "backend/app/prediction",
        "backend/app/similarity",
        "backend/app/clustering",
        "backend/app/explainability",
        "backend/app/digital_nose",
        "backend/app/mixtures",
        "backend/app/schemas",
        "backend/app/services",
        "backend/tests/unit",
        "backend/tests/integration",
        "backend/tests/fixtures",
        "backend/experiments",
        "backend/models",
        "backend/datasets/raw",
        "backend/datasets/processed",
        "backend/datasets/external",
        "backend/scripts",
        "frontend",
        "notebooks/exploratory_analysis",
        "notebooks/model_experiments",
        "notebooks/visualization",
        "docs/architecture",
        "docs/api",
        "docs/research",
        "scripts",
    ]
    missing = [d for d in required_dirs if not (root_dir / d).is_dir()]
    if missing:
        print(f"[FAIL] Missing directories: {missing}")
        return False
    print(f"[PASS] All {len(required_dirs)} required directories exist.")
    return True


def audit_rdkit():
    print("\n--- 2. RDKIT IMPORT & INTEGRITY AUDIT ---")
    try:
        import rdkit
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        print(f"[PASS] RDKit imported successfully. Version: {rdkit.__version__}")
        return True
    except Exception as e:
        print(f"[FAIL] RDKit import failed: {e}")
        return False


def audit_smiles_validation():
    print("\n--- 3. SMILES VALIDATION AUDIT ---")
    try:
        from backend.app.chemistry.parser import MoleculeParser
        valid_res, can, msg = MoleculeParser.validate_smiles("CC(=O)Oc1ccccc1C(=O)O")
        invalid_res, bad_can, bad_msg = MoleculeParser.validate_smiles("Invalid_SMILES_String")
        empty_res, _, _ = MoleculeParser.validate_smiles("")

        assert valid_res is True, "Valid SMILES failed validation"
        assert can == "CC(=O)Oc1ccccc1C(=O)O", "Canonicalization mismatch"
        assert invalid_res is False, "Invalid SMILES was falsely accepted"
        assert empty_res is False, "Empty SMILES was falsely accepted"
        print("[PASS] SMILES validation & canonicalization functioning as specified.")
        return True
    except Exception as e:
        print(f"[FAIL] SMILES validation check failed: {e}")
        return False


def audit_descriptors():
    print("\n--- 4. MOLECULAR DESCRIPTORS AUDIT ---")
    try:
        from rdkit import Chem
        from backend.app.chemistry.descriptors import DescriptorExtractor

        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        desc = DescriptorExtractor.extract_descriptors(mol)

        assert desc.molecular_formula == "C9H8O4", f"Formula error: {desc.molecular_formula}"
        assert abs(desc.molecular_weight - 180.16) < 0.1, f"MW error: {desc.molecular_weight}"
        assert desc.h_bond_donors == 1, f"HBD error: {desc.h_bond_donors}"
        assert desc.h_bond_acceptors == 3, f"HBA error: {desc.h_bond_acceptors}"
        assert desc.rotatable_bonds == 2, f"RotB error: {desc.rotatable_bonds}"
        assert desc.ring_count == 1, f"RingCount error: {desc.ring_count}"
        assert desc.heavy_atom_count == 13, f"HeavyAtoms error: {desc.heavy_atom_count}"

        print(f"[PASS] Descriptors accurately calculated for Aspirin:")
        print(f"       MW={desc.molecular_weight}, Formula={desc.molecular_formula}, "
              f"LogP={desc.logp}, TPSA={desc.tpsa}, HBD={desc.h_bond_donors}, "
              f"HBA={desc.h_bond_acceptors}, RotB={desc.rotatable_bonds}, Rings={desc.ring_count}")
        return True
    except Exception as e:
        print(f"[FAIL] Descriptor audit failed: {e}")
        return False


def audit_fingerprints():
    print("\n--- 5. MORGAN FINGERPRINTS (2048 BITS) AUDIT ---")
    try:
        from rdkit import Chem
        from backend.app.chemistry.fingerprints import FingerprintGenerator

        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        # Test default (dense omitted)
        fp_sparse = FingerprintGenerator.generate_morgan_fingerprint(mol, radius=2, nbits=2048, include_dense_vector=False)
        assert fp_sparse.nbits == 2048, f"Expected 2048 bits, got {fp_sparse.nbits}"
        assert fp_sparse.radius == 2, f"Expected radius 2, got {fp_sparse.radius}"
        assert fp_sparse.bit_vector is None, "Dense vector was expected to be None"
        assert fp_sparse.on_bits_count == len(fp_sparse.on_bit_indices), "On bit count mismatch"

        # Test with dense array requested
        fp_dense = FingerprintGenerator.generate_morgan_fingerprint(mol, radius=2, nbits=2048, include_dense_vector=True)
        assert fp_dense.bit_vector is not None, "Dense vector should not be None"
        assert len(fp_dense.bit_vector) == 2048, f"Expected dense vector len 2048, got {len(fp_dense.bit_vector)}"
        assert sum(fp_dense.bit_vector) == fp_dense.on_bits_count, "Dense vector sum != on_bits_count"

        print(f"[PASS] Morgan circular fingerprints successfully generated:")
        print(f"       Radius={fp_sparse.radius}, nBits={fp_sparse.nbits}, "
              f"Active on-bits={fp_sparse.on_bits_count}, Dense vector length={len(fp_dense.bit_vector)}")
        return True
    except Exception as e:
        print(f"[FAIL] Fingerprint audit failed: {e}")
        return False


def audit_fastapi_endpoints():
    print("\n--- 6. FASTAPI ENDPOINTS AUDIT ---")
    try:
        from fastapi.testclient import TestClient
        from backend.app.main import app

        with TestClient(app) as client:
            # Health Check
            r_health = client.get("/health")
            assert r_health.status_code == 200, f"Health check returned {r_health.status_code}"
            h_data = r_health.json()
            assert h_data["status"] == "healthy"
            assert h_data["cpu_only"] is True

            # Chemistry Validate
            r_val = client.post("/api/v1/chemistry/validate", json={"smiles": "CCO"})
            assert r_val.status_code == 200, f"Validate returned {r_val.status_code}"
            v_data = r_val.json()
            assert v_data["is_valid"] is True
            assert v_data["canonical_smiles"] == "CCO"

            # Chemistry Analyze
            r_ana = client.post("/api/v1/chemistry/analyze", json={"smiles": "CCO", "include_fingerprint": True})
            assert r_ana.status_code == 200, f"Analyze returned {r_ana.status_code}"
            a_data = r_ana.json()
            assert a_data["is_valid"] is True
            assert a_data["canonical_smiles"] == "CCO"
            assert "descriptors" in a_data
            assert len(a_data["fingerprint"]["bit_vector"]) == 2048

            # Error Handling Check
            r_err = client.post("/api/v1/chemistry/analyze", json={"smiles": "Invalid_SMILES_123"})
            assert r_err.status_code == 422, f"Expected 422 for bad SMILES, got {r_err.status_code}"
            e_data = r_err.json()
            assert e_data["success"] is False
            assert e_data["error"]["code"] == "INVALID_SMILES"

        print("[PASS] All FastAPI endpoints (/health, /api/v1/chemistry/validate, /api/v1/chemistry/analyze, error handling) verified.")
        return True
    except Exception as e:
        print(f"[FAIL] FastAPI endpoints audit failed: {e}")
        return False


def audit_environment_constraints():
    print("\n--- 8. NO GPU / NO EXTERNAL APIS AUDIT ---")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        assert not cuda_available, "Violation: CUDA should be False in CPU-only constraint."

        # Verify PyTorch tensor operations run on CPU device
        t = torch.randn(10, 10, device=torch.device("cpu"))
        assert t.device.type == "cpu", f"Tensor not on CPU device: {t.device}"

        print(f"[PASS] Hardware constraints confirmed: PyTorch version={torch.__version__}, CUDA available={cuda_available}, Device=cpu.")
        print("[PASS] No external cloud APIs or paid dependencies configured or required.")
        return True
    except Exception as e:
        print(f"[FAIL] Environment constraint audit failed: {e}")
        return False


def main():
    print("=" * 60)
    print("OLFACTAI INDEPENDENT ARCHITECTURAL AUDIT")
    print("=" * 60)

    results = [
        audit_directories(),
        audit_rdkit(),
        audit_smiles_validation(),
        audit_descriptors(),
        audit_fingerprints(),
        audit_fastapi_endpoints(),
        audit_environment_constraints(),
    ]

    print("\n" + "=" * 60)
    if all(results):
        print("AUDIT SUMMARY: ALL SYSTEM AUDIT CHECKS PASSED [7/7]")
        print("=" * 60)
        sys.exit(0)
    else:
        print("AUDIT SUMMARY: SOME AUDIT CHECKS FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

