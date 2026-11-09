"""High-performance molecular similarity search engine using RDKit Morgan fingerprints and Tanimoto similarity."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from rdkit import Chem, DataStructs, RDLogger  # type: ignore
from rdkit.Chem import Descriptors, rdMolDescriptors  # type: ignore

from backend.app.chemistry.parser import MoleculeParser
from backend.app.core.exceptions import InvalidSMILESError
from backend.app.similarity.models import (
    SimilarityBenchmarkResult,
    SimilarityMatch,
    SimilaritySearchResult,
)

# Suppress noisy deprecation logs from RDKit C++ backend
try:
    RDLogger.DisableLog("rdApp.warning")
except Exception:
    pass


class MolecularSimilarityEngine:
    """Fast nearest-neighbor molecular similarity search over curated chemical spaces."""

    @staticmethod
    def get_default_dataset_dir() -> Path:
        """Resolve the default processed dataset directory relative to the package or working directory."""
        pkg_root = Path(__file__).resolve().parent.parent.parent
        candidate = pkg_root / "datasets" / "processed"
        if candidate.exists():
            return candidate
        repo_candidate = Path("backend/datasets/processed")
        if repo_candidate.exists():
            return repo_candidate
        return candidate

    def __init__(
        self,
        dataset_dir: Optional[Union[str, Path]] = None,
        radius: int = 2,
        nbits: int = 2048,
        use_chirality: bool = True,
        auto_load: bool = True,
    ):
        """Initialize the molecular similarity engine.

        Args:
            dataset_dir: Directory containing preprocessed CSV files.
            radius: Radius for Morgan circular fingerprint (default 2, ECFP4-equivalent).
            nbits: Folded bit vector length (default 2048).
            use_chirality: Whether stereochemical centers are considered (default True).
            auto_load: Whether to load and index the dataset immediately upon construction.
        """
        self.dataset_dir = Path(dataset_dir) if dataset_dir else self.get_default_dataset_dir()
        self.radius = radius
        self.nbits = nbits
        self.use_chirality = use_chirality
        self.parser = MoleculeParser()

        # Cached dataset structures
        self.molecule_ids: List[str] = []
        self.canonical_smiles: List[str] = []
        self.inchikeys: List[str] = []
        self.fingerprints: List[DataStructs.ExplicitBitVect] = []
        self.active_labels: List[List[str]] = []
        self.descriptors_table: List[Dict[str, float]] = []

        # Fast lookup indexes
        self._id_to_index: Dict[str, int] = {}
        self._smiles_to_index: Dict[str, int] = {}
        self._inchikey_to_index: Dict[str, int] = {}
        self._is_loaded: bool = False

        if auto_load:
            self.load_dataset()

    @property
    def is_loaded(self) -> bool:
        """Check whether the corpus has been loaded and indexed."""
        return self._is_loaded

    @property
    def corpus_size(self) -> int:
        """Total number of indexed molecules in the corpus."""
        return len(self.molecule_ids)

    def compute_fingerprint(self, mol_or_smiles: Union[Chem.Mol, str]) -> DataStructs.ExplicitBitVect:
        """Compute Morgan circular fingerprint with exact Phase 2 configuration.

        Args:
            mol_or_smiles: RDKit Mol instance or SMILES string.

        Returns:
            ExplicitBitVect instance of length nbits.

        Raises:
            InvalidSMILESError: If SMILES string cannot be parsed.
            TypeError: If input is neither Chem.Mol nor str.
        """
        if isinstance(mol_or_smiles, str):
            mol = self.parser.parse_smiles(mol_or_smiles)
        elif isinstance(mol_or_smiles, Chem.Mol):
            mol = mol_or_smiles
        else:
            raise TypeError(f"Expected str or Chem.Mol, got {type(mol_or_smiles).__name__}")

        return rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.radius,
            nBits=self.nbits,
            useChirality=self.use_chirality,
        )

    def load_dataset(self) -> None:
        """Load molecules, odor annotations, and physicochemical descriptors from processed dataset."""
        molecules_path = self.dataset_dir / "molecules.csv"
        labels_path = self.dataset_dir / "odor_labels.csv"
        features_path = self.dataset_dir / "molecular_features.csv"

        if not molecules_path.exists():
            raise FileNotFoundError(f"Missing molecules file at {molecules_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing odor labels file at {labels_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"Missing molecular features file at {features_path}")

        mol_df = pd.read_csv(molecules_path)
        labels_df = pd.read_csv(labels_path)
        features_df = pd.read_csv(features_path)

        descriptor_cols = [
            "molecular_weight",
            "logp",
            "tpsa",
            "h_bond_donors",
            "h_bond_acceptors",
            "rotatable_bonds",
            "ring_count",
            "heavy_atom_count",
            "fraction_csp3",
        ]
        odor_cols = [c for c in labels_df.columns if c not in ["molecule_id", "canonical_smiles"]]

        # Preallocate cached structures
        n_mols = len(mol_df)
        self.molecule_ids = mol_df["molecule_id"].tolist()
        self.canonical_smiles = mol_df["canonical_smiles"].tolist()
        self.inchikeys = []
        self.fingerprints = []
        self.active_labels = []
        self.descriptors_table = []

        self._id_to_index.clear()
        self._smiles_to_index.clear()
        self._inchikey_to_index.clear()

        # Cache odor labels per molecule
        label_matrix = labels_df[odor_cols].values
        for i in range(n_mols):
            active_idx = np.where(label_matrix[i] == 1)[0]
            self.active_labels.append([odor_cols[idx] for idx in active_idx])

        # Cache descriptors per molecule
        desc_records = features_df[descriptor_cols].to_dict(orient="records")
        self.descriptors_table = desc_records

        # Precompute RDKit Mol objects, InChIKeys, and Morgan ExplicitBitVects
        for i, (m_id, smi) in enumerate(zip(self.molecule_ids, self.canonical_smiles)):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                mol = self.parser.parse_smiles(smi)

            inchikey = Chem.MolToInchiKey(mol) if mol else ""
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.radius,
                nBits=self.nbits,
                useChirality=self.use_chirality,
            )

            self.inchikeys.append(inchikey)
            self.fingerprints.append(fp)

            self._id_to_index[m_id] = i
            self._smiles_to_index[smi] = i
            if inchikey:
                self._inchikey_to_index[inchikey] = i

        self._is_loaded = True

    def _resolve_query(
        self, query: str
    ) -> Tuple[str, Optional[str], str, str, DataStructs.ExplicitBitVect, Dict[str, float]]:
        """Resolve a query string (molecule ID or SMILES) into standardized representations.

        Returns:
            Tuple of:
            - query_type ("dataset_id" or "smiles")
            - query_molecule_id (Optional[str])
            - query_canonical_smiles (str)
            - query_inchikey (str)
            - query_fp (ExplicitBitVect)
            - query_descriptors (Dict[str, float])
        """
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace.")

        trimmed = query.strip()

        # Case 1: Existing dataset molecule ID
        if trimmed in self._id_to_index:
            idx = self._id_to_index[trimmed]
            return (
                "dataset_id",
                trimmed,
                self.canonical_smiles[idx],
                self.inchikeys[idx],
                self.fingerprints[idx],
                self.descriptors_table[idx],
            )

        # If query follows molecule ID convention (e.g. starts with MOL_), give clear KeyError
        if trimmed.startswith("MOL_") or trimmed.startswith("mol_"):
            raise KeyError(f"Molecule ID '{trimmed}' not found in the indexed dataset corpus.")

        # Case 2: Query is a chemical SMILES string
        mol = self.parser.parse_smiles(trimmed)
        canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        inchikey = Chem.MolToInchiKey(mol)
        fp = self.compute_fingerprint(mol)

        # Check if this external SMILES or InChIKey matches a known molecule in our dataset
        matched_id: Optional[str] = None
        if canonical_smi in self._smiles_to_index:
            matched_id = self.molecule_ids[self._smiles_to_index[canonical_smi]]
        elif inchikey in self._inchikey_to_index:
            matched_id = self.molecule_ids[self._inchikey_to_index[inchikey]]

        # Compute physicochemical descriptors for external query
        descriptors = {
            "molecular_weight": round(float(Descriptors.MolWt(mol)), 4),
            "logp": round(float(Descriptors.MolLogP(mol)), 4),
            "tpsa": round(float(Descriptors.TPSA(mol)), 4),
            "h_bond_donors": int(Descriptors.NumHDonors(mol)),
            "h_bond_acceptors": int(Descriptors.NumHAcceptors(mol)),
            "rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
            "ring_count": int(Descriptors.RingCount(mol)),
            "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
            "fraction_csp3": round(float(Descriptors.FractionCSP3(mol)), 4),
        }

        return (
            "smiles",
            matched_id,
            canonical_smi,
            inchikey,
            fp,
            descriptors,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        exclude_self: bool = True,
    ) -> SimilaritySearchResult:
        """Perform nearest-neighbor molecular similarity search over the indexed corpus.

        Args:
            query: Dataset molecule ID (e.g. 'MOL_00001') or valid chemical SMILES string.
            top_k: Maximum number of nearest matches to return (must be > 0).
            min_similarity: Minimum Tanimoto similarity threshold in [0.0, 1.0].
            exclude_self: Whether to exclude the query molecule based on molecular identity.

        Returns:
            SimilaritySearchResult containing ranked nearest matches.

        Raises:
            ValueError: For invalid query, top_k <= 0, or min_similarity out of bounds.
            KeyError: If a molecule ID query is not found in the indexed corpus.
            InvalidSMILESError: If query is an invalid SMILES string.
        """
        if not self._is_loaded:
            raise RuntimeError("MolecularSimilarityEngine must be loaded before searching.")

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not (0.0 <= min_similarity <= 1.0):
            raise ValueError(f"min_similarity must be within [0.0, 1.0], got {min_similarity}")

        (
            query_type,
            query_mol_id,
            query_canonical_smiles,
            query_inchikey,
            query_fp,
            query_desc,
        ) = self._resolve_query(query)

        # Ultra-fast bulk Tanimoto similarity computation against all cached bit vectors
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, self.fingerprints)

        # Filter and rank matches
        candidates: List[Tuple[float, str, int]] = []
        for idx, sim in enumerate(similarities):
            # Check minimum similarity threshold
            if sim < min_similarity:
                continue

            # Identity-based self-match exclusion
            if exclude_self:
                cand_id = self.molecule_ids[idx]
                cand_smiles = self.canonical_smiles[idx]
                cand_inchikey = self.inchikeys[idx]

                if query_mol_id and cand_id == query_mol_id:
                    continue
                if cand_smiles == query_canonical_smiles:
                    continue
                if query_inchikey and cand_inchikey == query_inchikey:
                    continue

            # Append tuple (similarity, molecule_id, index) for deterministic sorting
            candidates.append((float(sim), self.molecule_ids[idx], idx))

        # Deterministic sorting: highest similarity first; tie-breaker by molecule_id ascending
        candidates.sort(key=lambda item: (-item[0], item[1]))

        # Package top-K results
        matches: List[SimilarityMatch] = []
        for rank, (sim, cand_id, idx) in enumerate(candidates[:top_k], start=1):
            matches.append(
                SimilarityMatch(
                    molecule_id=cand_id,
                    canonical_smiles=self.canonical_smiles[idx],
                    tanimoto_similarity=round(sim, 4),
                    rank=rank,
                    known_odor_labels=list(self.active_labels[idx]),
                    physicochemical_descriptors=dict(self.descriptors_table[idx]),
                )
            )

        return SimilaritySearchResult(
            query=query,
            query_type=query_type,
            query_molecule_id=query_mol_id,
            query_canonical_smiles=query_canonical_smiles,
            matches=matches,
            total_candidates_searched=len(self.fingerprints),
        )

    def benchmark_search_latency(
        self, query: str, top_k: int = 10, min_similarity: float = 0.0, n_repeats: int = 25
    ) -> SimilarityBenchmarkResult:
        """Empirically benchmark separate stages of similarity search.

        Stages:
        1. Fingerprint calculation / retrieval time
        2. BulkTanimoto search time
        3. Candidate filtering, deterministic ranking, and packaging time
        4. End-to-end query latency

        Args:
            query: Dataset molecule ID or SMILES.
            top_k: Number of nearest neighbors to retrieve.
            min_similarity: Threshold filter.
            n_repeats: Number of timing repetitions.

        Returns:
            SimilarityBenchmarkResult with timing statistics in milliseconds.
        """
        if not self._is_loaded:
            raise RuntimeError("MolecularSimilarityEngine must be loaded before benchmarking.")

        (
            query_type,
            query_mol_id,
            query_canonical_smiles,
            query_inchikey,
            query_fp,
            query_desc,
        ) = self._resolve_query(query)

        # 1. Fingerprint calculation / retrieval benchmark
        fp_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            if query_type == "dataset_id" and query in self._id_to_index:
                _ = self.fingerprints[self._id_to_index[query]]
            else:
                _ = self.compute_fingerprint(query_canonical_smiles)
            fp_times.append((time.perf_counter() - t0) * 1000.0)

        # 2. Bulk Tanimoto similarity benchmark
        bulk_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = DataStructs.BulkTanimotoSimilarity(query_fp, self.fingerprints)
            bulk_times.append((time.perf_counter() - t0) * 1000.0)

        # 3. Filtering, ranking, and packaging benchmark
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, self.fingerprints)
        ranking_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            candidates = []
            for idx, s in enumerate(sims):
                if s < min_similarity:
                    continue
                if query_mol_id and self.molecule_ids[idx] == query_mol_id:
                    continue
                if self.canonical_smiles[idx] == query_canonical_smiles:
                    continue
                candidates.append((s, self.molecule_ids[idx], idx))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            _matches = [
                SimilarityMatch(
                    molecule_id=cand_id,
                    canonical_smiles=self.canonical_smiles[idx],
                    tanimoto_similarity=round(s, 4),
                    rank=r,
                    known_odor_labels=self.active_labels[idx],
                    physicochemical_descriptors=self.descriptors_table[idx],
                )
                for r, (s, cand_id, idx) in enumerate(candidates[:top_k], start=1)
            ]
            ranking_times.append((time.perf_counter() - t0) * 1000.0)

        # 4. Total end-to-end benchmark
        e2e_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = self.search(query=query, top_k=top_k, min_similarity=min_similarity, exclude_self=True)
            e2e_times.append((time.perf_counter() - t0) * 1000.0)

        return SimilarityBenchmarkResult(
            fp_generation_ms=round(float(np.mean(fp_times)), 4),
            bulk_tanimoto_ms=round(float(np.mean(bulk_times)), 4),
            ranking_packaging_ms=round(float(np.mean(ranking_times)), 4),
            end_to_end_ms=round(float(np.mean(e2e_times)), 4),
            n_repeats=n_repeats,
            corpus_size=len(self.fingerprints),
        )
