"""Backend service integration layer for the OlfactAI Research Playground."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors

from backend.app.chemistry.descriptors import DescriptorExtractor
from backend.app.chemistry.parser import MoleculeParser
from backend.app.core.exceptions import InvalidSMILESError
from backend.app.ml.evaluation.calibration import TemperatureScaler
from backend.app.ml.evaluation.thresholds import apply_thresholds, load_thresholds
from backend.app.ml.evaluation.uncertainty import (
    compute_decision_margin_uncertainty,
    compute_predictive_entropy,
)
from backend.app.ml.features.graph import MolecularGraphFeaturizer
from backend.app.ml.models.dummy import DummyFrequencySmellModel
from backend.app.ml.models.gnn import MolecularGNNSmellModel
from backend.app.ml.models.pytorch_mlp import PyTorchMLPSmellModel
from backend.app.ml.models.xgboost_model import XGBoostSmellModel
from backend.app.ml.preprocessing.scaler import OlfactoryFeatureScaler
from backend.app.similarity.engine import MolecularSimilarityEngine
from backend.app.similarity.neighborhood import MolecularNeighborhoodAnalyzer

REPO_ROOT = Path(__file__).resolve().parent.parent


class PlaygroundService:
    """Encapsulates model inference, similarity queries, and chemical manifold inspection."""

    def __init__(self, repo_root: Optional[Path] = None):
        self.root = repo_root if repo_root is not None else REPO_ROOT
        self.parser = MoleculeParser()
        self._similarity_engine: Optional[MolecularSimilarityEngine] = None
        self._neighborhood_analyzer: Optional[MolecularNeighborhoodAnalyzer] = None
        self._loaded_models: Dict[str, Any] = {}

    @property
    def similarity_engine(self) -> MolecularSimilarityEngine:
        """Lazy-loaded MolecularSimilarityEngine singleton."""
        if self._similarity_engine is None:
            ds_dir = self.root / "backend" / "datasets" / "processed"
            self._similarity_engine = MolecularSimilarityEngine(dataset_dir=ds_dir, auto_load=True)
        return self._similarity_engine

    @property
    def neighborhood_analyzer(self) -> MolecularNeighborhoodAnalyzer:
        """Lazy-loaded MolecularNeighborhoodAnalyzer singleton."""
        if self._neighborhood_analyzer is None:
            self._neighborhood_analyzer = MolecularNeighborhoodAnalyzer(engine=self.similarity_engine)
        return self._neighborhood_analyzer

    def parse_molecule(self, smiles: str) -> Dict[str, Any]:
        """Validate SMILES, compute 2D depiction, and extract 9 physicochemical properties.

        Args:
            smiles: Input chemical SMILES string.

        Returns:
            Dict containing parsed mol object, canonical SMILES, descriptors, formula, and image.
        """
        mol = self.parser.parse_smiles(smiles)
        canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        descriptors = DescriptorExtractor.extract_descriptors(mol)

        # 2D depiction as PIL Image
        img = Draw.MolToImage(mol, size=(380, 280), kekulize=True)

        # Check if molecule is present in dataset
        matched_id = None
        if canonical_smi in self.similarity_engine._smiles_to_index:
            matched_id = self.similarity_engine.molecule_ids[
                self.similarity_engine._smiles_to_index[canonical_smi]
            ]

        return {
            "mol": mol,
            "canonical_smiles": canonical_smi,
            "molecular_formula": formula,
            "descriptors": descriptors,
            "image": img,
            "dataset_molecule_id": matched_id,
        }

    def _get_model(self, model_name: str, split: str) -> Any:
        """Load and cache model checkpoint on CPU."""
        key = f"{model_name}_{split}"
        if key in self._loaded_models:
            return self._loaded_models[key]

        models_dir = self.root / "backend" / "models"
        if model_name == "gnn":
            model = MolecularGNNSmellModel()
            model.load(models_dir / "phase4" / split / "gnn" / "model.pt")
        elif model_name == "xgboost":
            model = XGBoostSmellModel()
            model.load(models_dir / "phase3" / split / "xgboost" / "model.joblib")
        elif model_name == "pytorch_mlp":
            model = PyTorchMLPSmellModel()
            model.load(models_dir / "phase3" / split / "pytorch_mlp" / "model.pt")
        elif model_name == "dummy":
            model = DummyFrequencySmellModel()
            model.load(models_dir / "phase3" / split / "dummy" / "model.joblib")
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self._loaded_models[key] = model
        return model

    def predict_odor_profiles(
        self,
        smiles: str,
        split: str = "random",
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Execute multi-model inference across Phase 3 and Phase 4 models.

        Args:
            smiles: Chemical SMILES string.
            split: Protocol ('random' or 'scaffold').
            top_n: Number of highest-probability descriptors to return.

        Returns:
            Dict containing predictions, raw & calibrated probabilities, thresholds, and uncertainties.
        """
        mol_info = self.parse_molecule(smiles)
        mol = mol_info["mol"]
        desc = mol_info["descriptors"]

        # 1. Feature extraction for Phase 3 tabular models (XGBoost, MLP, Dummy)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)
        binary_fp = np.zeros((1, 2048), dtype=np.float32)
        for b in fp.GetOnBits():
            binary_fp[0, b] = 1.0

        cont_features = np.array(
            [[
                desc.molecular_weight,
                desc.logp,
                desc.tpsa,
                desc.h_bond_donors,
                desc.h_bond_acceptors,
                desc.rotatable_bonds,
                desc.ring_count,
                desc.heavy_atom_count,
                desc.fraction_csp3,
            ]],
            dtype=np.float32,
        )

        scaler_path = self.root / "backend" / "models" / "phase3" / split / "feature_scaler.joblib"
        scaler = OlfactoryFeatureScaler.load(scaler_path)
        scaled_cont = scaler.scaler.transform(cont_features)
        tabular_features = np.hstack([scaled_cont, binary_fp])

        # 2. Graph featurization for Phase 4 GNN
        graph = MolecularGraphFeaturizer.smiles_to_graph(mol_info["canonical_smiles"])

        # 3. Model Inference
        models_dir = self.root / "backend" / "models"
        results: Dict[str, Any] = {}

        # GNN Model
        gnn = self._get_model("gnn", split)
        gnn_logits = gnn.predict_logits([graph])
        gnn_raw_probs = gnn.predict_proba([graph])[0]

        cal_path = models_dir / "phase4" / split / "gnn" / "calibration.json"
        temp_scaler = TemperatureScaler.load(cal_path)
        gnn_cal_probs = temp_scaler.transform(gnn_logits)[0]

        gnn_thresh, labels, _ = load_thresholds(models_dir / "phase4" / split / "gnn" / "thresholds.json")
        gnn_binary = (gnn_cal_probs >= gnn_thresh).astype(int)
        gnn_entropy = float(compute_predictive_entropy(np.array([gnn_cal_probs]))["molecule_mean_entropy"][0])
        gnn_margins = compute_decision_margin_uncertainty(np.array([gnn_cal_probs]))[0]

        results["GNN (MPNN Phase 4)"] = {
            "raw_probabilities": gnn_raw_probs,
            "calibrated_probabilities": gnn_cal_probs,
            "thresholds": gnn_thresh,
            "binary_predictions": gnn_binary,
            "predictive_entropy": gnn_entropy,
            "decision_margin_min": float(np.min(gnn_margins)),
            "active_labels": [labels[i] for i in range(len(labels)) if gnn_binary[i] == 1],
            "top_descriptors": [
                (labels[i], float(gnn_cal_probs[i]), float(gnn_raw_probs[i]), float(gnn_thresh[i]), int(gnn_binary[i]))
                for i in np.argsort(gnn_cal_probs)[::-1][:top_n]
            ],
            "supports_calibration": True,
            "temperature": float(temp_scaler.temperature),
        }

        # XGBoost Model
        xgb = self._get_model("xgboost", split)
        xgb_probs = xgb.predict_proba(tabular_features)[0]
        xgb_thresh, _, _ = load_thresholds(models_dir / "phase3" / split / "xgboost" / "thresholds.json")
        xgb_binary = (xgb_probs >= xgb_thresh).astype(int)
        xgb_entropy = float(compute_predictive_entropy(np.array([xgb_probs]))["molecule_mean_entropy"][0])
        xgb_margins = compute_decision_margin_uncertainty(np.array([xgb_probs]))[0]

        results["XGBoost (Phase 3)"] = {
            "raw_probabilities": xgb_probs,
            "calibrated_probabilities": xgb_probs,  # Uncalibrated baseline
            "thresholds": xgb_thresh,
            "binary_predictions": xgb_binary,
            "predictive_entropy": xgb_entropy,
            "decision_margin_min": float(np.min(xgb_margins)),
            "active_labels": [labels[i] for i in range(len(labels)) if xgb_binary[i] == 1],
            "top_descriptors": [
                (labels[i], float(xgb_probs[i]), float(xgb_probs[i]), float(xgb_thresh[i]), int(xgb_binary[i]))
                for i in np.argsort(xgb_probs)[::-1][:top_n]
            ],
            "supports_calibration": False,
        }

        # PyTorch MLP Model
        mlp = self._get_model("pytorch_mlp", split)
        mlp_probs = mlp.predict_proba(tabular_features)[0]
        mlp_thresh, _, _ = load_thresholds(models_dir / "phase3" / split / "pytorch_mlp" / "thresholds.json")
        mlp_binary = (mlp_probs >= mlp_thresh).astype(int)
        mlp_entropy = float(compute_predictive_entropy(np.array([mlp_probs]))["molecule_mean_entropy"][0])
        mlp_margins = compute_decision_margin_uncertainty(np.array([mlp_probs]))[0]

        results["PyTorch MLP (Phase 3)"] = {
            "raw_probabilities": mlp_probs,
            "calibrated_probabilities": mlp_probs,
            "thresholds": mlp_thresh,
            "binary_predictions": mlp_binary,
            "predictive_entropy": mlp_entropy,
            "decision_margin_min": float(np.min(mlp_margins)),
            "active_labels": [labels[i] for i in range(len(labels)) if mlp_binary[i] == 1],
            "top_descriptors": [
                (labels[i], float(mlp_probs[i]), float(mlp_probs[i]), float(mlp_thresh[i]), int(mlp_binary[i]))
                for i in np.argsort(mlp_probs)[::-1][:top_n]
            ],
            "supports_calibration": False,
        }

        return {
            "molecule_info": mol_info,
            "split": split,
            "labels": labels,
            "model_results": results,
        }

    def search_similar(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Perform nearest-neighbor molecular similarity query over the 4,983-molecule corpus."""
        search_res = self.similarity_engine.search(query=query, top_k=top_k, exclude_self=True)
        return {
            "query": search_res.query,
            "query_canonical_smiles": search_res.query_canonical_smiles,
            "matches": [
                {
                    "rank": m.rank,
                    "molecule_id": m.molecule_id,
                    "canonical_smiles": m.canonical_smiles,
                    "tanimoto_similarity": m.tanimoto_similarity,
                    "known_odor_labels": m.known_odor_labels,
                    "molecular_weight": m.physicochemical_descriptors.get("molecular_weight", 0.0),
                    "logp": m.physicochemical_descriptors.get("logp", 0.0),
                    "tpsa": m.physicochemical_descriptors.get("tpsa", 0.0),
                }
                for m in search_res.matches
            ],
        }

    def analyze_neighborhood(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Perform structural neighborhood odor-profile analysis."""
        res = self.neighborhood_analyzer.analyze_neighborhood(query=query, k=top_k)
        return {
            "query": res.query,
            "k": res.k,
            "mean_structural_similarity": res.mean_structural_similarity,
            "median_structural_similarity": res.median_structural_similarity,
            "mean_odor_jaccard": res.mean_odor_jaccard,
            "median_odor_jaccard": res.median_odor_jaccard,
            "zero_odor_overlap_proportion": res.zero_odor_overlap_proportion,
            "top_neighbor_labels": res.top_neighbor_labels[:15],
            "neighbor_label_frequencies": res.neighbor_label_frequencies,
        }

    def load_chemical_space_data(self) -> pd.DataFrame:
        """Load saved 2D UMAP embeddings joined with dominant odor labels and descriptors."""
        emb_path = self.root / "backend" / "experiments" / "phase5" / "embeddings" / "umap_2d_embeddings.csv"
        features_path = self.root / "backend" / "datasets" / "processed" / "molecular_features.csv"
        labels_path = self.root / "backend" / "datasets" / "processed" / "odor_labels.csv"
        mols_path = self.root / "backend" / "datasets" / "processed" / "molecules.csv"

        emb_df = pd.read_csv(emb_path)
        feat_df = pd.read_csv(features_path)
        labels_df = pd.read_csv(labels_path)
        mols_df = pd.read_csv(mols_path)

        from backend.app.clustering.umap_embeddings import MolecularUMAP
        dominant = MolecularUMAP.assign_dominant_odor_labels(labels_df)

        merged = emb_df.copy()
        # Handle singleton disconnected molecules (e.g. single-atom [Zn] with zero Morgan bits)
        merged["umap_1"] = merged["umap_1"].fillna(0.0)
        merged["umap_2"] = merged["umap_2"].fillna(0.0)
        merged["dominant_odor"] = dominant.values
        merged["molecular_weight"] = feat_df["molecular_weight"].values
        merged["logp"] = feat_df["logp"].values
        merged["tpsa"] = feat_df["tpsa"].values
        merged["scaffold_split"] = mols_df["scaffold_split"].values

        return merged

