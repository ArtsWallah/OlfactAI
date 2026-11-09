"""OlfactAI — Interactive Research Playground (Streamlit Testing Interface).

Run via:
    streamlit run playground/app.py
"""

from pathlib import Path
import sys

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from playground.services import PlaygroundService

# Page configuration
st.set_page_config(
    page_title="OlfactAI — Research Playground",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_service() -> PlaygroundService:
    """Initialize and cache the backend service singleton."""
    return PlaygroundService(repo_root=REPO_ROOT)


@st.cache_data
def get_chemical_space() -> pd.DataFrame:
    """Load and cache the precomputed 2D UMAP coordinates joined with properties."""
    svc = get_service()
    return svc.load_chemical_space_data()


def main():
    service = get_service()

    st.title("🧪 OlfactAI — Research Playground")
    st.caption(
        "Interactive testing interface for molecular representation learning, multi-label odor prediction, "
        "similarity search, and chemical-space discovery (Phases 1–5). **CPU execution only.**"
    )

    # -------------------------------------------------------------
    # Sidebar: Input and Parameters
    # -------------------------------------------------------------
    st.sidebar.header("🔬 Molecular Query")

    sample_smiles = {
        "Vanillin (Vanilla, sweet)": "O=Cc1ccc(O)c(OC)c1",
        "Eugenol (Clove, spicy)": "C=CCc1ccc(O)c(OC)c1",
        "Isoamyl acetate (Banana, fruity)": "CC(=O)OCCC(C)C",
        "Benzyl alcohol (Floral, sweet)": "OCc1ccccc1",
        "Geosmin (Earthy, musty)": "CC1(C)CCCC2(C)CCCC12O",
        "Cubane (Synthetic novel, unlabeled)": "C12C3C4C1C5C2C3C45",
    }

    selected_sample = st.sidebar.selectbox("Select Sample Molecule:", ["-- Custom SMILES --"] + list(sample_smiles.keys()))

    default_smiles = "O=Cc1ccc(O)c(OC)c1"
    if selected_sample != "-- Custom SMILES --":
        default_smiles = sample_smiles[selected_sample]

    query_smiles = st.sidebar.text_input("Enter SMILES string:", value=default_smiles)

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Evaluation Settings")
    selected_split = st.sidebar.radio("Evaluation Split Protocol:", ["random", "scaffold"], index=0)
    top_k_similarity = st.sidebar.selectbox("Similarity Top-K Neighbors:", [5, 10, 20], index=1)
    top_n_descriptors = st.sidebar.slider("Top Predicted Odors to Display:", min_value=5, max_value=25, value=10)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Scientific Caveats:**\n\n"
        "• Model predictions represent learned associations between 2D molecular structures and historical dataset annotations.\n\n"
        "• Similarity in 2D Morgan space does not guarantee perceptual olfactory identity.\n\n"
        "• Never use benchmark classification metrics as proof of human sensory perception."
    )

    # Validate and parse input molecule
    try:
        mol_data = service.parse_molecule(query_smiles)
    except Exception as e:
        st.error(f"Invalid SMILES representation: {str(e)}")
        st.stop()

    canonical_smi = mol_data["canonical_smiles"]
    formula = mol_data["molecular_formula"]
    desc = mol_data["descriptors"]
    matched_id = mol_data["dataset_molecule_id"]

    # -------------------------------------------------------------
    # Header summary
    # -------------------------------------------------------------
    col_hdr1, col_hdr2 = st.columns([2, 1])
    with col_hdr1:
        st.markdown(f"**Canonical SMILES:** `{canonical_smi}`")
        if matched_id:
            st.success(f"Verified Dataset Molecule: **{matched_id}** (Present in 4,983-molecule curated corpus)")
        else:
            st.info("Novel / External Compound (Not in preprocessed training corpus)")

    # -------------------------------------------------------------
    # Tabs
    # -------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧬 Chemistry & Properties",
        "🎯 Odor Prediction",
        "🔍 Molecular Similarity",
        "🏘️ Neighborhood Analysis",
        "🗺️ Chemical Space (UMAP)",
        "📊 Uncertainty & Calibration",
    ])

    # -------------------------------------------------------------
    # TAB 1: Molecular Structure & Properties
    # -------------------------------------------------------------
    with tab1:
        col_img, col_props = st.columns([1, 2])
        with col_img:
            st.image(mol_data["image"], caption=f"2D Structure: {formula}", use_container_width=True)

        with col_props:
            st.subheader("Physicochemical Descriptors (RDKit)")
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.metric("Molecular Weight", f"{desc.molecular_weight:.2f} g/mol")
                st.metric("LogP (Wildman-Crippen)", f"{desc.logp:.2f}")
                st.metric("Fraction CSP3", f"{desc.fraction_csp3:.2f}")
            with pcol2:
                st.metric("TPSA", f"{desc.tpsa:.2f} Å²")
                st.metric("H-Bond Donors", f"{desc.h_bond_donors}")
                st.metric("H-Bond Acceptors", f"{desc.h_bond_acceptors}")
            with pcol3:
                st.metric("Rotatable Bonds", f"{desc.rotatable_bonds}")
                st.metric("Ring Count", f"{desc.ring_count}")
                st.metric("Heavy Atom Count", f"{desc.heavy_atom_count}")

    # -------------------------------------------------------------
    # TAB 2: Odor Prediction
    # -------------------------------------------------------------
    with tab2:
        st.subheader(f"Multi-Model Odor Descriptors ({selected_split.capitalize()} Split Protocol)")
        st.caption(
            "Comparing Phase 4 GNN against Phase 3 XGBoost and PyTorch MLP models. "
            "Descriptors are classified as **Active** when predicted probability exceeds the validation-optimized threshold."
        )

        with st.spinner("Running CPU model inference across GNN, XGBoost, and MLP..."):
            pred_results = service.predict_odor_profiles(
                smiles=canonical_smi,
                split=selected_split,
                top_n=top_n_descriptors,
            )

        model_dict = pred_results["model_results"]

        # Display Top Predictions side-by-side
        m_cols = st.columns(len(model_dict))
        for idx, (m_name, m_res) in enumerate(model_dict.items()):
            with m_cols[idx]:
                st.markdown(f"#### {m_name}")
                if m_res.get("supports_calibration", False):
                    st.caption(f"Temperature Scaler: T = {m_res['temperature']:.4f}")
                else:
                    st.caption("Uncalibrated probabilities")

                top_rows = []
                for label, cal_p, raw_p, th, is_act in m_res["top_descriptors"]:
                    top_rows.append({
                        "Descriptor": label,
                        "Cal. Prob": f"{cal_p:.3f}",
                        "Raw Prob": f"{raw_p:.3f}",
                        "Threshold": f"{th:.2f}",
                        "Active": "✅ Yes" if is_act == 1 else "❌ No",
                    })
                st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)

        # Consensus Active Descriptors
        st.markdown("---")
        st.subheader("Model Consensus on Active Descriptors")
        gnn_active = set(model_dict["GNN (MPNN Phase 4)"]["active_labels"])
        xgb_active = set(model_dict["XGBoost (Phase 3)"]["active_labels"])
        mlp_active = set(model_dict["PyTorch MLP (Phase 3)"]["active_labels"])

        all_active = sorted(list(gnn_active.union(xgb_active).union(mlp_active)))
        if all_active:
            consensus_data = []
            for lbl in all_active:
                votes = sum([lbl in gnn_active, lbl in xgb_active, lbl in mlp_active])
                consensus_data.append({
                    "Descriptor": lbl,
                    "Votes": f"{votes} / 3",
                    "GNN (Phase 4)": "✅" if lbl in gnn_active else "—",
                    "XGBoost (Phase 3)": "✅" if lbl in xgb_active else "—",
                    "PyTorch MLP": "✅" if lbl in mlp_active else "—",
                })
            st.dataframe(pd.DataFrame(consensus_data), use_container_width=True, hide_index=True)
        else:
            st.warning("No descriptors met decision threshold criteria across any model.")

    # -------------------------------------------------------------
    # TAB 3: Molecular Similarity
    # -------------------------------------------------------------
    with tab3:
        st.subheader(f"Top-{top_k_similarity} Nearest Neighbors in Curated Corpus (4,983 Molecules)")
        st.caption("Tanimoto similarity computed via RDKit 2,048-bit Morgan circular fingerprints (radius=2, chirality=True). Exact query self-match is excluded.")

        sim_res = service.search_similar(query=canonical_smi, top_k=top_k_similarity)
        matches = sim_res["matches"]

        if matches:
            sim_table = []
            for m in matches:
                sim_table.append({
                    "Rank": m["rank"],
                    "Molecule ID": m["molecule_id"],
                    "Tanimoto Similarity": f"{m['tanimoto_similarity']:.4f}",
                    "Known Odor Annotations": ", ".join(m["known_odor_labels"]) if m["known_odor_labels"] else "[unlabeled]",
                    "Canonical SMILES": m["canonical_smiles"],
                    "MW (g/mol)": f"{m['molecular_weight']:.1f}",
                    "LogP": f"{m['logp']:.2f}",
                })
            st.dataframe(pd.DataFrame(sim_table), use_container_width=True, hide_index=True)
        else:
            st.info("No similar molecules found matching criteria.")

    # -------------------------------------------------------------
    # TAB 4: Neighborhood Analysis
    # -------------------------------------------------------------
    with tab4:
        st.subheader("Structural Neighborhood Agreement")
        st.caption("Empirical relationship between query structure and neighboring odor annotations.")

        neigh_res = service.analyze_neighborhood(query=canonical_smi, top_k=top_k_similarity)

        ncol1, ncol2, ncol3, ncol4 = st.columns(4)
        with ncol1:
            st.metric("Mean Structural Tanimoto", f"{neigh_res['mean_structural_similarity']:.4f}")
        with ncol2:
            st.metric("Median Structural Tanimoto", f"{neigh_res['median_structural_similarity']:.4f}")
        with ncol3:
            st.metric("Mean Odor Jaccard", f"{neigh_res['mean_odor_jaccard']:.4f}")
        with ncol4:
            st.metric("Zero-Odor Overlap Pairs", f"{neigh_res['zero_odor_overlap_proportion']*100:.1f}%")

        st.markdown("#### Most Frequent Odor Descriptors in Neighborhood")
        top_labels = neigh_res["top_neighbor_labels"]
        if top_labels:
            chart_df = pd.DataFrame(
                [{"Descriptor": lbl, "Prevalence": round(freq * 100, 1), "Count": cnt} for lbl, cnt, freq in top_labels]
            )
            st.bar_chart(chart_df.set_index("Descriptor")["Prevalence"])
        else:
            st.info("No neighbor labels available.")

        st.info(
            "**Scientific Note:** High neighbor frequency of an annotation reflects statistical co-occurrence in historical "
            "perfumery records, not a guarantee that the query molecule possesses that smell."
        )

    # -------------------------------------------------------------
    # TAB 5: Chemical Space (UMAP)
    # -------------------------------------------------------------
    with tab5:
        st.subheader("2D Chemical Space Manifold (UMAP Exploration)")
        st.caption("Visualizing 4,983 molecules in 2,048-bit Morgan fingerprint space using Jaccard distance (random_state=42).")

        chem_space = get_chemical_space()

        color_by = st.selectbox(
            "Color Scatter Plot By:",
            ["Dominant Odor Descriptor", "Molecular Weight", "LogP", "TPSA", "Scaffold Split Partition"],
            index=0,
        )

        fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

        if color_by == "Dominant Odor Descriptor":
            top_descs = chem_space["dominant_odor"].value_counts().index[:8].tolist()
            chem_space["plot_color"] = chem_space["dominant_odor"].apply(lambda x: x if x in top_descs else "other")
            for desc_name in top_descs + ["other"]:
                sub = chem_space[chem_space["plot_color"] == desc_name]
                ax.scatter(sub["umap_1"], sub["umap_2"], s=10, alpha=0.5, label=desc_name, edgecolors="none")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Dominant Odor", fontsize=8)
        elif color_by == "Molecular Weight":
            sc = ax.scatter(chem_space["umap_1"], chem_space["umap_2"], c=chem_space["molecular_weight"], cmap="viridis", s=10, alpha=0.7)
            plt.colorbar(sc, ax=ax, label="MW (g/mol)")
        elif color_by == "LogP":
            sc = ax.scatter(chem_space["umap_1"], chem_space["umap_2"], c=chem_space["logp"], cmap="coolwarm", s=10, alpha=0.7)
            plt.colorbar(sc, ax=ax, label="LogP")
        elif color_by == "TPSA":
            sc = ax.scatter(chem_space["umap_1"], chem_space["umap_2"], c=chem_space["tpsa"], cmap="plasma", s=10, alpha=0.7)
            plt.colorbar(sc, ax=ax, label="TPSA (Å²)")
        elif color_by == "Scaffold Split Partition":
            for split_name in ["train", "val", "test"]:
                sub = chem_space[chem_space["scaffold_split"] == split_name]
                ax.scatter(sub["umap_1"], sub["umap_2"], s=10, alpha=0.5, label=f"Scaffold {split_name}", edgecolors="none")
            ax.legend(title="Split", fontsize=8)

        # Highlight query molecule if in dataset
        if matched_id:
            query_row = chem_space[chem_space["molecule_id"] == matched_id]
            if not query_row.empty:
                q_x = query_row["umap_1"].values[0]
                q_y = query_row["umap_2"].values[0]
                ax.scatter(
                    [q_x],
                    [q_y],
                    color="red",
                    s=180,
                    marker="*",
                    label=f"Query: {matched_id}",
                    edgecolors="black",
                    linewidths=1.2,
                    zorder=10,
                )
                ax.annotate(
                    f" {matched_id}",
                    (q_x, q_y),
                    fontsize=9,
                    fontweight="bold",
                    color="darkred",
                    zorder=11,
                )
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        else:
            st.info(
                "💡 **External Molecule:** Coordinates are not present in the saved 4,983-molecule dataset embedding. "
                "To maintain strict reproducibility, coordinates are not fabricated or re-fit for external inputs."
            )

        ax.set_title("2D Morgan Fingerprint UMAP Manifold", fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------------------------------------
    # TAB 6: Uncertainty & Calibration
    # -------------------------------------------------------------
    with tab6:
        st.subheader("Predictive Uncertainty & Calibration Diagnostics")
        st.caption(
            "Evaluation of model output confidence and probability dispersion across 138 odor targets."
        )

        gnn_res = model_dict["GNN (MPNN Phase 4)"]
        gnn_entropy = gnn_res["predictive_entropy"]
        margin_min = gnn_res["decision_margin_min"]

        ucol1, ucol2, ucol3 = st.columns(3)
        with ucol1:
            st.metric("GNN Predictive Entropy", f"{gnn_entropy:.4f} bits")
            st.caption("Corpus median: ~0.44 bits")
        with ucol2:
            uncertainty_status = "⚠️ High Uncertainty" if gnn_entropy > 0.44 else "✅ Low Uncertainty"
            st.metric("Uncertainty Tier", uncertainty_status)
            st.caption("Entropy correlates with Hamming error (r = 0.65 - 0.74)")
        with ucol3:
            st.metric("Minimum Decision Margin", f"{margin_min:.4f}")
            st.caption("Min distance between cal. prob and decision threshold")

        st.markdown("---")
        st.subheader("Raw Logit vs. Calibrated Probability Distribution")
        st.caption(
            f"Phase 4 temperature scaling optimizes single scalar T = {gnn_res['temperature']:.4f} "
            "on validation logits via NLL minimization to improve probability calibration."
        )

        cal_df = pd.DataFrame({
            "Calibrated Prob": gnn_res["calibrated_probabilities"],
            "Raw Sigmoid Prob": gnn_res["raw_probabilities"],
            "Threshold": gnn_res["thresholds"],
        })
        st.line_chart(cal_df.sort_values(by="Calibrated Prob", ascending=False).reset_index(drop=True))

        st.warning(
            "**Crucial Distinction:** Mathematical predictive entropy measures parameter and logit dispersion within the trained model. "
            "It does NOT measure biological perceptual ambiguity in human smell perception."
        )


if __name__ == "__main__":
    main()

