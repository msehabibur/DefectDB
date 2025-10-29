#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py: Streamlit interface for the DefectDB Studio application.
Purpose: Interactive database viewer and visualizer for semiconductor defect datasets.
"""

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
os.environ["STREAMLIT_WATCHDOG_ENABLED"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "false"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# ─── Core Imports ─────────────────────────────────────────────────────────────
import ssl
import certifi
import httplib2
import pandas as pd
import streamlit as st
from rich.console import Console
from rich.traceback import install as install_rich_traceback

# ─── Local Modules ────────────────────────────────────────────────────────────
from defect_utils import ROOT_FOLDER_ID_DEFAULT, load_csv_data
from page_plotter import render_plotter_page
from page_structures import render_structures_page
from ai_tool import gpt_query  # ✅ AI import

# ─── SSL & Config ─────────────────────────────────────────────────────────────
install_rich_traceback(show_locals=False)
console = Console()
st.set_page_config(page_title="DefectDB Studio ☀️🔋", layout="wide", page_icon="☀️")

httplib2.CA_CERTS = certifi.where()
ssl.create_default_context(cafile=certifi.where())
console.log("✅ Streamlit configuration initialised.")

# ─── Sidebar Configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration & Data Source")
    st.caption("Provide your Google Drive Root Folder ID to locate defect datasets.")
    root_id = st.text_input("Root Folder ID", value=ROOT_FOLDER_ID_DEFAULT)

    if st.button("🔍 Scan Google Drive"):
        with st.spinner("Scanning Google Drive for 'cdsete_defect_library_generation_pbesol.csv'..."):
            try:
                data = load_csv_data(root_id)
                if data is None:
                    st.error("File 'cdsete_defect_library_generation_pbesol.csv' not found in root.")
                    st.session_state["defect_data"] = None
                else:
                    st.success("✅ Loaded defect data from CSV file.")
                    st.session_state["defect_data"] = data
                    st.session_state["root_folder_id"] = root_id
            except Exception as exc:
                st.error(f"Error loading file: {exc}")
                st.session_state["defect_data"] = None

    st.divider()
    st.header("🔁 Session Control")
    if st.button("🔴 Clear Session & Restart"):
        st.session_state.clear()
        st.success("Session cleared! Please reload the app.")
        st.stop()

# ─── Retrieve Cached Session Data ─────────────────────────────────────────────
defect_data = st.session_state.get("defect_data")
root_folder_for_structures = st.session_state.get("root_folder_id", root_id)

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab_about, tab_data, tab_plot, tab_structures, tab_ai, tab_next = st.tabs([
    "💡 About DefectDB Studio ☀️🔋",
    "📂 Defect Dataset Viewer",
    "📈 Formation Energy Plotter",
    "🧱 Optimized Structures",
    "🤖 AI Q&A",
    "🚀 Next Steps"  # ✅ ADDED NEW TAB
])

# ─── ABOUT TAB ────────────────────────────────────────────────────────────────
with tab_about:
    st.title("☀️🔋 DefectDB Studio")
    st.subheader("An Interactive Database and Visualization Platform for Defect Thermodynamics in Cd–Se–Te")

    with st.container(border=True):
        st.markdown("""
        **Md Habibur Rahman**, **Yi Yang**, and **Arun Mannodi-Kanakkithodi** *School of Materials Engineering, Purdue University* *West Lafayette, IN 47907, USA* 📧 **Contact:** - Md Habibur Rahman — [rahma103@purdue.edu](mailto:rahma103@purdue.edu)  
        - Prof. Arun Mannodi-Kanakkithodi — [amannodi@purdue.edu](mailto:amannodi@purdue.edu)
        """)

    st.info(
        "DefectDB Studio enables researchers to browse, visualize, and analyze defect data "
        "collected from high-throughput DFT and ML workflows. It provides an intuitive interface "
        "to explore formation energies, charge-transition levels, and structural relaxations.",
        icon="☀️"
    )

    # ─── Computational Workflow ───────────────────────────────────────────────
    st.markdown("""
    #### ⚛️ Computational Workflow
    The defect and alloy configurations in this database were first generated using the **Shake-and-Break** method, which systematically perturbs atomic positions to explore diverse local minima and break potential symmetries in defective structures. These candidate structures were then optimized using **Machine Learning Force Fields (MLFFs)** trained on high-throughput DFT data to efficiently reach low-energy configurations.  
    Subsequently, all MLFF-optimized structures were refined at the **Density Functional Theory (DFT)** level using the **PBESol** exchange–correlation functional to ensure accurate energies, geometries, and defect formation thermodynamics.  
    This multistage approach — *Shake-and-Break → MLFF relaxation → PBESol refinement* — enables robust exploration of defect landscapes while drastically reducing computational cost compared to pure DFT-based searches.
    """)

    # ─── References Section ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📚 Related Publications & References")

    with st.container(border=True):
        st.markdown("""
**1. [DeFecT-FF: Accelerated Modeling of Defects in Cd–Zn–Te–Se–S Compounds Combining High-Throughput DFT and Machine Learning Force Fields](https://doi.org/10.48550/arXiv.2510.23514)** *Md Habibur Rahman, Arun Mannodi-Kanakkithodi* **arXiv:** 2510.23514 · **Submitted:** 27 Oct 2025  
A framework for predicting native point defects, impurities, and defect complexes in Cd/Zn-Te/Se/S semiconductors using active-learning-trained ML force fields (MLFFs). Available as a **nanoHUB** tool enabling rapid screening of low-energy defect configurations.  

---

**2. [Introducing CADeT-FF for Accelerated Modeling of Defect Thermodynamics in CdSeTe Solar Cells](https://nanohub.org/tools/cadetff)** *Md Habibur Rahman, Arun Mannodi-Kanakkithodi* **Publication date:** 2025/8/8  
Platform release on nanoHUB.org for automated defect generation, energetics computation, and ML-based analysis of CdSeTe defects.  

---
**3. [Using Machine Learning to Explore Defect Configurations in Cd/Zn–Se/Te Compounds](https://ieeexplore.ieee.org/document/10512345)** *Md Habibur Rahman, Ishaanvi Agrawal, Arun Mannodi-Kanakkithodi* **IEEE 53rd Photovoltaic Specialists Conference (PVSC)** · Pages 0717-0719 (2025)  
Describes a workflow combining DFT, active learning, and GNNs (ALIGNN) to predict crystal formation energies for over 13 000 hypothetical defects in Cd/Zn–Se/Te alloys.  


---

**4. [Defect Modeling in Semiconductors: The Role of First-Principles Simulations and Machine Learning](https://iopscience.iop.org/article/10.1088/2515-7639/adb181/meta)** *Md Habibur Rahman, Arun Mannodi-Kanakkithodi* **Journal of Physics: Materials**, Vol. 8, Issue 2, 022001 (2025) · IOP Publishing  
A comprehensive review covering DFT and ML techniques for studying vacancies, interstitials, substitutionals, and defect complexes in semiconductors.  

---

**5. [Accelerating Defect Predictions in Semiconductors Using Graph Neural Networks](https://pubs.aip.org/aip/aml/article/2/1/016122/3279661)** *Md Habibur Rahman, Prince Gollapalli, Panayotis Manganaris, Satyesh Kumar Yadav, Ghanshyam Pilania, Brian DeCost, Kamal Choudhary, Arun Mannodi-Kanakkithodi* **APL Machine Learning**, Vol. 2, Issue 1 (2024) · AIP Publishing  
Develops a GNN-powered framework for predicting and screening native defects and functional impurities across group IV, III–V, and II–VI semiconductors using one of the largest DFT-based defect datasets to date.  

---

**6. [Understanding Defect-Mediated Ion Migration in Semiconductors Using Atomistic Simulations and Machine Learning](https://pubs.acs.org/doi/full/10.1021/acsmaterialsau.4c00095)** *Md Habibur Rahman, Maitreyo Biswas, Arun Mannodi-Kanakkithodi* **ACS Materials Au**, Vol. 4, Issue 6, Pages 557–573 (2024) · American Chemical Society  
A Perspective discussing mechanisms of defect-mediated ion migration in semiconductors, with emphasis on halide perovskites and methods to suppress phase segregation and migration via composition and strain tuning.  
        """)

# ─── DATA TAB ─────────────────────────────────────────────────────────────────
with tab_data:
    st.header("📂 Loaded Defect Dataset")
    if defect_data is not None:
        st.success("✅ Dataset successfully loaded!")
        st.dataframe(defect_data, use_container_width=True)
        st.caption(f"Total records: **{len(defect_data):,}**")
    else:
        st.warning("Please scan a Google Drive folder from the sidebar to load defect data.")

# ─── PLOT TAB ────────────────────────────────────────────────────────────────
with tab_plot:
    if defect_data is not None:
        render_plotter_page(defect_data)
    else:
        st.info("Scan a Google Drive root folder to load the defect CSV data.")

# ─── STRUCTURES TAB ───────────────────────────────────────────────────────────
with tab_structures:
    if root_folder_for_structures:
        render_structures_page(root_folder_for_structures)
    else:
        st.info("Enter a Google Drive root folder ID to browse optimized structures.")

# ─── AI Q&A TAB ───────────────────────────────────────────────────────────────
with tab_ai:
    st.header("🤖 AI-Powered Defect Q&A")
    st.caption("Ask AI about your defect data for intelligent explanations.")

    if defect_data is None:
        st.warning("⚠️ Please load defect data from the sidebar first (Scan Google Drive).")
    else:
        if "AB" in defect_data.columns and "Defect" in defect_data.columns:
            col1, col2 = st.columns(2)

            with col1:
                compounds = sorted(defect_data["AB"].unique())
                selected_compound = st.selectbox("Select Compound", compounds)

            with col2:
                if selected_compound:
                    defects_for_compound = sorted(
                        defect_data[defect_data["AB"] == selected_compound]["Defect"].unique()
                    )
                    selected_defect = st.selectbox("Select Defect", defects_for_compound)

            st.divider()
            custom_query = st.text_area(
                "Or ask a custom question about defects:",
                placeholder="e.g., 'Explain the stability of As_Te in CdTe' or 'What affects defect formation energy?'",
                height=80
            )

            if st.button("🚀 Ask AI", type="primary"):
                with st.spinner("Contacting AI model..."):
                    if selected_compound and selected_defect:
                        mask = (defect_data["AB"] == selected_compound) & (defect_data["Defect"] == selected_defect)
                        defect_rows = defect_data[mask]

                        if not defect_rows.empty:
                            row = defect_rows.iloc[0]
                            context_info = [
                                f"Compound: {selected_compound}",
                                f"Defect: {selected_defect}"
                            ]
                            if "gap" in row and not pd.isna(row["gap"]):
                                context_info.append(f"Band gap: {row['gap']:.2f} eV")
                            if "VBM" in row and not pd.isna(row["VBM"]):
                                context_info.append(f"VBM: {row['VBM']:.2f} eV")

                            base_info = "\n".join(context_info)
                            question = custom_query.strip() or (
                                f"Explain the defect {selected_defect} in {selected_compound} "
                                f"and discuss its stability and impact on performance."
                            )

                            prompt = f"""You are an expert in semiconductor defect physics.
Given the following data:
{base_info}

Question: {question}

Explain in a scientific yet clear manner for materials researchers."""
                        else:
                            prompt = custom_query or "Explain defect formation in semiconductors."
                    else:
                        prompt = custom_query or "Explain defect formation in semiconductors."

                    # ✅ Uses the gpt_query function from ai_tool.py
                    result = gpt_query(prompt)

                    st.subheader("📝 AI Response")
                    with st.container(border=True):
                        st.markdown(result)

                    with st.expander("🔍 View Prompt"):
                        st.code(prompt, language="text")

        else:
            st.error("❌ Dataset missing required columns ('AB' or 'Defect'). Please check your data source.")

# ─── NEXT STEPS TAB ──────────────────────────────────────────────────────────
with tab_next:
    st.header("🚀 Next Steps: Extending DefectDB Studio")
    st.markdown("""
    The next phase of **DefectDB Studio** focuses on integrating advanced levels of theory and experimental validation.  
    Below are the planned steps to further enhance predictive accuracy and physical realism:
    """)
    st.markdown("""
    1. **Add HSE + SOC Data** Incorporate hybrid functional (HSE06) calculations including spin–orbit coupling to achieve band-edge and defect-level accuracy comparable to experimental results.  

    2. **Integrate XANES Spectra via FEFF9** For each optimized structure, compute **X-ray Absorption Near Edge Structure (XANES)** using the FEFF9 software. This will provide element-specific electronic and local structural fingerprints for each defect.  

    3. **Fitting to Experimental Signatures** Establish a machine learning–based inverse mapping (maybe?), for example  where **input → experimental XANES or defect-level spectra**, and **output → most probable defects** responsible for the observed features.  
        This will allow DefectDB Studio to serve as an intelligent platform that bridges theory and experiment for defect identification and validation.
    """)

console.log("☀️🔋 DefectDB Studio loaded successfully.")
