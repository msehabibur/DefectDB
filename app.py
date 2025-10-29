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
import requests
from rich.console import Console
from rich.traceback import install as install_rich_traceback

# ─── Local Modules ────────────────────────────────────────────────────────────
from defect_utils import (
    ROOT_FOLDER_ID_DEFAULT,
    load_csv_data,
)
from page_plotter import render_plotter_page
from page_structures import render_structures_page

# ─── SSL & Config ─────────────────────────────────────────────────────────────
install_rich_traceback(show_locals=False)
console = Console()
st.set_page_config(page_title="DefectDB Studio", layout="wide", page_icon="🧪")

httplib2.CA_CERTS = certifi.where()
ssl.create_default_context(cafile=certifi.where())
console.log("✅ Streamlit configuration initialised.")

# ─── AI Integration Functions ─────────────────────────────────────────────────
def hf_query(prompt: str, model: str = "mistralai/Mistral-7B-Instruct-v0.2") -> str:
    """
    Query Hugging Face Inference API with a prompt.

    Args:
        prompt: The input prompt for the model
        model: The Hugging Face model to use

    Returns:
        The generated text response
    """
    try:
        HF_API_TOKEN = st.secrets.get("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")
        if not HF_API_TOKEN:
            return "❌ Error: HF_API_TOKEN not found in secrets. Please configure it in .streamlit/secrets.toml"

        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 250,
                "temperature": 0.7,
                "top_p": 0.9,
                "return_full_text": False
            }
        }

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No response generated.")
            return "Unexpected response format from API."
        else:
            return f"❌ API Error {response.status_code}: {response.text}"

    except requests.exceptions.Timeout:
        return "❌ Request timed out. The model may be loading. Please try again in a moment."
    except Exception as e:
        return f"❌ Error: {str(e)}"

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
tab_about, tab_data, tab_plot, tab_structures, tab_ai = st.tabs([
    "💡 About DefectDB Studio",
    "📂 Defect Dataset Viewer",
    "📈 Formation Energy Plotter",
    "🧱 Optimized Structures",
    "🤖 AI Q&A"
])

# ─── ABOUT TAB ────────────────────────────────────────────────────────────────
with tab_about:
    st.title("🧪 DefectDB Studio")
    st.subheader("An Interactive Database and Visualization Platform for Defect Thermodynamics in Cd–Se–Te")

    with st.container(border=True):
        st.markdown("""
        **Md Habibur Rahman**, **Yi Yang**, and **Arun Mannodi-Kanakkithodi**
        *School of Materials Engineering, Purdue University*
        *West Lafayette, IN 47907, USA*
        """)

    st.info(
        "DefectDB Studio enables researchers to browse, visualize, and analyze defect data "
        "collected from high-throughput DFT and ML workflows. It provides an intuitive interface "
        "to explore formation energies, charge-transition levels, and structural relaxations.",
        icon="🔬"
    )

    st.markdown("---")
    st.header("🧭 Workflow Overview")

    with st.expander("📁 **1️⃣ Data Integration**", expanded=True):
        st.markdown("""
        - Connects to a specified **Google Drive folder** and searches for  
          `cdsete_defect_library_generation_pbesol.csv`.  
        - Once loaded, the data is cached for fast, interactive exploration.
        """)

    with st.expander("📊 **2️⃣ Formation Energy Visualization**", expanded=True):
        st.markdown("""
        - Each entry includes charge states, total energies, and reference chemical potentials.  
        - Visualize **formation energy vs Fermi level** diagrams with transition-level markers.  
        - Supports comparison across **multiple compositions** (e.g., CdSeₓTe₁₋ₓ).
        """)

    with st.expander("🧱 **3️⃣ Structure Browser**", expanded=True):
        st.markdown("""
        - Access **relaxed POSCAR/CIF files** of bulk and defect structures directly from Drive.
        """)

    st.markdown("---")
    st.header("💡 Purpose and Vision")
    st.markdown("""
    **DefectDB Studio** aims to
    - Democratize access to curated defect datasets for **semiconductors** and **chalcogenides**,
    - Accelerate discovery of **defect-tolerant materials** for renewable energy, and
    - Promote **FAIR data principles** (Findable, Accessible, Interoperable, Reusable).
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
    st.caption("Ask questions about defects and get AI-generated explanations based on the loaded dataset.")

    if defect_data is None:
        st.warning("⚠️ Please load defect data from the sidebar first (Scan Google Drive).")
    else:
        # Check if required columns exist
        if "AB" in defect_data.columns and "Defect" in defect_data.columns:
            col1, col2 = st.columns(2)

            with col1:
                # Get unique compounds
                compounds = sorted(defect_data["AB"].unique())
                selected_compound = st.selectbox("Select Compound", compounds)

            with col2:
                # Get defects for selected compound
                if selected_compound:
                    defects_for_compound = sorted(
                        defect_data[defect_data["AB"] == selected_compound]["Defect"].unique()
                    )
                    selected_defect = st.selectbox("Select Defect", defects_for_compound)

            # Custom query input
            st.divider()
            custom_query = st.text_area(
                "Or ask a custom question about defects:",
                placeholder="e.g., 'Explain the stability of As_Te in CdTe' or 'What affects defect formation energy?'",
                height=80
            )

            if st.button("🚀 Analyze with AI", type="primary"):
                with st.spinner("Querying AI model... This may take a moment if the model is loading."):
                    # Prepare context from data
                    if selected_compound and selected_defect:
                        # Filter data for selected compound and defect
                        mask = (defect_data["AB"] == selected_compound) & (defect_data["Defect"] == selected_defect)
                        defect_rows = defect_data[mask]

                        if not defect_rows.empty:
                            # Extract relevant information
                            row = defect_rows.iloc[0]
                            context_info = []
                            context_info.append(f"Compound: {selected_compound}")
                            context_info.append(f"Defect: {selected_defect}")

                            # Add available data fields
                            if "gap" in row and not pd.isna(row["gap"]):
                                context_info.append(f"Band gap: {row['gap']:.2f} eV")
                            if "VBM" in row and not pd.isna(row["VBM"]):
                                context_info.append(f"VBM: {row['VBM']:.2f} eV")

                            base_info = "\n".join(context_info)

                            # Determine the question
                            if custom_query.strip():
                                question = custom_query
                            else:
                                question = f"Explain the defect {selected_defect} in {selected_compound} and discuss its stability and potential impact on material properties."

                            # Build prompt
                            prompt = f"""Given the following defect data:
{base_info}

Question: {question}

Please provide a clear, scientific explanation in simple terms suitable for materials science researchers."""

                        else:
                            prompt = custom_query if custom_query.strip() else "Explain defect formation in semiconductors."
                    else:
                        prompt = custom_query if custom_query.strip() else "Explain defect formation in semiconductors."

                    # Query the AI model
                    result = hf_query(prompt)

                    # Display results
                    st.subheader("📝 AI Response")
                    with st.container(border=True):
                        st.markdown(result)

                    # Show the prompt used (for transparency)
                    with st.expander("🔍 View Full Prompt Sent to AI"):
                        st.code(prompt, language="text")

        else:
            st.error("❌ Dataset is missing required columns ('AB' or 'Defect'). Please check your data source.")

        # Add helpful information
        st.divider()
        st.info("""
        **💡 Tips for Better Results:**
        - Be specific in your questions
        - The AI uses the Mistral-7B model via Hugging Face
        - First request may be slower if the model needs to load
        - Responses are AI-generated and should be verified with domain expertise
        """, icon="💡")

console.log("🧪 DefectDB Studio loaded successfully.")
