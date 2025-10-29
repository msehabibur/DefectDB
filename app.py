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
from openai import OpenAI
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

# ─── OpenAI Setup ─────────────────────────────────────────────────────────────
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=OPENAI_API_KEY)
except KeyError:
    client = None
    st.warning("⚠️ Please add your OpenAI API key to `.streamlit/secrets.toml` as `OPENAI_API_KEY`.")

# ─── GPT Query Function ───────────────────────────────────────────────────────
def gpt_query(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Query OpenAI GPT model with a given prompt.

    Args:
        prompt (str): The input prompt for GPT.
        model (str): The GPT model name.

    Returns:
        str: The model's response text.
    """
    if client is None:
        return "❌ Error: OpenAI API key not found. Please configure it in .streamlit/secrets.toml."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a materials science expert specializing in semiconductor defects."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error from OpenAI API: {str(e)}"

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
    st.caption("Ask GPT about your defect data for intelligent explanations.")

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

            if st.button("🚀 Ask GPT", type="primary"):
                with st.spinner("Contacting GPT model..."):
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

                    result = gpt_query(prompt)

                    st.subheader("📝 GPT Response")
                    with st.container(border=True):
                        st.markdown(result)

                    with st.expander("🔍 View Prompt"):
                        st.code(prompt, language="text")

        else:
            st.error("❌ Dataset missing required columns ('AB' or 'Defect'). Please check your data source.")

        st.divider()
        st.info("""
        **💡 Tips:**
        - Be specific in your question
        - Uses GPT-4o-mini for fast, accurate responses
        - Ideal for interpreting defect energetics, charge states, and trends
        """, icon="💡")

console.log("🧪 DefectDB Studio loaded successfully.")
