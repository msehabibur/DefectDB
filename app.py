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
import streamlit as st
from rich.console import Console
from rich.traceback import install as install_rich_traceback

# ─── Local Modules ────────────────────────────────────────────────────────────
from defect_utils import (
    ROOT_FOLDER_ID_DEFAULT,
    load_csv_data,
    format_compound_latex,
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
tab_about, tab_data, tab_plot, tab_structures = st.tabs([
    "💡 About DefectDB Studio",
    "📂 Defect Dataset Viewer",
    "📈 Formation Energy Plotter",
    "🧱 Optimized Structures"
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

console.log("🧪 DefectDB Studio loaded successfully.")
