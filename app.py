"""Main entry point for the DefectDB Studio Streamlit app."""
import ssl

import certifi
import httplib2
import streamlit as st
from rich.console import Console
from rich.traceback import install as install_rich_traceback

from defect_utils import (
    ROOT_FOLDER_ID_DEFAULT,
    load_csv_data,
)
from page_home import render_home_page
from page_plotter import render_plotter_page
from page_structures import render_structures_page

# ── Config & SSL ──────────────────────────────────────────────────────────────
install_rich_traceback(show_locals=False)
console = Console()

st.set_page_config(page_title="DefectDB Studio", layout="wide")
httplib2.CA_CERTS = certifi.where()
ssl.create_default_context(cafile=certifi.where())
console.log("Streamlit configuration initialised.")

# ── Sidebar (Global Controls) ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Source")
    root_id = st.text_input("Root Folder ID", value=ROOT_FOLDER_ID_DEFAULT)

    if st.button("Scan Google Drive"):
        with st.spinner("Scanning Google Drive for 'cdsete_defect_library_generation_pbesol.csv'..."):
            try:
                data = load_csv_data(root_id)
                if data is None:
                    st.error("File 'cdsete_defect_library_generation_pbesol.csv' not found in root.")
                    st.session_state["defect_data"] = None
                else:
                    st.success("Loaded defect data from CSV file.")
                    st.session_state["defect_data"] = data
                    st.session_state["root_folder_id"] = root_id
            except Exception as exc:  # pragma: no cover - displayed to the user
                st.error(f"Error loading file: {exc}")
                st.session_state["defect_data"] = None

# ── Retrieve cached session state ─────────────────────────────────────────────
defect_data = st.session_state.get("defect_data")
root_folder_for_structures = st.session_state.get("root_folder_id", root_id)

# ── Main Tabs ─────────────────────────────────────────────────────────────────
home_tab, plot_tab, structures_tab = st.tabs([
    "🏠 Home",
    "📈 Formation Energy Plot",
    "🧱 Optimized Structures",
])

with home_tab:
    render_home_page()

with plot_tab:
    if defect_data is not None:
        render_plotter_page(defect_data)
    else:
        st.info("Scan a Google Drive root folder to load the defect CSV data.")

with structures_tab:
    if root_folder_for_structures:
        render_structures_page(root_folder_for_structures)
    else:
        st.info("Enter a Google Drive root folder ID to browse optimized structures.")
