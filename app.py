# app.py
import streamlit as st
import pandas as pd
import ssl
import httplib2
import certifi

# Import our new modules
from defect_utils import load_excel_data, ROOT_FOLDER_ID_DEFAULT
from page_plotter import render_plotter_page

# ── Config & SSL ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Defect Plotter", layout="wide")
httplib2.CA_CERTS = certifi.where()
ssl.create_default_context(cafile=certifi.where())

st.title("🧪 Defect Formation Energy Plotter")

# ── Sidebar (Global Controls) ────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Source")
    root_id = st.text_input("Root Folder ID", value=ROOT_FOLDER_ID_DEFAULT)
    
    if st.button("Scan Root for Excel File"):
        # --- FILENAME UPDATED HERE ---
        with st.spinner("Scanning Google Drive for 'cdsete_defect_library_generation_pbesol'..."):
            try:
                data = load_excel_data(root_id)
                if data is None:
                    st.error("File 'cdsete_defect_library_generation_pbesol' not found in root.")
                    st.session_state["defect_data"] = None
                else:
                    st.success("Loaded defect data from Excel file.")
                    st.session_state["defect_data"] = data
            
            except Exception as e:
                st.error(f"Error loading file: {e}")
                st.session_state["defect_data"] = None

# ── Main Page (Plotter) ──────────────────────────────────────────────────────
defect_data = st.session_state.get("defect_data")

if defect_data is not None:
    # We have data, so render the plotter UI
    render_plotter_page(defect_data)
else:
    st.info("Scan a Root Folder ID in the sidebar to load the defect data file.")
