# app.py
import streamlit as st
import pandas as pd
import ssl
import httplib2
import certifi

# Import the plotter page
from page_plotter import render_plotter_page

# ── Config & SSL ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Defect Plotter", layout="wide")
httplib2.CA_CERTS = certifi.where()
ssl.create_default_context(cafile=certifi.where())

st.title("🧪 Defect Formation Energy Plotter")

# ── File Uploader ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your Defect Data File (e.g., cdsete_defect_library_generation_pbesol.xlsx)", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Read the uploaded Excel file
        data = pd.read_excel(uploaded_file)
        st.success("File loaded successfully!")
        
        # Render the main plotter interface
        render_plotter_page(data)
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.exception(e)
else:
    st.info("Please upload your defect data file to begin.")
