# app.py
import streamlit as st
import ssl
import httplib2
import certifi

# Import our new modules
from defect_utils import (
    discover_compounds, 
    load_root_params, 
    DEFAULT_VBM, 
    DEFAULT_GAP,
    ROOT_FOLDER_ID_DEFAULT
)
from page_plotter import render_plotter_tab
from page_structures import render_structures_tab

# ── Config & SSL ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="DefectDB Studio", layout="wide")
httplib2.CA_CERTS = certifi.where()
ssl.create_default_context(cafile=certifi.where())

st.title("🧪 DefectDB Studio")

# ── Sidebar (Global Controls) ────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Source")
    root_id = st.text_input("Root Folder ID", value=ROOT_FOLDER_ID_DEFAULT)
    
    if st.button("Scan Root"):
        with st.spinner("Scanning Google Drive..."):
            try:
                comps = discover_compounds(root_id)
                if not comps:
                    st.error("No compound folders found (or missing permissions).")
                    st.stop()
                
                # Store discovered compounds and params in session state
                st.session_state["compounds"] = dict(sorted(comps.items(), key=lambda x: x[0].lower()))
                st.session_state["root_params"] = load_root_params(root_id)
                st.session_state["root_id"] = root_id # Store for later use
                
                if st.session_state["root_params"] is None:
                    st.warning("ROOT/data.csv not found — using defaults (VBM=0, gap=1.5) and μ=None.")
                else:
                    st.success("Loaded ROOT/data.csv.")
                st.success(f"Found {len(comps)} compound folder(s).")
            
            except Exception as e:
                st.error(f"Error: {e}")

# ── Main Page (Tabs) ─────────────────────────────────────────────────────────

# Get data from session state
compounds = st.session_state.get("compounds")
root_params = st.session_state.get("root_params")
scanned_root_id = st.session_state.get("root_id")

if compounds:
    tab1, tab2 = st.tabs(["📈 Formation Energy Plot", "🧱 Optimized Structures"])

    with tab1:
        # Pass all necessary data to the plotter tab function
        render_plotter_tab(
            root_id=scanned_root_id, 
            compounds=compounds, 
            root_params=root_params
        )

    with tab2:
        # Pass necessary data to the structures tab function
        render_structures_tab(compounds=compounds)

else:
    st.info("Scan a Root Folder ID in the sidebar to begin.")

# ── Notes Expander (REMOVED) ─────────────────────────────────────────────────
