# page_structures.py
import streamlit as st
from typing import Dict

# Import necessary functions from the backend
from defect_utils import (
    discover_defects, 
    discover_charge_states, 
    parse_charge_label_to_q, 
    find_structure_file
)

def render_structures_tab(compounds: Dict[str, str]):
    st.subheader("📦 Select Compound and Defects")

    comp_sel = st.selectbox(
        "Select a compound", 
        list(compounds.keys()), 
        key="struct_comp_sel"
    )
    comp_id = compounds[comp_sel]

    defects = discover_defects(comp_id)
    defect_names = sorted(defects.keys())
    
    chosen_defects = st.multiselect(
        "Defects", 
        defect_names, 
        default=defect_names, 
        key="struct_defect_multi"
    )

    st.markdown("---")
    st.subheader("📥 Download Optimized Structures")

    if st.button("List & download structures"):
        if not chosen_defects:
            st.warning("Please select at least one defect.")
            return

        with st.spinner("Finding structure files in Drive..."):
            drive_defects = discover_defects(comp_id)
            for dname in chosen_defects:
                did = drive_defects.get(dname)
                if not did:
                    st.info(f"Drive: defect folder '{dname}' not found."); continue
                
                charges = discover_charge_states(did)
                if not charges:
                    st.info(f"{dname}: no charge-state subfolders."); continue
                
                # sort by q (desc)
                items = []
                for qlbl, qid in charges.items():
                    q = parse_charge_label_to_q(qlbl)
                    items.append((q, qlbl, qid))
                items.sort(key=lambda x: (x[0] is None, -(x[0] or 0)))
                
                st.markdown(f"**{dname}**")
                for q, qlbl, qid in items:
                    blob, fname = find_structure_file(qid)
                    if blob is None:
                        st.write(f"• {qlbl}: _structure not found_")
                    else:
                        st.download_button(
                            label=f"Download {dname} {qlbl} → {fname}",
                            data=blob,
                            file_name=f"{comp_sel}_{dname}_{qlbl}_{fname}",
                            key=f"dl_{comp_sel}_{dname}_{qlbl}"
                        )
