import streamlit as st
from defect_utils import (
    discover_compounds,
    discover_defects,
    discover_charge_states,
    find_structure_file,
    ROOT_FOLDER_ID_DEFAULT,
)

@st.cache_data(show_spinner=False)
def _sorted_items(items):
    """Ensure mapping or list is sorted alphabetically by key."""
    if isinstance(items, dict):
        return tuple(sorted(items.items(), key=lambda kv: kv[0].lower()))
    elif isinstance(items, list):
        # if already a list of dicts (Drive output)
        try:
            return tuple(sorted([(i["name"], i["id"]) for i in items], key=lambda kv: kv[0].lower()))
        except Exception:
            return tuple()
    return tuple()

def render_structures_page(root_folder_id: str = ROOT_FOLDER_ID_DEFAULT):
    st.header("🔹 Optimized Structures (Google Drive)")
    compounds = discover_compounds(root_folder_id)

    if not compounds:
        st.info("No compound folders found. Please check your root folder ID or Drive access.")
        return

    compound_choices = _sorted_items(compounds)
    comp_labels = [label for label, _ in compound_choices]
    comp_ids = {label: value for label, value in compound_choices}

    comp_sel = st.selectbox("Select compound", comp_labels)
    comp_id = comp_ids[comp_sel]

    defects = discover_defects(comp_id)
    defect_choices = _sorted_items(defects)
    def_labels = [label for label, _ in defect_choices]
    def_ids = {label: value for label, value in defect_choices}

    chosen_defects = st.multiselect("Select defects", def_labels, default=def_labels)

    if st.button("List & Download Structures"):
        for dname in chosen_defects:
            st.subheader(f"📁 {dname}")
            defect_id = def_ids.get(dname)
            if not defect_id:
                st.warning(f"Defect {dname} not found.")
                continue

            charges = discover_charge_states(defect_id)
            if not charges:
                st.info(f"{dname}: No charge-state folders found.")
                continue

            for qlbl, qid in sorted(charges.items(), key=lambda kv: kv[0]):
                blob, fname = find_structure_file(qid)
                if blob is None:
                    st.text(f"{qlbl}: structure not found")
                else:
                    st.download_button(
                        label=f"⬇️ Download {dname} {qlbl} → {fname}",
                        data=blob,
                        file_name=f"{comp_sel}_{dname}_{qlbl}_{fname}",
                        key=f"dl_{dname}_{qlbl}"
                    )
