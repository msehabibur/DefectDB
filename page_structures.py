"""Optimized structures tab for DefectDB Studio."""
from __future__ import annotations

from typing import Dict, Tuple

import streamlit as st

from defect_utils import (
    discover_charge_states,
    discover_compounds,
    discover_defects,
    find_structure_file,
    parse_charge_label_to_q,
)


@st.cache_data(show_spinner=False)
def _sorted_items(items: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    """Return a stable, alphabetically sorted view of a mapping."""
    return tuple(sorted(items.items(), key=lambda kv: kv[0].lower()))


def render_structures_page(root_folder_id: str) -> None:
    """Render the optimized structures tab."""
    st.header("🧱 Optimized Structures")

    compounds = discover_compounds(root_folder_id)
    if not compounds:
        st.info("No compound folders were found in the provided Google Drive root folder.")
        return

    compound_choices = _sorted_items(compounds)
    comp_labels = [label for label, _ in compound_choices]
    comp_ids = {label: value for label, value in compound_choices}

    selected_compound = st.selectbox("Select a compound", comp_labels, key="structures_compound")
    compound_id = comp_ids[selected_compound]

    discovered_defects = discover_defects(compound_id)
    if not discovered_defects:
        st.info("No defect folders were found for the selected compound.")
        return

    defect_choices = _sorted_items(discovered_defects)
    defect_labels = [label for label, _ in defect_choices]
    selected_defects = st.multiselect(
        "Select defects",
        defect_labels,
        default=defect_labels,
        key=f"structures_defects_{compound_id}",
    )

    if not selected_defects:
        st.warning("Select at least one defect to view charge states and downloads.")
        return

    st.markdown("---")
    st.subheader("Available structures")

    drive_defects = discover_defects(compound_id)
    for defect_name in selected_defects:
        defect_id = drive_defects.get(defect_name)
        if not defect_id:
            st.info(f"{defect_name}: defect folder not found in Drive.")
            continue

        charges = discover_charge_states(defect_id)
        if not charges:
            st.info(f"{defect_name}: no charge-state folders detected.")
            continue

        charge_entries = []
        for charge_label, charge_id in charges.items():
            charge_value = parse_charge_label_to_q(charge_label)
            charge_entries.append((charge_value, charge_label, charge_id))

        # Sort from highest charge to lowest, with unknown charges last
        charge_entries.sort(key=lambda entry: (entry[0] is None, -(entry[0] or 0)))

        st.markdown(f"### {defect_name}")
        for charge_value, charge_label, charge_folder_id in charge_entries:
            blob, filename = find_structure_file(charge_folder_id)
            charge_display = charge_label
            if charge_value is not None and charge_label.lower() != f"q{charge_value:+d}".lower():
                charge_display = f"{charge_label} (q = {charge_value:+d})"

            if blob is None:
                st.write(f"• {charge_display}: _structure file not found_")
                continue

            st.download_button(
                label=f"Download {charge_display} → {filename}",
                data=blob,
                file_name=f"{selected_compound}_{defect_name}_{charge_label}_{filename}",
                key=f"download_{selected_compound}_{defect_name}_{charge_label}",
            )
