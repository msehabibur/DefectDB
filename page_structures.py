import io
import streamlit as st
from defect_utils import (
    discover_compounds,
    discover_defects,
    discover_charge_states,
    find_structure_file,
    ROOT_FOLDER_ID_DEFAULT,
)

# ASE + Pymatgen imports
try:
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter
    from ase.io import read
    from ase.visualize.plot import plot_atoms
    import matplotlib.pyplot as plt
    import pandas as pd
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False


@st.cache_data(show_spinner=False)
def _sorted_items(items):
    """Ensure mapping or list is sorted alphabetically by key."""
    if isinstance(items, dict):
        return tuple(sorted(items.items(), key=lambda kv: kv[0].lower()))
    elif isinstance(items, list):
        try:
            return tuple(sorted([(i["name"], i["id"]) for i in items], key=lambda kv: kv[0].lower()))
        except Exception:
            return tuple()
    return tuple()


def _clean_compound_name(name: str) -> str:
    """Remove LaTeX-style subscripts and formatting from compound names."""
    import re
    name = re.sub(r"\$_\{[^}]*\}", "", name)
    name = name.replace("\\", "").replace("{", "").replace("}", "")
    return name.strip()


def _parse_charge_label(label: str) -> str:
    """Convert charge folder labels to display format (+2, +1, 0, -1, -2)."""
    s = (label or "").strip().lower()

    if s in {"neutral", "neut", "0", "q0"}:
        return "0"
    if s.startswith("charged+"):
        return f"+{s.replace('charged+', '')}"
    if s.startswith("charged-"):
        return f"-{s.replace('charged-', '')}"
    if s.startswith("q+"):
        return f"+{s.replace('q+', '')}"
    if s.startswith("q-"):
        return f"-{s.replace('q-', '')}"
    if s.startswith("p"):
        return f"+{s[1:]}"
    if s.startswith("m"):
        return f"-{s[1:]}"

    import re
    m = re.search(r'([+\-]?\d+)', s)
    if m:
        num_str = m.group(1)
        if not num_str.startswith(('+', '-')):
            num = int(num_str)
            if num > 0:
                return f"+{num}"
            elif num < 0:
                return str(num)
            return "0"
        return num_str
    return label


def _visualize_structure(structure_blob: bytes, filename: str, compound: str, defect: str, charge: str):
    """Visualize structure using ASE, converting POSCAR → CIF via Pymatgen."""
    if not PYMATGEN_AVAILABLE:
        st.warning("⚠️ ASE or pymatgen not installed. Visualization unavailable.")
        return

    try:
        # Decode file contents
        structure_text = structure_blob.decode("utf-8")

        # Load with Pymatgen (CIF or POSCAR)
        fmt = "cif" if filename.lower().endswith(".cif") else "poscar"
        structure = Structure.from_str(structure_text, fmt=fmt)

        # Convert to CIF in-memory for ASE visualization
        from io import StringIO
        cif_buffer = StringIO()
        cif_writer = CifWriter(structure)
        cif_writer.write_file(cif_buffer)
        cif_data = cif_writer.__str__()
        cif_temp = StringIO(cif_data)

        # Load CIF via ASE
        atoms = read(cif_temp, format="cif")

        st.success(f"✅ Structure loaded: {structure.composition.reduced_formula}")

        # === Structure information ===
        with st.expander("📊 Structure Information", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Formula", structure.composition.reduced_formula)
                st.metric("# of Sites", len(structure))
            with col2:
                st.metric("Volume (Å³)", f"{structure.volume:.2f}")
                st.metric("Density (g/cm³)", f"{structure.density:.3f}")
            with col3:
                lat = structure.lattice
                st.metric("a (Å)", f"{lat.a:.3f}")
                st.metric("b (Å)", f"{lat.b:.3f}")
                st.metric("c (Å)", f"{lat.c:.3f}")

        # === ASE Visualization ===
        st.subheader("🔬 Crystal Structure (ASE Visualization)")

        fig, ax = plt.subplots(figsize=(6, 6))
        plot_atoms(atoms, ax, rotation=("45x,10y,0z"), radii=0.35, show_unit_cell=2)
        ax.set_title(f"{compound} | {defect} | Charge {charge}", fontsize=12)
        st.pyplot(fig)

        # === Lattice parameters ===
        with st.expander("📐 Lattice Parameters", expanded=False):
            st.write(f"**a** = {lat.a:.3f} Å, **b** = {lat.b:.3f} Å, **c** = {lat.c:.3f} Å")
            st.write(f"**α** = {lat.alpha:.2f}°, **β** = {lat.beta:.2f}°, **γ** = {lat.gamma:.2f}°")

        # === Atomic positions ===
        with st.expander("⚛️ Atomic Positions", expanded=False):
            positions = []
            for site in structure:
                positions.append({
                    "Element": site.species_string,
                    "x": f"{site.frac_coords[0]:.6f}",
                    "y": f"{site.frac_coords[1]:.6f}",
                    "z": f"{site.frac_coords[2]:.6f}",
                })
            st.dataframe(pd.DataFrame(positions), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error visualizing structure: {e}")
        with st.expander("📄 Raw Structure File", expanded=True):
            st.code(structure_blob.decode("utf-8"), language="text")


def render_structures_page(root_folder_id: str = ROOT_FOLDER_ID_DEFAULT):
    """Main Streamlit page for browsing and visualizing defect structures."""
    st.header("🧱 Optimized Structures")

    st.markdown("""
    **Interactive Structure Browser**

    Select a compound → defect → charge state to visualize and download optimized crystal structures.  
    Visualization is powered by **ASE (Atomic Simulation Environment)**.
    """)

    # === Step 1️⃣: Select Compound ===
    st.subheader("Step 1️⃣: Select Compound")
    compounds = discover_compounds(root_folder_id)
    if not compounds:
        st.warning("No compound folders found. Please check your Google Drive root folder ID.")
        return

    compound_choices = _sorted_items(compounds)
    comp_labels = [label for label, _ in compound_choices]
    comp_ids = {label: value for label, value in compound_choices}

    # Show plain names (no LaTeX)
    comp_display_names = [_clean_compound_name(label) for label in comp_labels]
    comp_display_map = {display: original for display, original in zip(comp_display_names, comp_labels)}

    comp_sel_display = st.selectbox(
        "Choose a compound:",
        comp_display_names,
        index=None,
        placeholder="Select a compound..."
    )
    if not comp_sel_display:
        st.info("👆 Please select a compound to continue")
        return

    comp_sel = comp_display_map[comp_sel_display]
    comp_id = comp_ids[comp_sel]

    # === Step 2️⃣: Select Defect ===
    st.divider()
    st.subheader("Step 2️⃣: Select Defect")
    defects = discover_defects(comp_id)
    if not defects:
        st.warning(f"No defects found for {comp_sel_display}")
        return

    defect_choices = _sorted_items(defects)
    def_labels = [label for label, _ in defect_choices]
    def_ids = {label: value for label, value in defect_choices}

    defect_sel = st.selectbox(
        "Choose a defect:",
        def_labels,
        index=None,
        placeholder="Select a defect (e.g., As_Te, Cl_Te, V_Cd)..."
    )
    if not defect_sel:
        st.info("👆 Please select a defect to continue")
        return

    defect_id = def_ids[defect_sel]

    # === Step 3️⃣: Select Charge State ===
    st.divider()
    st.subheader("Step 3️⃣: Select Charge State")
    charges = discover_charge_states(defect_id)
    if not charges:
        st.warning(f"No charge states found for {defect_sel}")
        return

    charge_display_map = {_parse_charge_label(k): k for k in charges.keys()}
    charge_display_labels = sorted(
        charge_display_map.keys(),
        key=lambda x: int(x) if x.lstrip("+-").isdigit() else 0
    )

    charge_sel_display = st.selectbox(
        "Choose a charge state:",
        charge_display_labels,
        index=None,
        placeholder="Select charge state (e.g., +2, +1, 0, -1, -2)..."
    )
    if not charge_sel_display:
        st.info("👆 Please select a charge state to view the structure")
        return

    charge_orig_label = charge_display_map[charge_sel_display]
    charge_folder_id = charges[charge_orig_label]

    # === Step 4️⃣: Load and visualize structure ===
    st.divider()
    st.subheader(f"📊 Structure: {comp_sel_display} | {defect_sel} | Charge {charge_sel_display}")

    with st.spinner("Loading structure from Google Drive..."):
        structure_blob, structure_filename = find_structure_file(charge_folder_id)

    if structure_blob is None:
        st.error("❌ Structure file not found for this charge state.")
        return

    st.success(f"✅ Found structure file: **{structure_filename}**")
    _visualize_structure(structure_blob, structure_filename, comp_sel, defect_sel, charge_sel_display)

    # === Step 5️⃣: Download Option ===
    st.divider()
    st.subheader("💾 Download Structure")

    download_confirm = st.checkbox("Enable download of this structure (POSCAR/CIF)")
    if download_confirm:
        st.download_button(
            label=f"⬇️ Download {structure_filename}",
            data=structure_blob,
            file_name=f"{comp_sel}_{defect_sel}_charge{charge_sel_display}_{structure_filename}",
            mime="text/plain",
            use_container_width=True
        )
        st.success("✅ Click above to download. Your selections remain saved.")
