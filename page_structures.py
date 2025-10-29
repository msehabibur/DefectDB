import io
import streamlit as st
from defect_utils import (
    discover_compounds,
    discover_defects,
    discover_charge_states,
    find_structure_file,
    ROOT_FOLDER_ID_DEFAULT,
)

# Note: pymatgen.vis.structure_vtk.StructureVis is deprecated and won't display anything.
# This implementation uses Crystal Toolkit for visualization instead.

try:
    from crystal_toolkit.components import StructureMoleculeComponent
    CRYSTAL_TOOLKIT_AVAILABLE = True
except ImportError:
    CRYSTAL_TOOLKIT_AVAILABLE = False

try:
    from pymatgen.core import Structure
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


def _format_compound_latex(compound_name: str) -> str:
    """Convert compound names to LaTeX format with subscripts."""
    import re

    if not compound_name:
        return compound_name

    try:
        pattern = r'(\d+\.?\d*)'

        def replace_numbers(match):
            num = match.group(1)
            try:
                if '.' in num or (len(num) == 1 and int(num) < 10):
                    return f'$_{{{num}}}$'
            except (ValueError, TypeError):
                pass
            return num

        parts = re.split(r'(?=[A-Z])', compound_name)
        formatted_parts = []

        for part in parts:
            if not part:
                continue
            match = re.match(r'([A-Z][a-z]?)(.*)$', part)
            if match:
                element = match.group(1)
                rest = match.group(2)
                if rest and re.match(r'^\d', rest):
                    rest = re.sub(pattern, replace_numbers, rest)
                formatted_parts.append(element + rest)
            else:
                formatted_parts.append(part)

        return ''.join(formatted_parts)
    except Exception:
        return compound_name


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

    # Try to extract number
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
    """Visualize structure using available tools."""

    if not PYMATGEN_AVAILABLE:
        st.warning("⚠️ pymatgen is not installed. Structure visualization is not available.")
        st.info("Install with: `pip install pymatgen`")
        return

    try:
        # Parse structure from blob
        structure_text = structure_blob.decode('utf-8')

        # Determine file format and parse
        if filename.lower().endswith('.cif'):
            structure = Structure.from_str(structure_text, fmt='cif')
        else:  # POSCAR/CONTCAR
            structure = Structure.from_str(structure_text, fmt='poscar')

        st.success(f"✅ Structure loaded: {structure.composition.reduced_formula}")

        # Show structure information
        with st.expander("📊 Structure Information", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Formula", structure.composition.reduced_formula)
                st.metric("# of Sites", len(structure))
            with col2:
                st.metric("Volume (Ų)", f"{structure.volume:.2f}")
                st.metric("Density (g/cm³)", f"{structure.density:.3f}")
            with col3:
                lattice = structure.lattice
                st.metric("a (Å)", f"{lattice.a:.3f}")
                st.metric("b (Å)", f"{lattice.b:.3f}")
                st.metric("c (Å)", f"{lattice.c:.3f}")

        # Try Crystal Toolkit visualization
        if CRYSTAL_TOOLKIT_AVAILABLE:
            st.subheader("🔬 3D Structure Visualization")
            try:
                struct_component = StructureMoleculeComponent(structure, id=f"structure_{compound}_{defect}_{charge}")
                struct_component.show()
            except Exception as e:
                st.warning(f"Could not display 3D visualization: {e}")
                st.info("Showing structure data instead...")
                st.json(structure.as_dict())
        else:
            st.info("💡 Install Crystal Toolkit for interactive 3D visualization:")
            st.code("pip install crystal-toolkit", language="bash")

            # Show structure as JSON for reference
            with st.expander("📄 Structure Data (JSON format)", expanded=False):
                st.json(structure.as_dict())

        # Show lattice parameters
        with st.expander("📐 Lattice Parameters", expanded=False):
            lattice = structure.lattice
            st.write(f"**a** = {lattice.a:.4f} Å")
            st.write(f"**b** = {lattice.b:.4f} Å")
            st.write(f"**c** = {lattice.c:.4f} Å")
            st.write(f"**α** = {lattice.alpha:.2f}°")
            st.write(f"**β** = {lattice.beta:.2f}°")
            st.write(f"**γ** = {lattice.gamma:.2f}°")

        # Show atomic positions
        with st.expander("⚛️ Atomic Positions", expanded=False):
            import pandas as pd
            positions_data = []
            for i, site in enumerate(structure):
                positions_data.append({
                    "Site": i + 1,
                    "Element": site.species_string,
                    "x": f"{site.frac_coords[0]:.6f}",
                    "y": f"{site.frac_coords[1]:.6f}",
                    "z": f"{site.frac_coords[2]:.6f}",
                })
            df_positions = pd.DataFrame(positions_data)
            st.dataframe(df_positions, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error visualizing structure: {e}")
        st.info("Showing raw structure file instead...")
        with st.expander("📄 Raw Structure File", expanded=True):
            st.code(structure_text, language="text")


def render_structures_page(root_folder_id: str = ROOT_FOLDER_ID_DEFAULT):
    st.header("🧱 Optimized Structures")

    st.info("""
    **Interactive Structure Browser**

    Select a compound, then a defect, then a charge state to visualize and download structures.
    """)

    # Step 1: Select Compound
    st.subheader("Step 1️⃣: Select Compound")
    compounds = discover_compounds(root_folder_id)

    if not compounds:
        st.warning("No compound folders found. Please check your root folder ID or Drive access.")
        return

    compound_choices = _sorted_items(compounds)
    comp_labels = [label for label, _ in compound_choices]
    comp_ids = {label: value for label, value in compound_choices}

    # Format compound names with LaTeX subscripts for display
    comp_display_names = [_format_compound_latex(label) for label in comp_labels]
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

    # Step 2: Select Defect
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

    # Step 3: Select Charge State
    st.divider()
    st.subheader("Step 3️⃣: Select Charge State")

    charges = discover_charge_states(defect_id)
    if not charges:
        st.warning(f"No charge states found for {defect_sel}")
        return

    # Parse and format charge labels
    charge_display_map = {}
    for orig_label in charges.keys():
        display_label = _parse_charge_label(orig_label)
        charge_display_map[display_label] = orig_label

    charge_display_labels = sorted(
        charge_display_map.keys(),
        key=lambda x: int(x) if x not in ['+', '-'] else 0
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

    # Step 4: Load and Visualize Structure
    st.divider()
    st.subheader(f"📊 Structure: {comp_sel_display} | {defect_sel} | Charge {charge_sel_display}")

    with st.spinner("Loading structure from Google Drive..."):
        structure_blob, structure_filename = find_structure_file(charge_folder_id)

    if structure_blob is None:
        st.error("❌ Structure file not found for this charge state.")
        return

    st.success(f"✅ Found structure file: **{structure_filename}**")

    # Visualize the structure
    _visualize_structure(structure_blob, structure_filename, comp_sel, defect_sel, charge_sel_display)

    # Step 5: Download Option
    st.divider()
    st.subheader("💾 Download Structure")

    download_confirm = st.checkbox("Do you want to download this structure (POSCAR/CIF)?")

    if download_confirm:
        st.download_button(
            label=f"⬇️ Download {structure_filename}",
            data=structure_blob,
            file_name=f"{comp_sel}_{defect_sel}_charge{charge_sel_display}_{structure_filename}",
            key=f"download_{comp_sel}_{defect_sel}_{charge_sel_display}",
            mime="text/plain",
            use_container_width=True
        )
        st.success("✅ Click the button above to download. Your selections will remain unchanged.")
