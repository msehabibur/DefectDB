"""Landing page components for DefectDB Studio."""
import streamlit as st

INTRO_TEXT = (
    "DefectDB Studio is an interactive web platform that automates the collection, "
    "analysis, and visualization of defect energetics in semiconductors."
)

DETAIL_TEXT = (
    "It connects directly to Google Drive folders containing VASP outputs and "
    "automatically builds a database of formation energies and optimized structures. "
    "Researchers can parse VASP files, compute formation energies, plot charge-state "
    "diagrams, visualize optimized geometries, and integrate correction data from a single "
    "CSV file. This project supports Purdue University's ongoing research on machine "
    "learning–accelerated defect prediction and semiconductor informatics."
)

AUTHORS = [
    "Md Habibur Rahman",
    "Yi Yang",
    "Arun Mannodi-Kanakkithodi",
]

AFFILIATION = (
    "School of Materials Engineering, Purdue University, West Lafayette, IN 47907, USA"
)


def render_home_page() -> None:
    """Render the landing page tab."""
    st.title("🏠 Welcome to DefectDB Studio")

    st.subheader("Project Team")
    with st.container(border=True):
        for author in AUTHORS:
            st.markdown(f"**{author}**")
        st.markdown(f"*{AFFILIATION}*")

    st.markdown("---")
    st.subheader("About the Tool")
    st.write(INTRO_TEXT)
    st.write(DETAIL_TEXT)

    st.markdown("---")
    st.subheader("Quick Start")
    st.markdown(
        "1. Enter a Google Drive root folder ID in the sidebar and scan for the `cdsete_defect_library_generation_pbesol.csv` file.\n"
        "2. Use the **Formation Energy Plot** tab to explore charge-state energetics.\n"
        "3. Browse optimized structures and download geometries from the **Optimized Structures** tab."
    )
