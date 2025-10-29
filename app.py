#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py: Streamlit interface for DefectDB Studio application.
Purpose: Clean UI for AI-powered defect data visualization and analysis.
"""

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
os.environ["STREAMLIT_WATCHDOG_ENABLED"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# ─── Core Imports ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict

# ─── AI Tool Integration ──────────────────────────────────────────────────────
from ai_tool import (
    initialize_openai_client,
    load_local_csvs,
    build_context_from_csvs,
    query_ai,
    get_csv_summary
)

# ─── Streamlit Configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="DefectDB Studio",
    layout="wide",
    page_icon="🧪"
)


# ─── Initialize OpenAI Client ─────────────────────────────────────────────────
@st.cache_resource
def get_openai_client():
    """Initialize and cache OpenAI client."""
    return initialize_openai_client()


# ─── Load CSV Data ────────────────────────────────────────────────────────────
@st.cache_data
def get_local_csv_data() -> Dict[str, pd.DataFrame]:
    """Load and cache local CSV data."""
    return load_local_csvs("./data")


# ─── Sidebar Navigation ───────────────────────────────────────────────────────
st.sidebar.title("DefectDB Studio")
st.sidebar.caption("AI-Powered Defect Analysis Platform")

page = st.sidebar.selectbox(
    "Select Page",
    ["AI Assistant", "Formation Energy Plot"]
)

st.sidebar.divider()
st.sidebar.caption("Version 2.0")


# ─── PAGE 1: AI Assistant ─────────────────────────────────────────────────────
if page == "AI Assistant":
    st.title("AI Materials Assistant")
    st.caption("Ask questions about defect formation energy and semiconductor thermodynamics")

    # Initialize OpenAI client
    client = get_openai_client()

    if client is None:
        st.warning(
            "OpenAI API key not configured. "
            "Please add your API key to .streamlit/secrets.toml as OPENAI_API_KEY."
        )
    else:
        # Load local CSV data
        csv_data = get_local_csv_data()

        if csv_data:
            st.success(f"Loaded {len(csv_data)} CSV file(s) from ./data directory")

            with st.expander("View loaded data files"):
                st.text(get_csv_summary(csv_data))
        else:
            st.info("No CSV files found in ./data directory. Add .csv files to enable data-driven analysis.")

        st.divider()

        # User input area
        user_question = st.text_area(
            "Your Question",
            placeholder="e.g., Explain defect formation of As_Te in CdTe",
            height=100,
            help="Ask about defect physics, formation energies, charge states, or material properties"
        )

        # Ask AI button
        if st.button("Ask AI", type="primary", use_container_width=True):
            if not user_question.strip():
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Analyzing your question with AI..."):
                    # Build context from CSV data
                    context = build_context_from_csvs(csv_data, max_rows=3)

                    # Query the AI
                    response = query_ai(
                        prompt=user_question,
                        context=context,
                        client=client
                    )

                    # Display response
                    st.divider()
                    st.subheader("Response")

                    with st.container(border=True):
                        st.markdown(response)


# ─── PAGE 2: Formation Energy Plot ────────────────────────────────────────────
elif page == "Formation Energy Plot":
    st.title("Formation Energy Plot")
    st.caption("Visualize formation energies across different compounds")

    # Compound selection
    compounds = ["CdTe", "CdSeTe", "CdSe", "CdZnTe", "ZnTe"]

    selected_compound = st.selectbox(
        "Select Compound",
        compounds,
        help="Choose a compound to visualize formation energy data"
    )

    st.divider()

    # Placeholder data for demonstration
    # This can be connected to actual CSV data later
    defect_types = ["V_Cd", "V_Te", "As_Te", "Sb_Te", "Cd_i", "Te_i"]
    formation_energies = np.random.uniform(0.5, 3.5, len(defect_types))

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        "Defect": defect_types,
        "Formation Energy (eV)": formation_energies
    })

    # Display plot
    st.subheader(f"Formation Energies for {selected_compound}")

    # Bar chart
    st.bar_chart(
        plot_data.set_index("Defect"),
        height=400,
        use_container_width=True
    )

    # Data table
    with st.expander("View data table"):
        st.dataframe(
            plot_data,
            use_container_width=True,
            hide_index=True
        )

    st.info(
        f"Showing placeholder formation energy data for {selected_compound}. "
        "Connect to CSV data to display actual computed values."
    )


# ─── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("Developed at Purdue University")
st.sidebar.caption("School of Materials Engineering")
