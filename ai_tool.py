#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_tool.py: AI logic and data access module for DefectDB Studio.
Purpose: Handles OpenAI integration, local CSV loading, and optional Google Drive access.
"""

import os
import glob
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import streamlit as st
from openai import OpenAI


# ─── OpenAI Client Initialization ────────────────────────────────────────────
def initialize_openai_client() -> Optional[OpenAI]:
    """
    Initialize and return an OpenAI client using the API key from Streamlit secrets.

    Returns:
        OpenAI client instance if successful, None otherwise.
    """
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            return None
        return OpenAI(api_key=api_key)
    except Exception:
        return None


# ─── Local CSV Data Loading ──────────────────────────────────────────────────
def load_local_csvs(data_dir: str = "./data") -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from the specified local directory.

    Args:
        data_dir (str): Path to the directory containing CSV files.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping filenames to DataFrames.
    """
    csv_files = {}
    data_path = Path(data_dir)

    if not data_path.exists():
        return csv_files

    try:
        csv_paths = list(data_path.glob("*.csv"))
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path)
                csv_files[csv_path.name] = df
            except Exception as e:
                print(f"Error loading {csv_path.name}: {e}")
                continue
    except Exception as e:
        print(f"Error scanning directory {data_dir}: {e}")

    return csv_files


# ─── Google Drive Integration (Placeholder) ──────────────────────────────────
def load_from_google_drive(file_id: str = None) -> Optional[pd.DataFrame]:
    """
    Placeholder function for Google Drive integration.

    To implement:
    1. Install PyDrive: pip install PyDrive2
    2. Set up Google Drive API credentials
    3. Use PyDrive to authenticate and download files

    Example implementation:
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive

        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)

        file = drive.CreateFile({'id': file_id})
        file.GetContentFile('temp.csv')
        df = pd.read_csv('temp.csv')
        return df

    Args:
        file_id (str): Google Drive file ID to load.

    Returns:
        pd.DataFrame or None: Loaded DataFrame if successful.
    """
    # TODO: Implement Google Drive integration
    return None


# ─── Context Building from CSV Data ──────────────────────────────────────────
def build_context_from_csvs(csv_data: Dict[str, pd.DataFrame],
                           max_rows: int = 5) -> str:
    """
    Build a context string from loaded CSV data for the AI model.

    Args:
        csv_data (Dict[str, pd.DataFrame]): Dictionary of CSV files and their data.
        max_rows (int): Maximum number of sample rows to include per file.

    Returns:
        str: Formatted context string summarizing available data.
    """
    if not csv_data:
        return "No local CSV data available."

    context_parts = ["Available defect data from local CSV files:"]

    for filename, df in csv_data.items():
        context_parts.append(f"\n--- {filename} ---")
        context_parts.append(f"Total records: {len(df)}")
        context_parts.append(f"Columns: {', '.join(df.columns.tolist())}")

        # Add sample data
        if not df.empty:
            sample = df.head(max_rows)
            context_parts.append("\nSample data:")
            context_parts.append(sample.to_string(index=False, max_rows=max_rows))

    return "\n".join(context_parts)


# ─── AI Query Function ───────────────────────────────────────────────────────
def query_ai(prompt: str, context: str = "", client: Optional[OpenAI] = None) -> str:
    """
    Query the OpenAI GPT model with user prompt and optional context.

    This function sends a user question along with context (CSV data summaries)
    to GPT-5 and returns a domain-specific answer about defect formation energy
    and semiconductor thermodynamics.

    Args:
        prompt (str): User's question or query.
        context (str): Additional context from CSV files or other data sources.
        client (Optional[OpenAI]): OpenAI client instance.

    Returns:
        str: AI-generated response or error message.
    """
    if client is None:
        return (
            "Error: OpenAI API key not configured. "
            "Please add your API key to .streamlit/secrets.toml as OPENAI_API_KEY."
        )

    if not prompt.strip():
        return "Error: Please provide a question or prompt."

    # Build the system prompt for materials science expert
    system_prompt = """You are an expert materials scientist specializing in semiconductor defect physics,
defect energetics, charge states, and thermodynamics. Your expertise includes:

- Formation energy calculations and their dependence on chemical potentials
- Charge transition levels and their impact on carrier dynamics
- Defect stability in various semiconductor systems (especially Cd-Se-Te alloys)
- Band structure modifications due to defects
- Thermodynamic principles governing defect formation

When analyzing defect data, provide clear, scientifically accurate explanations that:
1. Reference specific energy values and trends from the provided data
2. Discuss the physical mechanisms underlying defect behavior
3. Explain implications for material properties and device performance
4. Use precise terminology without unnecessary LaTeX equations unless explicitly requested

Maintain an academic tone and focus on actionable insights for researchers."""

    # Combine user prompt with context
    if context:
        full_prompt = f"{context}\n\n---\n\nUser Question: {prompt}"
    else:
        full_prompt = prompt

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        error_msg = str(e)

        # Handle common errors gracefully
        if "model_not_found" in error_msg or "does not exist" in error_msg:
            return (
                "Error: The GPT-5 model is not available. "
                "Please check your OpenAI API access or try a different model like 'gpt-4' or 'gpt-4-turbo'."
            )
        elif "authentication" in error_msg.lower():
            return "Error: Authentication failed. Please verify your OpenAI API key."
        elif "rate_limit" in error_msg.lower():
            return "Error: Rate limit exceeded. Please try again in a moment."
        else:
            return f"Error: An unexpected issue occurred while querying the AI model. Details: {error_msg}"


# ─── Utility Functions ────────────────────────────────────────────────────────
def get_csv_summary(csv_data: Dict[str, pd.DataFrame]) -> str:
    """
    Generate a brief summary of loaded CSV files.

    Args:
        csv_data (Dict[str, pd.DataFrame]): Dictionary of CSV data.

    Returns:
        str: Summary text.
    """
    if not csv_data:
        return "No CSV files loaded."

    summaries = []
    for filename, df in csv_data.items():
        summaries.append(f"- {filename}: {len(df)} rows, {len(df.columns)} columns")

    return "\n".join(summaries)


def search_defect_in_data(csv_data: Dict[str, pd.DataFrame],
                         compound: str = None,
                         defect: str = None) -> Optional[pd.DataFrame]:
    """
    Search for specific defect entries across loaded CSV files.

    Args:
        csv_data (Dict[str, pd.DataFrame]): Dictionary of CSV data.
        compound (str): Compound name to filter (e.g., 'CdTe').
        defect (str): Defect type to filter (e.g., 'As_Te').

    Returns:
        pd.DataFrame or None: Filtered data if found.
    """
    all_results = []

    for filename, df in csv_data.items():
        filtered = df.copy()

        if compound and "AB" in df.columns:
            filtered = filtered[filtered["AB"].str.contains(compound, case=False, na=False)]

        if defect and "Defect" in df.columns:
            filtered = filtered[filtered["Defect"].str.contains(defect, case=False, na=False)]

        if not filtered.empty:
            all_results.append(filtered)

    if all_results:
        return pd.concat(all_results, ignore_index=True)

    return None
