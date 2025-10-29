#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_tool.py: Centralized AI utility module for DefectDB Studio.
Handles GPT model setup, querying, and optional data context from CSVs or Google Drive.
"""

import os
import pandas as pd
from openai import OpenAI
import streamlit as st

# ─── GPT Client Initialization ────────────────────────────────────────────────
def _init_client():
    """
    Initialize OpenAI client from Streamlit secrets or environment.
    Returns:
        OpenAI: initialized client or None if API key missing.
    """
    api_key = None
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.warning("⚠️ Please add your OpenAI API key to `.streamlit/secrets.toml` or environment variables.")
        return None

    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize OpenAI client: {e}")
        return None


# ─── Query GPT Model ──────────────────────────────────────────────────────────
def gpt_query(prompt: str, model: str = "gpt-5") -> str:
    """
    Send a prompt to GPT and return the generated text.

    Args:
        prompt (str): user prompt
        model (str): model name (default: gpt-5)
    Returns:
        str: GPT model output or error message
    """
    client = _init_client()
    if client is None:
        return "❌ Error: No valid OpenAI API key configured."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a materials science expert specializing in semiconductor defects, "
                        "formation energies, charge states, and thermodynamics. Provide concise, "
                        "scientific, data-driven explanations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error from GPT API: {e}"


# ─── Load Local CSV Files ─────────────────────────────────────────────────────
def load_local_csvs(folder: str = "./data"):
    """
    Load all CSV files from a local folder into a dict of DataFrames.
    Args:
        folder (str): path to folder containing CSVs
    Returns:
        dict[str, pd.DataFrame]: {filename_stem: DataFrame}
    """
    csvs = {}
    folder_path = os.path.expanduser(folder)
    if not os.path.exists(folder_path):
        st.info(f"ℹ️ No local data folder found at {folder_path}")
        return csvs

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(folder_path, file))
                csvs[file.replace(".csv", "")] = df
            except Exception as e:
                st.warning(f"⚠️ Could not load {file}: {e}")
    return csvs


# ─── Optional Google Drive Integration Stub ───────────────────────────────────
def load_google_drive_csvs(service=None, folder_id=None):
    """
    Placeholder for Google Drive CSV loading.
    Replace with actual implementation using PyDrive or googleapiclient if desired.
    """
    st.info("🔗 Google Drive integration not configured yet.")
    return None


# ─── Build Context from CSVs ──────────────────────────────────────────────────
def build_context_from_dataframes(dataframes: dict) -> str:
    """
    Construct a short textual context summary from loaded CSVs for GPT prompt.
    Args:
        dataframes (dict): dict of pandas DataFrames
    Returns:
        str: summarized context string
    """
    if not dataframes:
        return "No CSV datasets loaded."
    summaries = []
    for name, df in dataframes.items():
        cols = ", ".join(df.columns[:6])  # preview first few columns
        summaries.append(f"{name}: {len(df)} rows, columns [{cols}]")
    return "\n".join(summaries)
