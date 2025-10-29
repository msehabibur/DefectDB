#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_tool.py: Centralized AI utility module for DefectDB Studio.
Adds Google Drive access + CSV downloading + GPT integration (compatible with GPT-5 API).
"""

import os
import io
import pandas as pd
import streamlit as st
from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account


# ─── GPT Client Initialization ────────────────────────────────────────────────
def _init_client():
    """
    Initialize OpenAI client from Streamlit secrets or environment variables.
    """
    api_key = None
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.warning("⚠️ Please add your OpenAI API key to `.streamlit/secrets.toml` or as an environment variable.")
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize OpenAI client: {e}")
        return None


# ─── Google Drive Service Initialization ──────────────────────────────────────
def _init_drive_service():
    """
    Initialize Google Drive service using credentials stored in Streamlit secrets.
    Expect a `GOOGLE_SERVICE_ACCOUNT_JSON` entry in secrets.toml.
    """
    try:
        creds_dict = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
    except KeyError:
        st.warning("⚠️ Google Drive access not configured. Add service account JSON to secrets.toml.")
        return None

    try:
        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        service = build("drive", "v3", credentials=creds)
        return service
    except Exception as e:
        st.error(f"❌ Failed to initialize Google Drive API: {e}")
        return None


# ─── Download CSV Files from Google Drive ─────────────────────────────────────
def load_google_drive_csvs(folder_id: str) -> dict:
    """
    Scan a Google Drive folder for CSV files and download them as DataFrames.

    Args:
        folder_id (str): Google Drive folder ID.

    Returns:
        dict[str, pd.DataFrame]: {filename: DataFrame}
    """
    service = _init_drive_service()
    if service is None:
        return {}

    csvs = {}

    try:
        results = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and mimeType='text/csv' and trashed=false",
                fields="files(id, name)"
            )
            .execute()
        )
        files = results.get("files", [])

        if not files:
            st.info("ℹ️ No CSV files found in the specified Google Drive folder.")
            return {}

        for file in files:
            file_id = file["id"]
            file_name = file["name"]
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()

            fh.seek(0)
            try:
                df = pd.read_csv(fh)
                csvs[file_name.replace(".csv", "")] = df
                st.success(f"✅ Downloaded {file_name} ({len(df)} rows)")
            except Exception as e:
                st.warning(f"⚠️ Could not read {file_name}: {e}")

    except Exception as e:
        st.error(f"❌ Error reading from Google Drive: {e}")

    return csvs


# ─── Load Local CSVs ──────────────────────────────────────────────────────────
def load_local_csvs(folder: str = "./data") -> dict:
    """
    Load all CSV files from a local folder.
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


# ─── Build Context for GPT ────────────────────────────────────────────────────
def build_context_from_dataframes(dataframes: dict) -> str:
    """
    Create a short summary string from DataFrames to help GPT understand context.
    """
    if not dataframes:
        return "No CSV datasets loaded."
    summaries = []
    for name, df in dataframes.items():
        cols = ", ".join(df.columns[:6])
        summaries.append(f"{name}: {len(df)} rows, columns [{cols}]")
    return "\n".join(summaries)


# ─── GPT Query ────────────────────────────────────────────────────────────────
def gpt_query(prompt: str, model: str = "gpt-5", context_data: dict = None) -> str:
    """
    Query GPT with optional context built from Drive/Local CSVs.
    Compatible with new GPT-5 API (no unsupported params).
    """
    client = _init_client()
    if client is None:
        return "❌ Error: No valid OpenAI API key configured."

    # Build textual context from any loaded DataFrames
    context_text = build_context_from_dataframes(context_data or {})
    full_prompt = f"Context:\n{context_text}\n\nUser Question:\n{prompt}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a materials science expert specializing in semiconductor defects, "
                        "formation energies, charge states, and thermodynamics. "
                        "You have access to the user's DFT dataset and Drive CSV files. "
                        "Use that data to generate accurate, scientific explanations."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
            max_completion_tokens=800,  # ✅ GPT-5 compatible
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error from GPT API: {e}"
