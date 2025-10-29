#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_tool.py: Centralized AI utility module for DefectDB Studio.
Adds Google Drive access + CSV downloading + GPT integration (GPT-5 compatible).
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
    """Initialize OpenAI client from secrets or environment variables."""
    api_key = None
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.warning("⚠️ Please add your OpenAI API key to `.streamlit/secrets.toml` or environment.")
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize OpenAI client: {e}")
        return None


# ─── Google Drive Service Initialization ──────────────────────────────────────
def _init_drive_service():
    """Initialize Google Drive API service using credentials in secrets.toml."""
    try:
        creds_dict = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
    except KeyError:
        st.warning("⚠️ Google Drive access not configured. Add service account JSON to secrets.toml.")
        return None

    try:
        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        st.error(f"❌ Failed to initialize Google Drive API: {e}")
        return None


# ─── Download CSV Files from Google Drive ─────────────────────────────────────
def load_google_drive_csvs(folder_id: str) -> dict:
    """Download all CSV files in a Drive folder into pandas DataFrames."""
    service = _init_drive_service()
    if service is None:
        return {}

    csvs = {}
    try:
        results = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and mimeType='text/csv' and trashed=false",
                fields="files(id, name)",
            )
            .execute()
        )
        files = results.get("files", [])
        if not files:
            st.info("ℹ️ No CSV files found in this Google Drive folder.")
            return {}

        for f in files:
            fid, name = f["id"], f["name"]
            request = service.files().get_media(fileId=fid)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            try:
                df = pd.read_csv(fh)
                csvs[name.replace(".csv", "")] = df
                st.success(f"✅ Downloaded {name} ({len(df)} rows)")
            except Exception as e:
                st.warning(f"⚠️ Could not read {name}: {e}")
    except Exception as e:
        st.error(f"❌ Google Drive error: {e}")

    return csvs


# ─── Load Local CSVs ──────────────────────────────────────────────────────────
def load_local_csvs(folder: str = "./data") -> dict:
    """Load all CSV files from a local folder."""
    csvs = {}
    path = os.path.expanduser(folder)
    if not os.path.exists(path):
        st.info(f"ℹ️ No local data folder found at {path}")
        return csvs

    for file in os.listdir(path):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(path, file))
                csvs[file.replace(".csv", "")] = df
            except Exception as e:
                st.warning(f"⚠️ Could not load {file}: {e}")
    return csvs


# ─── Build Context from CSVs ──────────────────────────────────────────────────
def build_context_from_dataframes(dataframes: dict) -> str:
    """Summarize loaded CSVs to provide GPT context."""
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
    Query GPT-5 with optional context from Drive/Local CSVs.
    Handles new SDK response format (output_text).
    """
    client = _init_client()
    if client is None:
        return "❌ Error: No valid OpenAI API key configured."

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
            max_completion_tokens=800,
        )

        # ✅ Handle both new & legacy response structures
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()
        elif response.choices and hasattr(response.choices[0], "message"):
            return response.choices[0].message.content.strip()
        else:
            return "⚠️ No text output returned by the AI model."

    except Exception as e:
        return f"❌ Error from GPT API: {e}"
