#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_tool.py  —  AI utilities for DefectDB Studio
• Reads CSVs from:
  1. ./data/            (local)
  2. Google-Drive folder (optional)
  3. A fixed GitHub URL  (DefectDB public repo)
• Caches all DataFrames in _ALL_DFS and injects a summary into every prompt.
• Uses the OpenAI 'o3' model (no extra params).
"""

import os, io, pandas as pd, streamlit as st, requests
from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account


# ───────────────────────────  CONSTANTS  ──────────────────────────────────────
GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "msehabibur/DefectDB/main/cdsete_defect_library_generation_pbesol.csv"
)

# ───────────────────────────  OPENAI  ─────────────────────────────────────────
def _init_client():
    key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("❌ OPENAI_API_KEY missing.")
        return None
    return OpenAI(api_key=key)


# ──────────────────────────  GOOGLE DRIVE  ────────────────────────────────────
def _init_drive_service():
    info = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not info:
        return None
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _download_drive_csvs(folder_id: str) -> dict:
    svc = _init_drive_service()
    if svc is None or not folder_id:
        return {}
    csvs, q = {}, f"'{folder_id}' in parents and mimeType='text/csv' and trashed=false"
    for f in svc.files().list(q=q, fields="files(id, name)").execute().get("files", []):
        bio = io.BytesIO()
        MediaIoBaseDownload(bio, svc.files().get_media(fileId=f["id"])).next_chunk()
        bio.seek(0)
        try:
            csvs[f["name"].removesuffix('.csv')] = pd.read_csv(bio)
        except Exception as e:
            st.warning(f"Drive CSV {f['name']} skipped ({e})")
    return csvs


# ────────────────────────────  LOCAL CSV  ─────────────────────────────────────
def _load_local_csvs(path: str = "./data") -> dict:
    csvs, path = {}, os.path.expanduser(path)
    if not os.path.isdir(path):
        return csvs
    for fn in os.listdir(path):
        if fn.endswith(".csv"):
            try:
                csvs[fn.removesuffix(".csv")] = pd.read_csv(os.path.join(path, fn))
            except Exception as e:
                st.warning(f"Local CSV {fn} skipped ({e})")
    return csvs


# ────────────────────────────  GITHUB CSV  ────────────────────────────────────
def _download_github_csv(url: str) -> dict:
    try:
        df = pd.read_csv(url)
        name = os.path.basename(url).removesuffix(".csv")
        return {name: df}
    except Exception as e:
        st.warning(f"GitHub CSV download failed ({e})")
        return {}


# ────────────────────  GLOBAL DATAFRAME CACHE  ───────────────────────────────
_ALL_DFS: dict[str, pd.DataFrame] = {}


def _initial_load():
    """Populate the cache exactly once at import."""
    _ALL_DFS.update(_load_local_csvs("./data"))

    drive_id = st.secrets.get("DRIVE_FOLDER_ID") or os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if drive_id:
        _ALL_DFS.update(_download_drive_csvs(drive_id))

    _ALL_DFS.update(_download_github_csv(GITHUB_CSV_URL))

    if _ALL_DFS:
        st.info("📊 CSVs loaded: " + ", ".join(_ALL_DFS.keys()))
    else:
        st.warning("⚠️  No CSV datasets loaded.")


_initial_load()


# ────────────────────────────  GPT CALL  ──────────────────────────────────────
def _context_summary() -> str:
    if not _ALL_DFS:
        return "No CSV data loaded."
    return "\n".join(
        f"{k}: {len(df)} rows, cols [{', '.join(df.columns[:6])}]"
        for k, df in _ALL_DFS.items()
    )


def gpt_query(prompt: str, model: str = "o3") -> str:
    client = _init_client()
    if client is None:
        return "❌ OpenAI client not initialised."

    full_prompt = (
        "Context datasets:\n" + _context_summary() +
        "\n\nQuestion for semiconductor-defect expert:\n" + prompt
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are a materials-science expert. Use the tabulated data where relevant."},
                {"role": "user", "content": full_prompt},
            ],
        )
        if getattr(resp, "output_text", None):
            return resp.output_text.strip()
        if resp.choices and getattr(resp.choices[0], "message", None):
            return resp.choices[0].message.content.strip()
        return "⚠️ Model returned no text."
    except Exception as e:
        return f"❌ OpenAI error: {e}"
