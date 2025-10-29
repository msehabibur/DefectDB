#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_tool.py  —  self-contained AI utilities for DefectDB Studio

• Auto-loads CSV files from ./data AND from a Google-Drive folder specified in
  `st.secrets["DRIVE_FOLDER_ID"]` or the env-var GOOGLE_DRIVE_FOLDER_ID.
• Caches all DataFrames in _ALL_DFS.
• Every gpt_query() call automatically injects a concise summary of those
  DataFrames into the prompt (no manual context passing needed).
"""

import os, io, pandas as pd, streamlit as st
from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account


# ───────────────────────────  OPENAI  ─────────────────────────────────────────
def _init_client():
    key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("❌ OPENAI_API_KEY missing (env or secrets.toml).")
        return None
    return OpenAI(api_key=key)


# ──────────────────────────  GOOGLE DRIVE  ────────────────────────────────────
def _init_drive_service():
    info = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if info is None:
        return None  # Drive access optional
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _download_drive_csvs(folder_id: str) -> dict:
    """Return {name: DataFrame} for every CSV in a Drive folder."""
    svc = _init_drive_service()
    if svc is None or not folder_id:
        return {}
    csvs, query = {}, f"'{folder_id}' in parents and mimeType='text/csv' and trashed=false"
    for f in svc.files().list(q=query, fields="files(id, name)").execute().get("files", []):
        fh = io.BytesIO()
        MediaIoBaseDownload(fh, svc.files().get_media(fileId=f["id"])).next_chunk()
        fh.seek(0)
        try:
            csvs[f["name"].removesuffix(".csv")] = pd.read_csv(fh)
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


# ─────────────────────  GLOBAL DATAFRAME CACHE  ───────────────────────────────
_ALL_DFS: dict[str, pd.DataFrame] = {}     # gets populated at import-time


def _initial_data_load():
    """Populate _ALL_DFS once when the module is imported."""
    local = _load_local_csvs("./data")
    drive_id = st.secrets.get("DRIVE_FOLDER_ID") or os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    drive = _download_drive_csvs(drive_id) if drive_id else {}
    _ALL_DFS.update({**local, **drive})
    if _ALL_DFS:
        st.info(f"📊 CSVs in memory: {list(_ALL_DFS.keys())}")


_initial_data_load()


def refresh_drive_data(folder_id: str):
    """
    Reload CSVs from a new Drive folder and replace the cache.
    Call this once if the user supplies a different folder ID at runtime.
    """
    _ALL_DFS.clear()
    _ALL_DFS.update(_load_local_csvs("./data"))
    _ALL_DFS.update(_download_drive_csvs(folder_id))
    st.success(f"🔄 Drive data refreshed. CSVs now loaded: {list(_ALL_DFS.keys())}")


# ────────────────────  GPT PROMPT & CALL  (model=o3)  ─────────────────────────
def _context_summary() -> str:
    if not _ALL_DFS:
        return "No CSV datasets loaded."
    return "\n".join(
        f"{k}: {len(df)} rows, cols: {', '.join(df.columns[:6])}"
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
                {
                    "role": "system",
                    "content": (
                        "You are a materials-science expert specialising in semiconductor "
                        "defects. Use the tabulated data above wherever relevant."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
        )
        # Works for new & legacy SDK formats
        if getattr(resp, "output_text", None):
            return resp.output_text.strip()
        if resp.choices and getattr(resp.choices[0], "message", None):
            return resp.choices[0].message.content.strip()
        return "⚠️ Model returned no text."
    except Exception as e:
        return f"❌ OpenAI error: {e}"
