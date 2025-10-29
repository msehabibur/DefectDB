#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_tool.py — AI utilities for DefectDB Studio

Data sources (merged, in this order):
  1) Local CSVs: ./data/*.csv
  2) Google Drive folder (optional, via secrets/ENV)
  3) GitHub CSV (public raw link; override via ENV)

All DataFrames are cached and accessible immediately from app.py.
Every gpt_query() includes a compact context summary of loaded tables.
"""

from __future__ import annotations
import os, io, time
import pandas as pd
import streamlit as st

# OpenAI (o3) — minimal, compatible call
from openai import OpenAI

# Optional Google Drive (lazy import inside functions to avoid hard dep)
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaIoBaseDownload
# from google.oauth2 import service_account


# ────────────────────────── CONFIG ──────────────────────────
DEFAULT_GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/msehabibur/DefectDB/main/cdsete_defect_library_generation_pbesol.csv"
)
GITHUB_CSV_URL = (
    st.secrets.get("GITHUB_CSV_URL")
    or os.getenv("DEFECTDB_GITHUB_CSV_URL")
    or DEFAULT_GITHUB_CSV_URL
)

DEFAULT_DRIVE_FOLDER_ID = (
    st.secrets.get("DRIVE_FOLDER_ID") or os.getenv("GOOGLE_DRIVE_FOLDER_ID") or ""
)

LOCAL_DATA_DIR = os.getenv("DEFECTDB_LOCAL_DATA_DIR", "./data")


# ───────────────────── OpenAI client (o3) ───────────────────
def _init_client() -> OpenAI | None:
    key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("❌ OPENAI_API_KEY missing (set in .streamlit/secrets.toml or env).")
        return None
    return OpenAI(api_key=key)


# ─────────────── Helpers: normalize DataFrames ──────────────
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # strip/uniquify columns to avoid accidental mismatches
    cols = []
    seen = set()
    for c in df.columns:
        base = str(c).strip()
        name = base
        idx = 1
        while name in seen:
            idx += 1
            name = f"{base}.{idx}"
        seen.add(name)
        cols.append(name)
    df = df.copy()
    df.columns = cols
    return df


# ─────────────── Cached loaders: Local / GitHub / Drive ───────────────
@st.cache_data(show_spinner=False)
def _load_local_csvs_cached(path: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        return out
    for fn in os.listdir(path):
        if fn.lower().endswith(".csv"):
            fpath = os.path.join(path, fn)
            try:
                df = pd.read_csv(fpath)
                out[os.path.splitext(fn)[0]] = _normalize_df(df)
            except Exception as e:
                st.warning(f"⚠️ Local CSV skipped: {fn} ({e})")
    return out


@st.cache_data(show_spinner=False)
def _download_github_csv_cached(url: str, retries: int = 2, backoff: float = 0.8) -> dict[str, pd.DataFrame]:
    err = None
    for attempt in range(retries + 1):
        try:
            df = pd.read_csv(url)
            name = os.path.basename(url).removesuffix(".csv")
            return {name: _normalize_df(df)}
        except Exception as e:
            err = e
            time.sleep(backoff * (attempt + 1))
    st.warning(f"⚠️ GitHub CSV download failed: {err}")
    return {}


@st.cache_data(show_spinner=False)
def _download_drive_csvs_cached(folder_id: str) -> dict[str, pd.DataFrame]:
    if not folder_id:
        return {}
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        from google.oauth2 import service_account
    except Exception:
        st.info("ℹ️ Google Drive dependencies not installed; skipping Drive load.")
        return {}

    info = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not info:
        st.info("ℹ️ GOOGLE_SERVICE_ACCOUNT_JSON not set; skipping Drive load.")
        return {}

    try:
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        svc = build("drive", "v3", credentials=creds, cache_discovery=False)
        q = f"'{folder_id}' in parents and mimeType='text/csv' and trashed=false"
        files = svc.files().list(q=q, fields="files(id,name)").execute().get("files", [])
        out: dict[str, pd.DataFrame] = {}
        for f in files:
            bio = io.BytesIO()
            MediaIoBaseDownload(bio, svc.files().get_media(fileId=f["id"])).next_chunk()
            bio.seek(0)
            try:
                df = pd.read_csv(bio)
                out[f["name"].removesuffix(".csv")] = _normalize_df(df)
            except Exception as e:
                st.warning(f"⚠️ Drive CSV skipped: {f['name']} ({e})")
        return out
    except Exception as e:
        st.warning(f"⚠️ Drive load error: {e}")
        return {}


# ─────────────── Global cache (module-level) ────────────────
_ALL_DFS: dict[str, pd.DataFrame] = {}


def _initial_load():
    """Populate the in-memory dataset cache once at import."""
    _ALL_DFS.clear()
    # Local first (developer convenience)
    _ALL_DFS.update(_load_local_csvs_cached(LOCAL_DATA_DIR))
    # Drive (optional)
    if DEFAULT_DRIVE_FOLDER_ID:
        _ALL_DFS.update(_download_drive_csvs_cached(DEFAULT_DRIVE_FOLDER_ID))
    # GitHub dataset (hard requirement for your request)
    _ALL_DFS.update(_download_github_csv_cached(GITHUB_CSV_URL))

    # Friendly status
    if _ALL_DFS:
        st.info("📊 Datasets loaded: " + ", ".join(_ALL_DFS.keys()))
    else:
        st.warning("⚠️ No CSV datasets loaded from local/Drive/GitHub.")


_initial_load()


# ─────────────——— Public helpers you can use in app.py —————————
def list_datasets() -> list[str]:
    """Return loaded dataset names (keys of _ALL_DFS)."""
    return list(_ALL_DFS.keys())


def get_dataframe(name: str) -> pd.DataFrame | None:
    """Return a DataFrame by name (or None if not present)."""
    return _ALL_DFS.get(name)


def refresh_all(drive_folder_id: str | None = None, github_url: str | None = None) -> None:
    """Force reload of all sources; call if user changes Drive folder or GitHub URL."""
    # Clear Streamlit cache + local dict
    _load_local_csvs_cached.clear()
    _download_drive_csvs_cached.clear()
    _download_github_csv_cached.clear()
    _ALL_DFS.clear()

    # Re-run with overrides if supplied
    local = _load_local_csvs_cached(LOCAL_DATA_DIR)
    drive = _download_drive_csvs_cached(drive_folder_id or DEFAULT_DRIVE_FOLDER_ID)
    gh = _download_github_csv_cached(github_url or GITHUB_CSV_URL)
    _ALL_DFS.update({**local, **drive, **gh})
    st.success("🔄 Data sources refreshed: " + ", ".join(_ALL_DFS.keys()))


# ───────────────────── Prompt context + call ─────────────────────
def _context_summary(max_cols: int = 6) -> str:
    if not _ALL_DFS:
        return "No CSV data loaded."
    lines = []
    for k, df in _ALL_DFS.items():
        cols = ", ".join(map(str, df.columns[:max_cols]))
        lines.append(f"{k}: {len(df)} rows, cols [{cols}]")
    return "\n".join(lines)


def gpt_query(prompt: str, model: str = "o3") -> str:
    """
    Query the OpenAI 'o3' model, automatically injecting a compact summary
    of all loaded CSVs so the model can ground its answer.
    """
    client = _init_client()
    if client is None:
        return "❌ OpenAI client not initialised."

    full_prompt = (
        "Context datasets (summarised):\n"
        + _context_summary()
        + "\n\nQuestion for semiconductor-defect expert:\n"
        + (prompt or "Summarise the dataset.")
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a materials-science expert on semiconductor defects. "
                        "Use the tabulated data wherever relevant; prefer precise numbers."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
        )
        # New SDKs sometimes provide output_text; older use choices[0].message.content
        if getattr(resp, "output_text", None):
            text = (resp.output_text or "").strip()
            return text or "⚠️ Model returned empty output_text."
        if getattr(resp, "choices", None) and getattr(resp.choices[0], "message", None):
            text = (resp.choices[0].message.content or "").strip()
            return text or "⚠️ Model returned empty message content."
        return "⚠️ Model returned no text."
    except Exception as e:
        return f"❌ OpenAI error: {e}"
