#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_tool.py — AI utilities for DefectDB Studio
Uses the OpenAI “o3” reasoning model (no temperature / max_tokens overrides).
Also keeps optional helpers for local & Drive CSVs, but these are unchanged.
"""

import os
import io
import pandas as pd
import streamlit as st
from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account


# ───────────────────────────  OPENAI  ─────────────────────────────────────────
def _init_client():
    key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("❌ OPENAI_API_KEY missing in secrets.toml or environment.")
        return None
    return OpenAI(api_key=key)


# ───────────────────────────  GOOGLE DRIVE  ───────────────────────────────────
def _init_drive_service():
    try:
        creds_info = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
    except KeyError:
        return None  # Drive access is optional
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)


def load_google_drive_csvs(folder_id: str) -> dict:
    svc = _init_drive_service()
    if svc is None:
        return {}
    csvs, res = {}, svc.files().list(
        q=f"'{folder_id}' in parents and mimeType='text/csv' and trashed=false",
        fields="files(id, name)"
    ).execute()
    for f in res.get("files", []):
        bio = io.BytesIO()
        MediaIoBaseDownload(bio, svc.files().get_media(fileId=f["id"])).next_chunk()
        bio.seek(0)
        try:
            df = pd.read_csv(bio)
            csvs[f["name"].removesuffix(".csv")] = df
        except Exception as e:
            st.warning(f"Couldn’t read {f['name']}: {e}")
    return csvs


# ─────────────────────────────  LOCAL CSV  ────────────────────────────────────
def load_local_csvs(folder: str = "./data") -> dict:
    path, csvs = os.path.expanduser(folder), {}
    if not os.path.isdir(path):
        return csvs
    for fn in os.listdir(path):
        if fn.endswith(".csv"):
            try:
                csvs[fn.removesuffix(".csv")] = pd.read_csv(os.path.join(path, fn))
            except Exception as e:
                st.warning(f"Couldn’t read {fn}: {e}")
    return csvs


# ────────────────────────  GPT PROMPT & CALL  ────────────────────────────────
def _summarise(dataframes: dict) -> str:
    if not dataframes:
        return "No CSV data provided."
    lines = [
        f"{name}: {len(df)} rows, columns [{', '.join(df.columns[:6])}]"
        for name, df in dataframes.items()
    ]
    return "\n".join(lines)


def gpt_query(prompt: str, model: str = "o3", context_data: dict | None = None) -> str:
    client = _init_client()
    if client is None:
        return "❌ OpenAI client not initialised."

    full_prompt = (
        "Context:\n" + _summarise(context_data or {}) +
        "\n\nQuestion for materials-science defect expert:\n" + prompt
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a materials-science expert on semiconductor defects. "
                        "Answer clearly and accurately using any supplied data."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
        )
        # New SDKs sometimes expose `output_text`; older ones use choices→message→content
        if getattr(resp, "output_text", None):
            return resp.output_text.strip()
        if resp.choices and getattr(resp.choices[0], "message", None):
            return resp.choices[0].message.content.strip()
        return "⚠️ Model returned no text."
    except Exception as e:
        return f"❌ OpenAI error: {e}"
