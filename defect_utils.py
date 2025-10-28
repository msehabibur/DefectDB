# defect_utils.py
import io
import time
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── Config ───────────────────────────────────────────────────────────────────
ROOT_FOLDER_ID_DEFAULT = "1gYTtFpPIRCDWpLBW855RA6XwG0buifbi"
CSV_FILE_NAME = "cdsete_defect_library_generation_pbesol.csv" # Updated filename

# ── Auth ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def drive_service():
    # Load the info from secrets
    info = dict(st.secrets["gdrive_service_account"])
    
    # --- FIX ---
    # Remove the 'audience' key if it exists.
    # Its presence tricks the factory into creating an ID-token-credential
    # (which requests an 'id_token') instead of an Access-token-credential
    # (which requests an 'access_token').
    info.pop("audience", None)
    # --- END FIX ---

    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https.www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _with_retries(fn, *, tries: int = 3, base_delay: float = 0.8):
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(base_delay * (2 ** k))

# ── Drive helpers ────────────────────────────────────────────────────────────
def list_children(folder_id: str) -> List[Dict]:
    svc = drive_service()
    q = f"'{folder_id}' in parents and trashed = false"
    out, token = [], None
    while True:
        def _do():
            return svc.files().list(
                q=q, spaces="drive",
                fields="nextPageToken, files(id,name,mimeType)",
                pageToken=token, pageSize=1000
            ).execute()
        resp = _with_
