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
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gdrive_service_account"]),
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
        resp = _with_retries(_do)
        out.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return out

def download_bytes(file_id: str) -> bytes:
    svc = drive_service()
    req = svc.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        def _do():
            return downloader.next_chunk()
        _, done = _with_retries(_do)
    return fh.getvalue()

def find_file_in_folder_by_name(folder_id: str, filename: str) -> Optional[Dict]:
    flc = filename.lower()
    for f in list_children(folder_id):
        if f["mimeType"] != "application/vnd.google-apps.folder" and f["name"].lower() == flc:
            return f
    return None

# ── Main Data Loading Function ───────────────────────────────────────────────
def load_csv_data(root_folder_id: str) -> Optional[pd.DataFrame]: # Renamed function
    """
    Finds and loads the specific defect CSV file from the given root folder.
    """
    meta = find_file_in_folder_by_name(root_folder_id, CSV_FILE_NAME) # Use CSV_FILE_NAME
    if not meta:
        return None
    
    try:
        raw = download_bytes(meta["id"])
        df = pd.read_csv(io.BytesIO(raw)) # Use pd.read_csv
        return df
    except Exception as e:
        st.error(f"Failed to read CSV file: {e}")
        return None
