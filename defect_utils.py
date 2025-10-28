import io, ssl, time, gzip, httplib2, certifi
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
ROOT_FOLDER_ID_DEFAULT = "1gYTtFpPIRCDWpLBW855RA6XwG0buifbi"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

httplib2.CA_CERTS = certifi.where()
ssl.create_default_context(cafile=certifi.where())

# ────────────────────────────────────────────────
# GOOGLE DRIVE SERVICE
# ────────────────────────────────────────────────
def drive_service():
    """Authenticate using Streamlit secrets or service_account.json."""
    import streamlit as st
    try:
        if "gdrive_service_account" in st.secrets:
            creds = service_account.Credentials.from_service_account_info(
                dict(st.secrets["gdrive_service_account"]), scopes=SCOPES
            )
        else:
            creds = service_account.Credentials.from_service_account_file(
                "service_account.json", scopes=SCOPES
            )
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        st.warning(f"⚠️ Google Drive not initialized: {e}")
        return None


# ────────────────────────────────────────────────
# RETRY WRAPPER
# ────────────────────────────────────────────────
def _with_retries(fn, tries: int = 3, base_delay: float = 0.8):
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(base_delay * (2 ** k))


# ────────────────────────────────────────────────
# DRIVE HELPERS
# ────────────────────────────────────────────────
def list_children(folder_id: str) -> List[Dict]:
    svc = drive_service()
    if svc is None:
        return []
    q = f"'{folder_id}' in parents and trashed=false"
    files, token = [], None
    while True:
        def _do():
            return svc.files().list(
                q=q,
                spaces="drive",
                fields="nextPageToken, files(id,name,mimeType,modifiedTime,size)",
                pageToken=token,
                pageSize=1000,
            ).execute()
        resp = _with_retries(_do)
        files.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return files


def download_bytes(file_id: str) -> bytes:
    svc = drive_service()
    if svc is None:
        return b""
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


# ────────────────────────────────────────────────
# DISCOVERY HELPERS (RETURN DICTS)
# ────────────────────────────────────────────────
def discover_compounds(root_folder_id: Optional[str] = None) -> Dict[str, str]:
    root_id = root_folder_id or ROOT_FOLDER_ID_DEFAULT
    return {
        f["name"]: f["id"]
        for f in list_children(root_id)
        if f["mimeType"] == "application/vnd.google-apps.folder"
    }


def discover_defects(compound_folder_id: str) -> Dict[str, str]:
    return {
        f["name"]: f["id"]
        for f in list_children(compound_folder_id)
        if f["mimeType"] == "application/vnd.google-apps.folder" and f["name"].lower() != "bulk"
    }


def discover_charge_states(defect_folder_id: str) -> Dict[str, str]:
    return {
        f["name"]: f["id"]
        for f in list_children(defect_folder_id)
        if f["mimeType"] == "application/vnd.google-apps.folder"
    }


def find_bulk_folder(compound_folder_id: str) -> Optional[str]:
    for f in list_children(compound_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder" and f["name"].lower() == "bulk":
            return f["id"]
    return None


# ────────────────────────────────────────────────
# CHARGE PARSER
# ────────────────────────────────────────────────
def parse_charge_label_to_q(label: str) -> Optional[int]:
    s = (label or "").strip().lower()
    if s in {"neutral", "neut"}:
        return 0
    if s.startswith("charged+"):
        return int(s.replace("charged+", ""))
    if s.startswith("charged-"):
        return -int(s.replace("charged-", ""))
    if s.startswith("q+"):
        return int(s.replace("q+", ""))
    if s.startswith("q-"):
        return -int(s.replace("q-", ""))
    if s in {"0", "q0", "neutral"}:
        return 0
    if s.startswith("m"):
        return -int(s[1:])
    if s.startswith("p"):
        return int(s[1:])
    # fallback: extract integer
    import re
    m = re.search(r"([+\-]?\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None


# ────────────────────────────────────────────────
# STRUCTURE FILES
# ────────────────────────────────────────────────
STRUCTURE_FILE_PRIORITY = [
    "CONTCAR", "POSCAR", "Relaxed.cif", "Final.cif", "structure.cif", "optimized.cif"
]

def find_structure_file(folder_id: str):
    files = list_children(folder_id)
    byname = {f["name"].lower(): f for f in files}
    for candidate in STRUCTURE_FILE_PRIORITY:
        if candidate.lower() in byname:
            f = byname[candidate.lower()]
            data = download_bytes(f["id"])
            return data, f["name"]
    for f in files:
        if f["name"].lower().endswith(".cif"):
            data = download_bytes(f["id"])
            return data, f["name"]
    return None, ""
