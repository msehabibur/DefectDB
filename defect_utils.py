"""Utility functions for interacting with Google Drive and defect data."""
from __future__ import annotations

import io
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_FOLDER_ID_DEFAULT = "1gYTtFpPIRCDWpLBW855RA6XwG0buifbi"
CSV_FILE_NAME = "cdsete_defect_library_generation_pbesol.csv"
STRUCTURE_FILE_PRIORITY = (
    "CONTCAR",
    "CONTCAR.gz",
    "CONTCAR.cif",
    "POSCAR",
    "POSCAR.gz",
    "POSCAR.cif",
    "structure.cif",
)


# ── Auth ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def drive_service():
    """Return an authenticated Google Drive service client."""
    info = dict(st.secrets["gdrive_service_account"])
    info.pop("audience", None)

    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _with_retries(fn, *, tries: int = 3, base_delay: float = 0.8):
    for attempt in range(tries):
        try:
            return fn()
        except Exception:  # pragma: no cover - depends on network I/O
            if attempt == tries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


# ── Drive helpers ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def list_children(folder_id: str) -> List[Dict]:
    """Return metadata for all children of a Google Drive folder."""
    svc = drive_service()
    query = f"'{folder_id}' in parents and trashed = false"
    out: List[Dict] = []
    token: Optional[str] = None

    while True:
        def _do():
            return (
                svc.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id,name,mimeType)",
                    pageToken=token,
                    pageSize=1000,
                )
                .execute()
            )

        resp = _with_retries(_do)
        out.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return out


def download_bytes(file_id: str) -> bytes:
    """Download the bytes for a file from Google Drive."""
    svc = drive_service()
    request = svc.files().get_media(fileId=file_id)
    file_handle = io.BytesIO()
    downloader = MediaIoBaseDownload(file_handle, request)
    done = False

    while not done:
        def _do():
            return downloader.next_chunk()

        _, done = _with_retries(_do)
    return file_handle.getvalue()


def find_file_in_folder_by_name(folder_id: str, filename: str) -> Optional[Dict]:
    """Return metadata for a non-folder file whose name matches ``filename``."""
    expected_name = filename.lower()
    for entry in list_children(folder_id):
        if entry["mimeType"] == "application/vnd.google-apps.folder":
            continue
        if entry["name"].lower() == expected_name:
            return entry
    return None


# ── Main Data Loading Function ────────────────────────────────────────────────
def load_csv_data(root_folder_id: str) -> Optional[pd.DataFrame]:
    """Find and load the defect CSV file from the given Drive root folder."""
    meta = find_file_in_folder_by_name(root_folder_id, CSV_FILE_NAME)
    if not meta:
        return None

    try:
        raw_bytes = download_bytes(meta["id"])
        return pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:  # pragma: no cover - user-facing error message
        st.error(f"Failed to read CSV file: {exc}")
        return None


# ── Discovery helpers for structures ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def discover_compounds(root_folder_id: str) -> Dict[str, str]:
    """Return mapping of compound folder names to IDs under the root folder."""
    compounds: Dict[str, str] = {}
    for entry in list_children(root_folder_id):
        if entry["mimeType"] == "application/vnd.google-apps.folder":
            compounds[entry["name"]] = entry["id"]
    return compounds


@st.cache_data(show_spinner=False)
def discover_defects(compound_folder_id: str) -> Dict[str, str]:
    """Return mapping of defect folder names to IDs within a compound folder."""
    defects: Dict[str, str] = {}
    for entry in list_children(compound_folder_id):
        if entry["mimeType"] == "application/vnd.google-apps.folder":
            defects[entry["name"]] = entry["id"]
    return defects


@st.cache_data(show_spinner=False)
def discover_charge_states(defect_folder_id: str) -> Dict[str, str]:
    """Return mapping of charge-state folder names to IDs within a defect folder."""
    charges: Dict[str, str] = {}
    for entry in list_children(defect_folder_id):
        if entry["mimeType"] == "application/vnd.google-apps.folder":
            charges[entry["name"]] = entry["id"]
    return charges


def parse_charge_label_to_q(label: str) -> Optional[int]:
    """Extract an integer charge state from a folder label if possible."""
    if not label:
        return None

    match = re.search(r"([+-]?\d+)", label)
    if not match:
        return None

    try:
        return int(match.group(1))
    except ValueError:
        return None


def find_structure_file(charge_folder_id: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Return the bytes and name of the highest priority structure file in a folder."""
    children = list_children(charge_folder_id)
    files_by_name = {child["name"]: child for child in children if child["mimeType"] != "application/vnd.google-apps.folder"}

    for candidate in STRUCTURE_FILE_PRIORITY:
        for name, meta in files_by_name.items():
            if name.lower() == candidate.lower():
                return download_bytes(meta["id"]), name

    if files_by_name:
        # Fall back to the first file alphabetically if nothing matches the priority list
        fallback_name = sorted(files_by_name.keys())[0]
        meta = files_by_name[fallback_name]
        return download_bytes(meta["id"]), fallback_name

    return None, None
