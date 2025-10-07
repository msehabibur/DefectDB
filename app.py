# app.py
# Database viewer for DefectDB on Google Drive
# - Lists compounds (folders) under a root Drive folder
# - For a compound: lists defect folders
# - For a defect: finds charge-state subfolders and parses total energy
# - Shows Bulk/ total energy if present
#
# Requirements:
#   streamlit
#   google-api-python-client
#   google-auth
#   google-auth-httplib2
#   pandas
#   numpy
#   pymatgen
#
# Secrets (.streamlit/secrets.toml) must contain [gdrive_service_account] as you set up.

import io
import gzip
import re
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# ---------- Config ----------
st.set_page_config(page_title="DefectDB Browser (Drive)", layout="wide")
ROOT_FOLDER_ID_DEFAULT = "1gYTtFpPIRCDWpLBW855RA6XwG0buifbi"  # <- your shared folder

# ---------- Auth & Drive client ----------
@st.cache_resource(show_spinner=False)
def drive_service():
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gdrive_service_account"]),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

# ---------- Drive helpers ----------
def list_children(folder_id: str) -> List[Dict]:
    svc = drive_service()
    q = f"'{folder_id}' in parents and trashed = false"
    out = []
    token = None
    while True:
        resp = svc.files().list(
            q=q,
            spaces="drive",
            fields="nextPageToken, files(id,name,mimeType,modifiedTime,size)",
            pageToken=token,
            pageSize=1000,
        ).execute()
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
        _, done = downloader.next_chunk()
    return fh.getvalue()

# ---------- Parsing helpers ----------
def maybe_gunzip(name: str, data: bytes) -> bytes:
    return gzip.decompress(data) if name.lower().endswith(".gz") else data

def parse_outcar_energy(raw: bytes) -> Optional[float]:
    """Try pymatgen; fallback to regex on TOTEN / energy without entropy."""
    try:
        from pymatgen.io.vasp.outputs import Outcar
        with tempfile.NamedTemporaryFile(delete=True, suffix=".OUTCAR") as tmp:
            tmp.write(raw); tmp.flush()
            out = Outcar(tmp.name)
            if getattr(out, "final_energy", None) is not None:
                return float(out.final_energy)
    except Exception:
        pass
    text = raw.decode(errors="ignore")
    m = None
    for pat in [
        r"free\s+energy\s+TOTEN\s*=\s*([-\d\.Ee+]+)",
        r"energy\s+without\s+entropy\s*=\s*([-\d\.Ee+]+)",
    ]:
        hits = list(re.finditer(pat, text))
        if hits:
            m = hits[-1]
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def parse_vasprun_energy(raw: bytes) -> Optional[float]:
    try:
        from pymatgen.io.vasp.outputs import Vasprun
        with tempfile.NamedTemporaryFile(delete=True, suffix=".xml") as tmp:
            tmp.write(raw); tmp.flush()
            vr = Vasprun(tmp.name, parse_dos=False, parse_eigen=False)
            if getattr(vr, "final_energy", None) is not None:
                return float(vr.final_energy)
    except Exception:
        return None
    return None

def parse_oszicar_energy(raw: bytes) -> Optional[float]:
    text = raw.decode(errors="ignore")
    for line in reversed([l for l in text.splitlines() if l.strip()]):
        m = re.search(r"E0\s*=\s*([-\d\.Ee+]+)", line)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None

def parse_total_energy_for_folder(folder_id: str) -> Tuple[Optional[float], str]:
    """
    Look for OUTCAR(.gz) first, then vasprun.xml(.gz), then OSZICAR(.gz) in a folder.
    Return (energy, source_label).
    """
    kids = list_children(folder_id)
    # Prefer OUTCAR
    cand = {k["name"].lower(): k for k in kids}
    def try_file(name_options, parser):
        for nm in name_options:
            if nm in cand:
                raw = download_bytes(cand[nm]["id"])
                raw = maybe_gunzip(nm, raw)
                e = parser(raw)
                if e is not None:
                    return e, nm
        return None, ""
    # Try OUTCAR
    e, src = try_file(["outcar.gz", "outcar"], parse_outcar_energy)
    if e is not None: return e, src.upper()
    # Try vasprun.xml
    e, src = try_file(["vasprun.xml.gz", "vasprun.xml"], parse_vasprun_energy)
    if e is not None: return e, src
    # Try OSZICAR
    e, src = try_file(["oszicar.gz", "oszicar"], parse_oszicar_energy)
    if e is not None: return e, src.upper()
    return None, "not_found"

# ---------- Discovery logic ----------
def discover_compounds(root_folder_id: str) -> Dict[str, str]:
    """Return {compound_name: folder_id} for all immediate subfolders."""
    m = {}
    for f in list_children(root_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder":
            m[f["name"]] = f["id"]
    return dict(sorted(m.items(), key=lambda x: x[0].lower()))

def discover_defects(compound_folder_id: str) -> Dict[str, str]:
    """Return {defect_name: folder_id} for defect subfolders (excluding Bulk)."""
    m = {}
    for f in list_children(compound_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder":
            if f["name"].lower() != "bulk":
                m[f["name"]] = f["id"]
    return dict(sorted(m.items(), key=lambda x: x[0].lower()))

def discover_charge_states(defect_folder_id: str) -> Dict[str, str]:
    """Return {charge_label: folder_id} for subfolders (q+2, q0, q-1, ...)"""
    m = {}
    for f in list_children(defect_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder":
            m[f["name"]] = f["id"]
    # Sort by integer charge if possible: q+2 > q+1 > q0 > q-1 ...
    def key_fn(lbl):
        s = lbl.lower().replace("q", "")
        try:
            return int(s)
        except Exception:
            return -999999
    return dict(sorted(m.items(), key=lambda x: key_fn(x[0]), reverse=True))

def find_bulk_folder(compound_folder_id: str) -> Optional[str]:
    for f in list_children(compound_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder" and f["name"].lower() == "bulk":
            return f["id"]
    return None

# ---------- UI ----------
st.title("🧪 DefectDB Browser")

with st.sidebar:
    st.header("Data Source")
    root_id = st.text_input("Root Folder ID", value=ROOT_FOLDER_ID_DEFAULT)
    refresh = st.button("Scan")

if refresh:
    try:
        # 1) Compounds
        compounds = discover_compounds(root_id)
        if not compounds:
            st.error("No compound folders found in this root. Make sure the service account has Viewer access.")
            st.stop()

        # Overview table
        overview_rows = []
        for comp, comp_id in compounds.items():
            bulk_id = find_bulk_folder(comp_id)
            overview_rows.append({"Compound": comp, "Has Bulk folder": "Yes" if bulk_id else "No"})
        st.subheader("📦 Compounds")
        st.dataframe(pd.DataFrame(overview_rows), use_container_width=True)

        # 2) Select compound
        comp_sel = st.selectbox("Select a compound", list(compounds.keys()))
        comp_id = compounds[comp_sel]

        # 3) Bulk energy (if present)
        st.markdown("### 🧱 Bulk Energy")
        bulk_id = find_bulk_folder(comp_id)
        if bulk_id:
            if st.button("Read Bulk Total Energy", key=f"bulk_{comp_sel}"):
                try:
                    ebulk, src = parse_total_energy_for_folder(bulk_id)
                    if ebulk is None:
                        st.error("Could not parse bulk energy from OUTCAR/vasprun.xml/OSZICAR.")
                    else:
                        st.success(f"Bulk total energy: **{ebulk:.6f} eV** (from: {src})")
                except Exception as e:
                    st.error(f"Error reading bulk energy: {e}")
        else:
            st.info("No `Bulk/` subfolder found in this compound.")

        # 4) Defects
        st.markdown("### 🧬 Defects")
        defects = discover_defects(comp_id)
        if not defects:
            st.warning("No defect folders found inside this compound.")
            st.stop()

        st.dataframe(pd.DataFrame({"Defect": list(defects.keys())}), use_container_width=True)

        defect_sel = st.selectbox("Select a defect", list(defects.keys()))
        defect_id = defects[defect_sel]

        # 5) Charge states & energies
        st.markdown("### ⚡ Charge States — Total Energies")
        charges = discover_charge_states(defect_id)
        if not charges:
            st.info("No charge-state subfolders (e.g., q+2, q0, q-1) found.")
        else:
            rows = []
            for qlbl, qid in charges.items():
                try:
                    e, src = parse_total_energy_for_folder(qid)
                    rows.append({"Charge": qlbl, "Total Energy (eV)": e, "Source": src})
                except Exception as e:
                    rows.append({"Charge": qlbl, "Total Energy (eV)": None, "Source": f"error: {e}"})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.download_button("Download energies (CSV)", df.to_csv(index=False).encode(), file_name=f"{comp_sel}_{defect_sel}_energies.csv")

    except HttpError as he:
        st.error(f"Google Drive API error: {he}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
