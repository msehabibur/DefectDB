# app.py — dynamic DefectDB browser (Drive)
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

    e, src = try_file(["outcar.gz", "outcar"], parse_outcar_energy)
    if e is not None:
        return e, src.upper()
    e, src = try_file(["vasprun.xml.gz", "vasprun.xml"], parse_vasprun_energy)
    if e is not None:
        return e, src
    e, src = try_file(["oszicar.gz", "oszicar"], parse_oszicar_energy)
    if e is not None:
        return e, src.upper()
    return None, "not_found"

# ---------- Discovery logic ----------
def discover_compounds(root_folder_id: str) -> Dict[str, str]:
    m = {}
    for f in list_children(root_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder":
            m[f["name"]] = f["id"]
    return dict(sorted(m.items(), key=lambda x: x[0].lower()))

def discover_defects(compound_folder_id: str) -> Dict[str, str]:
    m = {}
    for f in list_children(compound_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder" and f["name"].lower() != "bulk":
            m[f["name"]] = f["id"]
    return dict(sorted(m.items(), key=lambda x: x[0].lower()))

def discover_charge_states(defect_folder_id: str) -> Dict[str, str]:
    m = {}
    for f in list_children(defect_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder":
            m[f["name"]] = f["id"]

    def parse_q(lbl: str) -> Optional[int]:
        s = lbl.strip().lower().replace("q", "")
        try:
            return int(s)
        except Exception:
            return None

    # sort: numeric charges first (desc), then unknowns
    return dict(sorted(m.items(), key=lambda x: (parse_q(x[0]) is None, -(parse_q(x[0]) or 0))))

def find_bulk_folder(compound_folder_id: str) -> Optional[str]:
    for f in list_children(compound_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder" and f["name"].lower() == "bulk":
            return f["id"]
    return None

# ---------- Aggregation ----------
def scan_compound(compound_name: str, compound_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      bulk_df: 1-row table (Compound, Energy, Source) or 'not_found'
      defects_df: rows for each (Defect, Charge, Energy, Source), including not_found cases
    """
    # Bulk
    bulk_id = find_bulk_folder(compound_id)
    if bulk_id:
        ebulk, src = parse_total_energy_for_folder(bulk_id)
        bulk_rows = [{
            "Compound": compound_name,
            "Total Energy (eV)": ebulk,
            "Source": src if src else "not_found"
        }]
    else:
        bulk_rows = [{
            "Compound": compound_name,
            "Total Energy (eV)": None,
            "Source": "not_found"
        }]
    bulk_df = pd.DataFrame(bulk_rows)

    # Defects & charges
    defects = discover_defects(compound_id)
    defect_rows = []
    if not defects:
        # No defects at all -> emit a placeholder line
        defect_rows.append({
            "Compound": compound_name,
            "Defect": "—",
            "Charge": "—",
            "Total Energy (eV)": None,
            "Source": "no_defect_folders"
        })
    else:
        for defect_name, defect_id in defects.items():
            charges = discover_charge_states(defect_id)
            if not charges:
                defect_rows.append({
                    "Compound": compound_name,
                    "Defect": defect_name,
                    "Charge": "—",
                    "Total Energy (eV)": None,
                    "Source": "no_charge_state_folder"
                })
                continue
            for qlbl, qid in charges.items():
                try:
                    e, src = parse_total_energy_for_folder(qid)
                    defect_rows.append({
                        "Compound": compound_name,
                        "Defect": defect_name,
                        "Charge": qlbl,
                        "Total Energy (eV)": e,
                        "Source": src if src else "not_found"
                    })
                except Exception as ex:
                    defect_rows.append({
                        "Compound": compound_name,
                        "Defect": defect_name,
                        "Charge": qlbl,
                        "Total Energy (eV)": None,
                        "Source": f"error: {ex}"
                    })
    defects_df = pd.DataFrame(defect_rows)
    return bulk_df, defects_df

# ---------- UI ----------
st.title("🧪 DefectDB Browser")

with st.sidebar:
    st.header("Data Source")
    root_id = st.text_input("Root Folder ID", value=ROOT_FOLDER_ID_DEFAULT)
    if st.button("Scan Root"):
        try:
            compounds = discover_compounds(root_id)
            if not compounds:
                st.error("No compound folders found in this root. Make sure the service account has Viewer access.")
                st.stop()
            st.session_state["compounds"] = compounds
            st.success(f"Found {len(compounds)} compound folder(s).")
        except HttpError as he:
            st.error(f"Google Drive API error: {he}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

compounds = st.session_state.get("compounds", None)
if compounds:
    st.subheader("📦 Compounds")
    overview_rows = []
    for comp, comp_id in compounds.items():
        overview_rows.append({"Compound": comp, "Has Bulk": "Yes" if find_bulk_folder(comp_id) else "No"})
    st.dataframe(pd.DataFrame(overview_rows), width="stretch")

    comp_sel = st.selectbox("Select a compound", list(compounds.keys()))
    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("Scan Selected Compound"):
            comp_id = compounds[comp_sel]
            try:
                bulk_df, defects_df = scan_compound(comp_sel, comp_id)
                st.session_state["bulk_df"] = bulk_df
                st.session_state["defects_df"] = defects_df
                st.success("Scan complete.")
            except HttpError as he:
                st.error(f"Google Drive API error: {he}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    with colB:
        if st.button("Scan ALL Compounds"):
            all_bulk, all_defects = [], []
            try:
                for cname, cid in compounds.items():
                    bdf, ddf = scan_compound(cname, cid)
                    all_bulk.append(bdf)
                    all_defects.append(ddf)
                st.session_state["bulk_df"] = pd.concat(all_bulk, ignore_index=True)
                st.session_state["defects_df"] = pd.concat(all_defects, ignore_index=True)
                st.success("Full scan complete.")
            except HttpError as he:
                st.error(f"Google Drive API error: {he}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# Results
bulk_df = st.session_state.get("bulk_df", None)
defects_df = st.session_state.get("defects_df", None)

if bulk_df is not None or defects_df is not None:
    st.markdown("### 🧱 Bulk Energy (per compound)")
    if bulk_df is not None:
        # Show 'not_found' clearly even when energy is NaN
        show_bulk = bulk_df.copy()
        st.dataframe(show_bulk, width="stretch")
        st.download_button(
            "Download bulk energies (CSV)",
            show_bulk.to_csv(index=False).encode(),
            file_name="bulk_energies.csv",
        )
    else:
        st.info("No bulk data yet. Click a scan button above.")

    st.markdown("### 🧬 Defects — Charge-State Energies")
    if defects_df is not None:
        show_def = defects_df.copy()
        st.dataframe(show_def, width="stretch")
        st.download_button(
            "Download defect energies (CSV)",
            show_def.to_csv(index=False).encode(),
            file_name="defect_energies.csv",
        )
    else:
        st.info("No defect data yet. Click a scan button above.")

# ---------- Help ----------
with st.expander("ℹ️ Help / Expected Folder Structure"):
    st.markdown(
        """
**Folder structure expected:**
- Root Folder (the ID you paste here)
  - `Compound_A/`
    - `Bulk/`  ← (optional) contains `OUTCAR(.gz)`, `vasprun.xml(.gz)`, or `OSZICAR(.gz)`
    - `V_Cd/`
      - `q0/`, `q+1/`, `q-1/`, ... each with a VASP output file
    - `Cl_Te/`
      - `q0/`, `q+1/`, `q-1/`, ...
  - `Compound_B/`
    - ...

**Energy parsing priority:** `OUTCAR` → `vasprun.xml` → `OSZICAR`.  
If no recognized file is found, the **Source** column will show `not_found`.
"""
    )
