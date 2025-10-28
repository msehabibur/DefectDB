# defect_utils.py
import io
import gzip
import re
import ssl
import time
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st # Needed for st.cache_resource and st.secrets

import certifi
import httplib2
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── Config ───────────────────────────────────────────────────────────────────
ROOT_FOLDER_ID_DEFAULT = "1gYTtFpPIRCDWpLBW855RA6XwG0buifbi"
DEFAULT_VBM = 0.0
DEFAULT_GAP = 1.5

# ── Auth ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def drive_service():
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gdrive_service_account"]),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
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
                fields="nextPageToken, files(id,name,mimeType,modifiedTime,size)",
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

# ── VASP parsers ─────────────────────────────────────────────────────────────
def maybe_gunzip(name: str, data: bytes) -> bytes:
    return gzip.decompress(data) if name.lower().endswith(".gz") else data

def parse_outcar_energy(raw: bytes) -> Optional[float]:
    try:
        from pymatgen.io.vasp.outputs import Outcar
        with tempfile.NamedTemporaryFile(delete=True, suffix=".OUTCAR") as tmp:
            tmp.write(raw); tmp.flush()
            out = Outcar(tmp.name)
            if getattr(out, "final_energy", None) is not None:
                return float(out.final_energy)
    except Exception:
        pass
    txt = raw.decode(errors="ignore")
    m = None
    for pat in [r"free\s+energy\s+TOTEN\s*=\s*([-\d\.Ee+]+)", r"energy\s+without\s+entropy\s*=\s*([-\d\.Ee+]+)"]:
        hits = list(re.finditer(pat, txt))
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
    txt = raw.decode(errors="ignore")
    for line in reversed([l for l in txt.splitlines() if l.strip()]):
        m = re.search(r"E0\s*=\s*([-\d\.Ee+]+)", line)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None

def parse_total_energy_for_folder(folder_id: str) -> Tuple[Optional[float], str]:
    kids = list_children(folder_id)
    cand = {k["name"].lower(): k for k in kids}
    def try_file(names, parser):
        for nm in names:
            if nm in cand:
                raw = download_bytes(cand[nm]["id"])
                raw = maybe_gunzip(nm, raw)
                e = parser(raw)
                if e is not None: return e, nm
        return None, ""
    e, src = try_file(["outcar.gz", "outcar"], parse_outcar_energy)
    if e is not None: return e, src.upper()
    e, src = try_file(["vasprun.xml.gz", "vasprun.xml"], parse_vasprun_energy)
    if e is not None: return e, src
    e, src = try_file(["oszicar.gz", "oszicar"], parse_oszicar_energy)
    if e is not None: return e, src.upper()
    return None, "not_found"

def find_structure_file(folder_id: str) -> Tuple[Optional[bytes], str]:
    kids = list_children(folder_id)
    cand = {k["name"].lower(): k for k in kids}
    for nm in ["contcar.gz", "contcar", "poscar.gz", "poscar"]:
        if nm in cand:
            raw = download_bytes(cand[nm]["id"]); raw = maybe_gunzip(nm, raw)
            return raw, cand[nm]["name"]
    for f in kids:
        if f["name"].lower().endswith(".cif"):
            return download_bytes(f["id"]), f["name"]
    return None, ""

# ── Discovery ────────────────────────────────────────────────────────────────
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
    return {f["name"]: f["id"] for f in list_children(defect_folder_id)
            if f["mimeType"] == "application/vnd.google-apps.folder"}

def find_bulk_folder(compound_folder_id: str) -> Optional[str]:
    for f in list_children(compound_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder" and f["name"].lower() == "bulk":
            return f["id"]
    return None

# ── Charge label → q (STANDARD mapping; Neutral handled) ─────────────────────
def parse_charge_label_to_q(label: str) -> Optional[int]:
    s = (label or "").strip().lower()
    if s in {"neutral", "neut"}: return 0
    m = re.match(r'^[pm]\s*(\d+)$', s)
    if m:
        n = int(m.group(1))
        return +n if s.startswith('p') else -n
    m = re.match(r'^q\s*([+\-]?\d+)$', s)
    if m: return int(m.group(1))
    m = re.match(r'^[+\-]?\d+$', s)
    if m: return int(s)
    m = re.match(r'^(charged|charge)\s*([+\-]?)\s*(\d+)$', s)
    if m:
        sign = m.group(2) or "0"
        n = int(m.group(3))
        if n == 0: return 0
        return +n if sign == '+' else -n
    m = re.search(r'([+\-]?\d+)', s)
    if m:
        try: return int(m.group(1))
        except Exception: return None
    return None

# ── data.csv loading (global or per-compound) ────────────────────────────────
def load_root_params(root_folder_id: str) -> Optional[pd.DataFrame]:
    meta = find_file_in_folder_by_name(root_folder_id, "data.csv")
    if not meta:
        return None
    raw = download_bytes(meta["id"])
    try:
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return None

def get_params_row_for_compound(params_df: Optional[pd.DataFrame], compound: str) -> Tuple[float, float, Optional[pd.Series]]:
    vbm, gap, row = DEFAULT_VBM, DEFAULT_GAP, None
    if params_df is None or params_df.empty:
        return vbm, gap, row

    cols = {c.lower(): c for c in params_df.columns}
    if "compound" in cols:
        cc = cols["compound"]
        rows = params_df[params_df[cc].astype(str) == str(compound)]
        if not rows.empty:
            r = rows.iloc[0]
            for key in ["VBM (eV)", "VBM"]:
                if key in r.index:
                    try: vbm = float(r[key]); break
                    except Exception: pass
            for key in ["Bandgap (eV)", "gap"]:
                if key in r.index:
                    try: gap = float(r[key]); break
                    except Exception: pass
            row = r
            return vbm, gap, row
    row = params_df.iloc[0]
    return vbm, gap, row

def mu_from_params_row(row: Optional[pd.Series], defect_name: str, chem_pot: str) -> Optional[float]:
    if row is None:
        return None
    keys = [
        f"{defect_name} ({chem_pot})",
        f"{defect_name}({chem_pot})",
    ]
    for k in keys:
        if k in row.index:
            try:
                v = float(row[k])
                if np.isnan(v): return None
                return v
            except Exception:
                continue
    return None

# ── Build correction/energy table ────────────────────────────────────────────
def build_correction_table_for_compound(
    root_id: str,
    compound: str,
    comp_id: str,
    chem_pot: str,  # 'Cd-rich' or 'Te-rich'
    root_params: Optional[pd.DataFrame],
    restrict_defects: Optional[List[str]] = None,
) -> pd.DataFrame:
    vbm, gap, params_row = get_params_row_for_compound(root_params, compound)

    toten_pure, _src = (None, "not_found")
    bulk_id = find_bulk_folder(comp_id)
    if bulk_id:
        toten_pure, _src = parse_total_energy_for_folder(bulk_id)

    defects = discover_defects(comp_id)
    names = list(defects.keys())
    if restrict_defects:
        names = [n for n in names if n in restrict_defects]

    rows = []
    for dname in sorted(names):
        did = defects[dname]
        charges = discover_charge_states(did)

        vals = {"Toten_p2": np.nan, "Toten_p1": np.nan, "Toten_neut": np.nan, "Toten_m1": np.nan, "Toten_m2": np.nan}
        for qlbl, qid in charges.items():
            q = parse_charge_label_to_q(qlbl)
            e, _ = parse_total_energy_for_folder(qid)
            if q is None: continue
            if q == +2: vals["Toten_p2"] = e
            elif q == +1: vals["Toten_p1"] = e
            elif q == 0:  vals["Toten_neut"] = e
            elif q == -1: vals["Toten_m1"] = e
            elif q == -2: vals["Toten_m2"] = e

        corr = {"Corr_p2": 0.0, "Corr_p1": 0.0, "Corr_neut": 0.0, "Corr_m1": 0.0, "Corr_m2": 0.0}
        mu_val = mu_from_params_row(params_row, dname, chem_pot)

        rows.append({
            "Compound": compound,
            "Defect": dname,
            "Type": "Native" if ("_i" not in dname and "_" in dname) else "Extrinsic",
            "Plot": "Y",
            "Label": dname,
            "gap": gap,
            "VBM": vbm,
            "Toten_pure": toten_pure,
            **vals,
            **corr,
            "mu": mu_val if mu_val is not None else np.nan,
        })

    return pd.DataFrame(rows)
