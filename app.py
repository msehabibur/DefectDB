# app.py — DefectDB Browser (Drive) + Auto-built Defect Analysis (no manual upload needed)
#
# WHAT THIS APP DOES
# 1) Browses your Google Drive DefectDB:
#    - Lists compounds
#    - Shows Bulk energy (if present)
#    - Shows defect folders and charge-state subfolders (q+2, q+1, q0, q-1, q-2)
# 2) Builds a correction/energy DataFrame automatically (no upload):
#    - Reads VBM and Bandgap and per-defect μ from ROOT/data.csv
#    - Pulls total energies from VASP outputs (OUTCAR(.gz) → vasprun.xml(.gz) → OSZICAR(.gz))
#    - Sets Corr_* to 0.0 for now (can be extended)
#    - Lets you pick which defects to analyze and the chemical potential set (Cd-rich/Te-rich/Medium)
#    - Displays and lets you download the constructed DataFrame
#    - Plots formation energy vs Fermi level (lower envelope across available charges)
#
# REQUIREMENTS
#   streamlit
#   google-api-python-client
#   google-auth
#   google-auth-httplib2
#   pandas
#   numpy
#   pymatgen
#   certifi
#
# SECRETS: .streamlit/secrets.toml must contain [gdrive_service_account] (service account JSON).
#
# NOTES
# - Uses certifi to avoid SSL issues, includes light retries.
# - Uses width="stretch" (no deprecated use_container_width).

import io
import gzip
import re
import ssl
import time
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import certifi
import httplib2
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# ── Global config & SSL hardening ─────────────────────────────────────────────
st.set_page_config(page_title="DefectDB Browser (Drive) + Analysis", layout="wide")
httplib2.CA_CERTS = certifi.where()
ssl_ctx = ssl.create_default_context(cafile=certifi.where())
ROOT_FOLDER_ID_DEFAULT = "1gYTtFpPIRCDWpLBW855RA6XwG0buifbi"  # <- your root folder

# ── Auth ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def drive_service():
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gdrive_service_account"]),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

# ── Retry helper ─────────────────────────────────────────────────────────────
def _with_retries(fn, *, tries=3, base_delay=0.8):
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
    out = []
    token = None
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

def find_child_by_lowername(folder_id: str, name_lc: str) -> Optional[Dict]:
    """Finds child file/folder with exact lower-cased name match."""
    kids = list_children(folder_id)
    for f in kids:
        if f["name"].lower() == name_lc:
            return f
    return None

def find_file_in_folder_by_name(folder_id: str, filename: str) -> Optional[Dict]:
    """Return file metadata whose name exactly matches (case-insensitive)."""
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
        try: return float(m.group(1))
        except Exception: return None
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
            try: return float(m.group(1))
            except Exception: return None
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
            raw = download_bytes(cand[nm]["id"])
            raw = maybe_gunzip(nm, raw)
            return raw, cand[nm]["name"]
    for f in kids:
        if f["name"].lower().endswith(".cif"):
            return download_bytes(f["id"]), f["name"]
    return None, ""

# ── Discovery ────────────────────────────────────────────────────────────────
def discover_compounds(root_folder_id: str) -> Dict[str, str]:
    return {f["name"]: f["id"] for f in list_children(root_folder_id)
            if f["mimeType"] == "application/vnd.google-apps.folder"}

def discover_defects(compound_folder_id: str) -> Dict[str, str]:
    return {f["name"]: f["id"] for f in list_children(compound_folder_id)
            if f["mimeType"] == "application/vnd.google-apps.folder" and f["name"].lower() != "bulk"}

def discover_charge_states(defect_folder_id: str) -> Dict[str, str]:
    m = {f["name"]: f["id"] for f in list_children(defect_folder_id)
         if f["mimeType"] == "application/vnd.google-apps.folder"}
    def parse_q(lbl: str) -> Optional[int]:
        s = lbl.strip().lower().replace("q", "")
        try: return int(s)
        except Exception: return None
    return dict(sorted(m.items(), key=lambda x: (parse_q(x[0]) is None, -(parse_q(x[0]) or 0))))

def find_bulk_folder(compound_folder_id: str) -> Optional[str]:
    for f in list_children(compound_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder" and f["name"].lower() == "bulk":
            return f["id"]
    return None

# ── Read ROOT/data.csv (VBM, Gap, per-defect μ) ─────────────────────────────
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

def mu_lookup_from_data_csv(row: pd.Series, defect_name: str, chem_pot: str) -> Optional[float]:
    """
    defect_name like 'V_Cd', 'As_Te', 'Cl_Te'
    chem_pot in {'Cd-rich','Te-rich','A-rich','Medium','B-rich'}
    We look for columns like: 'V_Cd (Cd-rich) μ (eV)' or '(Te-rich)' etc.
    """
    chem_alias = {"A-rich": "Cd-rich", "B-rich": "Te-rich"}
    chem = chem_alias.get(chem_pot, chem_pot)
    candidates = [
        f"{defect_name} ({chem}) μ (eV)",
        f"{defect_name} ({chem}) μ",
        f"{defect_name}_{chem}_mu",
    ]
    for key in candidates:
        if key in row.index:
            try:
                v = float(row[key])
                if np.isnan(v): return None
                return v
            except Exception:
                continue
    return None

# ── Build correction/energy table automatically for a compound ──────────────
def build_correction_table_for_compound(root_id: str, compound: str, comp_id: str,
                                        chem_pot: str, root_params: Optional[pd.DataFrame],
                                        restrict_defects: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
    Compound, Defect, Type, Plot, Label, gap, VBM, Toten_pure,
    Toten_p2, Toten_p1, Toten_neut, Toten_m1, Toten_m2,
    Corr_p2, Corr_p1, Corr_neut, Corr_m1, Corr_m2,
    mu_A_rich, mu_med, mu_Te_rich
    - Corr_* are set to 0.0 here (placeholders)
    - μ columns: we populate only the selected set (A-rich/Medium/Te-rich aliases) where possible
    """
    # VBM & gap from data.csv row, if present
    vbm = 0.0
    gap = 1.5
    muA = muM = muT = np.nan
    row_params = None
    if root_params is not None:
        row = root_params[root_params["Compound"].astype(str) == compound]
        if not row.empty:
            row = row.iloc[0]
            row_params = row
            # permit both "VBM" and "VBM (eV)" / "Bandgap (eV)" / "gap"
            for key in ["VBM (eV)", "VBM"]:
                if key in row.index:
                    try:
                        vbm = float(row[key])
                        break
                    except Exception:
                        pass
            for key in ["Bandgap (eV)", "gap"]:
                if key in row.index:
                    try:
                        gap = float(row[key]); break
                    except Exception:
                        pass

    # Bulk energy
    toten_pure, _src = (None, "not_found")
    bulk_id = find_bulk_folder(comp_id)
    if bulk_id:
        toten_pure, _src = parse_total_energy_for_folder(bulk_id)

    # Discover defects
    defects = discover_defects(comp_id)
    names = list(defects.keys())
    if restrict_defects:
        names = [n for n in names if n in restrict_defects]

    rows = []
    for dname in sorted(names):
        did = defects[dname]
        charges = discover_charge_states(did)
        # collect charge energies
        vals = {"Toten_p2": np.nan, "Toten_p1": np.nan, "Toten_neut": np.nan, "Toten_m1": np.nan, "Toten_m2": np.nan}
        for qlbl, qid in charges.items():
            qstr = qlbl.strip().lower().replace("q", "")
            try:
                q = int(qstr)
            except Exception:
                q = None
            e, _ = parse_total_energy_for_folder(qid)
            if q == +2: vals["Toten_p2"] = e
            elif q == +1: vals["Toten_p1"] = e
            elif q == 0: vals["Toten_neut"] = e
            elif q == -1: vals["Toten_m1"] = e
            elif q == -2: vals["Toten_m2"] = e

        # corrections default 0.0
        corr = {"Corr_p2": 0.0, "Corr_p1": 0.0, "Corr_neut": 0.0, "Corr_m1": 0.0, "Corr_m2": 0.0}

        # μ: fill relevant column based on selection; keep others NaN
        mu_map = {"mu_A_rich": np.nan, "mu_med": np.nan, "mu_Te_rich": np.nan}
        if row_params is not None:
            if chem_pot in ["A-rich", "Cd-rich"]:
                mv = mu_lookup_from_data_csv(row_params, dname, "Cd-rich")
                if mv is not None: mu_map["mu_A_rich"] = mv
            elif chem_pot == "Medium":
                # if you later add Medium columns, handle here; leave NaN for now
                mu_map["mu_med"] = np.nan
            elif chem_pot in ["Te-rich", "B-rich"]:
                mv = mu_lookup_from_data_csv(row_params, dname, "Te-rich")
                if mv is not None: mu_map["mu_Te_rich"] = mv

        rows.append({
            "Compound": compound,
            "Defect": dname,
            "Type": "Native" if ("_i" not in dname and "_" in dname and dname.split("_")[0] in ["V","Cd","Zn","Te","Se"]) else "Extrinsic",
            "Plot": "Y",
            "Label": dname,
            "gap": gap,
            "VBM": vbm,
            "Toten_pure": toten_pure,
            **vals,
            **corr,
            **mu_map
        })

    df = pd.DataFrame(rows)
    return df

# ── Formation energy plotting ────────────────────────────────────────────────
def _coerce_float(x):
    try:
        if x is None: return np.nan
        if isinstance(x, (int, float, np.floating)): return float(x)
        s = str(x).strip().replace(",", "").replace("−", "-")
        return float(s)
    except Exception:
        return np.nan

def _available_charges(row):
    out = []
    for q, tcol, ccol in [(+2,"Toten_p2","Corr_p2"), (+1,"Toten_p1","Corr_p1"),
                          (0,"Toten_neut","Corr_neut"), (-1,"Toten_m1","Corr_m1"), (-2,"Toten_m2","Corr_m2")]:
        tq = _coerce_float(row.get(tcol)); cq = _coerce_float(row.get(ccol))
        if not np.isnan(tq) and not np.isnan(cq): out.append((q, tq, cq))
    return out

def _pick_mu_from_row(row, chem_pot: str) -> float:
    if chem_pot in ["A-rich", "Cd-rich"]:
        v = _coerce_float(row.get("mu_A_rich"))
        return 0.0 if np.isnan(v) else v
    if chem_pot == "Medium":
        v = _coerce_float(row.get("mu_med"))
        return 0.0 if np.isnan(v) else v
    if chem_pot in ["Te-rich", "B-rich"]:
        v = _coerce_float(row.get("mu_Te_rich"))
        return 0.0 if np.isnan(v) else v
    return 0.0

def plot_formation_energy(df_sub: pd.DataFrame, chem_pot: str, title: str):
    plt.rcParams["font.size"] = 22
    fig, ax = plt.subplots(figsize=(7,6))
    if df_sub.empty:
        ax.text(0.5,0.5,"No data", ha="center", va="center", transform=ax.transAxes)
        st.pyplot(fig, clear_figure=True); return
    gap = _coerce_float(df_sub["gap"].iloc[0]); gap = 1.5 if (np.isnan(gap) or gap<=0) else gap
    vbm = _coerce_float(df_sub["VBM"].iloc[0]); vbm = 0.0 if np.isnan(vbm) else vbm
    EF = np.arange(0.0, gap+1e-9, 0.01)

    for _, r in df_sub.iterrows():
        charges = _available_charges(r)
        if not charges:
            continue
        mu = _pick_mu_from_row(r, chem_pot)
        toten_pure = _coerce_float(r.get("Toten_pure"))
        if np.isnan(toten_pure):
            continue
        curves = []
        for q, tq, cq in charges:
            Ef_q = (tq - toten_pure + mu + q*(EF + vbm) + cq)
            curves.append(Ef_q)
        env = np.min(np.vstack(curves), axis=0)
        label = str(r.get("Label") or r.get("Defect"))
        ax.plot(EF, env, linewidth=3, label=label)

    ax.axvline(0.0, linestyle="dotted", color="black")
    ax.axvline(gap, linestyle="dotted", color="black")
    ax.fill_betweenx([0,12], -10, 0, alpha=0.2, color="lightgrey")
    ax.fill_betweenx([0,12], gap, gap+10, alpha=0.2, color="lightgrey")
    ax.set_xlim(-0.1, gap+0.1); ax.set_ylim(0,10)
    ax.set_xlabel("Fermi Level (eV)"); ax.set_ylabel("Defect Formation Energy (eV)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    st.pyplot(fig, clear_figure=True)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    st.download_button("Download plot (PNG)", buf.getvalue(), file_name=f"{title}.png")

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🧪 DefectDB Browser (Drive) + Auto Analysis")

with st.sidebar:
    st.header("Data Source")
    root_id = st.text_input("Root Folder ID", value=ROOT_FOLDER_ID_DEFAULT)
    if st.button("Scan Root"):
        try:
            compounds = discover_compounds(root_id)
            if not compounds:
                st.error("No compound folders found in this root (or missing permissions).")
                st.stop()
            st.session_state["compounds"] = dict(sorted(compounds.items(), key=lambda x: x[0].lower()))
            # also cache root params (data.csv) right away
            st.session_state["root_params"] = load_root_params(root_id)
            if st.session_state["root_params"] is None:
                st.warning("ROOT data.csv not found — VBM/Gap/μ will default (VBM=0, gap=1.5, μ=0).")
            else:
                st.success("Loaded ROOT/data.csv successfully.")
            st.success(f"Found {len(compounds)} compound folder(s).")
        except Exception as e:
            st.error(f"Error: {e}")

compounds = st.session_state.get("compounds")
root_params = st.session_state.get("root_params")

# Overview
if compounds:
    st.subheader("📦 Compounds")
    overview = []
    for c, cid in compounds.items():
        overview.append({"Compound": c, "Has Bulk": "Yes" if find_bulk_folder(cid) else "No"})
    st.dataframe(pd.DataFrame(overview), width="stretch")

    # Pick a compound
    comp_sel = st.selectbox("Select a compound", list(compounds.keys()))
    comp_id = compounds[comp_sel]

    # Discover defects for this compound
    defects = discover_defects(comp_id)
    defect_names = sorted(defects.keys())
    st.markdown("### 🧬 Choose defects to analyze")
    chosen_defects = st.multiselect("Defects", defect_names, default=defect_names)

    chem_pot_choice = st.selectbox("Chemical potential set", ["Cd-rich", "Te-rich", "A-rich", "Medium", "B-rich"])
    st.caption("Tip: A-rich ≈ Cd-rich, B-rich ≈ Te-rich are treated as aliases for μ lookup.")

    # Build the correction table
    if st.button("Build table & plot"):
        with st.spinner("Building table from Drive and data.csv..."):
            try:
                table = build_correction_table_for_compound(
                    root_id, comp_sel, comp_id, chem_pot_choice, root_params, chosen_defects
                )
                if table.empty:
                    st.warning("No rows were produced — check folder contents.")
                else:
                    st.success("Constructed correction/energy table.")
                    st.dataframe(table, width="stretch")
                    st.download_button("Download table (CSV)", table.to_csv(index=False).encode(),
                                       file_name=f"{comp_sel}_corrections.csv")

                    # Plot formation energy vs EF for the chosen defects
                    sub = table[table["Defect"].isin(chosen_defects)].copy()
                    plot_formation_energy(sub, chem_pot_choice,
                                          title=f"{comp_sel}_{chem_pot_choice}_formation_plot")

            except Exception as e:
                st.error(f"Failed to build/plot: {e}")

# Help
with st.expander("ℹ️ Help / Expected Structure & Columns"):
    st.markdown(
        """
**Drive structure**
- Root (the folder ID you enter)
  - `data.csv`  ← contains compound-level info (VBM, Bandgap, per-defect μ columns)
  - `CdTe/`
    - `Bulk/`    (OUTCAR(.gz) / vasprun.xml(.gz) / OSZICAR(.gz))
    - `V_Cd/`
      - `q0/`, `q+1/`, `q-1/`, ... each with a VASP output (OUTCAR/vasprun.xml/OSZICAR)
    - `Cl_Te/`
      - `q0/`, ...
  - `CdSe0.06Te0.94/`
    - ...

**`data.csv` columns**
- `Compound`
- `VBM (eV)` (or `VBM`)
- `Bandgap (eV)` (or `gap`)
- Per-defect μ columns, e.g. `V_Cd (Cd-rich) μ (eV)`, `V_Cd (Te-rich) μ (eV)`, `As_Te (Cd-rich) μ (eV)`, etc.

**Formation-energy formula**  
`E_f(q, EF) = Toten_q − Toten_pure + μ + q·(EF + VBM) + Corr_q`  
Here `Corr_q` defaults to 0.0 (you can add real corrections later).
"""
    )
