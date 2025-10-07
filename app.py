# app.py — DefectDB Browser (Drive) + Defect Analysis
#
# WHAT THIS APP DOES
# ──────────────────
# 1) BROWSE GOOGLE DRIVE DATABASE
#    • Lists compounds (folders) under a Root Drive folder ID
#    • For a compound: shows Bulk/ (optional) and any defect folders
#    • For a defect: lists charge-state subfolders (e.g., q+2, q0, q-1, …)
#    • Parses total energy in each charge folder by preferring OUTCAR(.gz) → vasprun.xml(.gz) → OSZICAR(.gz)
#
# 2) DEFECT ANALYSIS (YOUR NEW FLOW)
#    • Upload a correction-energy table (xlsx/csv) with columns like:
#      Compound, Defect, Type, Plot, Label, gap, VBM, Toten_pure,
#      Toten_p2, Toten_p1, Toten_neut, Toten_m1, Toten_m2,
#      Corr_p2, Corr_p1, Corr_neut, Corr_m1, Corr_m2, mu_A_rich, mu_med, mu_Te_rich
#    • Pick the defects to analyze (from the uploaded table)
#    • Choose Action:
#         (1) Plot defect formation energy vs Fermi level
#              E_f(q, EF) = Toten_q - Toten_pure + μ + q * (EF + VBM) + Corr_q
#              (plots the lower envelope over available charge states)
#         (2) Download DFT-optimized structures from Drive (CONTCAR/POSCAR/*.cif)
#
# REQUIREMENTS
# ────────────
# streamlit
# google-api-python-client
# google-auth
# google-auth-httplib2
# pandas
# numpy
# pymatgen
# certifi
#
# SECRETS
# ───────
# .streamlit/secrets.toml must define [gdrive_service_account] with your service account JSON.
#
# NOTES
# ─────
# • Uses certifi to fix SSL on minimal hosts/containers.
# • Uses width="stretch" (replacing use_container_width).
# • “Not found” cases are surfaced in the UI and CSVs.

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

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG & SSL HARDENING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DefectDB Browser (Drive)", layout="wide")

# Point httplib2 to certifi CA bundle for stable SSL on minimal hosts
httplib2.CA_CERTS = certifi.where()
ssl_ctx = ssl.create_default_context(cafile=certifi.where())

# Default root folder ID (change to yours)
ROOT_FOLDER_ID_DEFAULT = "1gYTtFpPIRCDWpLBW855RA6XwG0buifbi"


# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE CLIENT
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def drive_service():
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gdrive_service_account"]),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ─────────────────────────────────────────────────────────────────────────────
# SMALL RETRY WRAPPER (helps transient SSL/network hiccups)
# ─────────────────────────────────────────────────────────────────────────────
def _with_retries(fn, *, tries: int = 3, base_delay: float = 0.8):
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(base_delay * (2 ** k))


# ─────────────────────────────────────────────────────────────────────────────
# DRIVE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def list_children(folder_id: str) -> List[Dict]:
    svc = drive_service()
    q = f"'{folder_id}' in parents and trashed = false"
    out: List[Dict] = []
    token = None
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


# ─────────────────────────────────────────────────────────────────────────────
# VASP PARSING HELPERS (OUTCAR, vasprun.xml, OSZICAR)
# ─────────────────────────────────────────────────────────────────────────────
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
    """Look for OUTCAR(.gz) → vasprun.xml(.gz) → OSZICAR(.gz)."""
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
    if e is not None: return e, src.upper()

    e, src = try_file(["vasprun.xml.gz", "vasprun.xml"], parse_vasprun_energy)
    if e is not None: return e, src

    e, src = try_file(["oszicar.gz", "oszicar"], parse_oszicar_energy)
    if e is not None: return e, src.upper()

    return None, "not_found"


def find_structure_file(folder_id: str) -> Tuple[Optional[bytes], str]:
    """
    Try to fetch a representative optimized structure file from a charge folder.
    Preference: CONTCAR(.gz) → POSCAR(.gz) → *.cif (first found).
    """
    kids = list_children(folder_id)
    cand = {k["name"].lower(): k for k in kids}

    order = [
        "contcar.gz", "contcar",
        "poscar.gz", "poscar",
    ]
    for nm in order:
        if nm in cand:
            raw = download_bytes(cand[nm]["id"])
            raw = maybe_gunzip(nm, raw)
            return raw, cand[nm]["name"]

    # Fallback to any .cif
    for f in kids:
        if f["name"].lower().endswith(".cif"):
            raw = download_bytes(f["id"])
            return raw, f["name"]

    return None, ""


# ─────────────────────────────────────────────────────────────────────────────
# DISCOVERY LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def discover_compounds(root_folder_id: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for f in list_children(root_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder":
            m[f["name"]] = f["id"]
    return dict(sorted(m.items(), key=lambda x: x[0].lower()))


def discover_defects(compound_folder_id: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for f in list_children(compound_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder" and f["name"].lower() != "bulk":
            m[f["name"]] = f["id"]
    return dict(sorted(m.items(), key=lambda x: x[0].lower()))


def discover_charge_states(defect_folder_id: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for f in list_children(defect_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder":
            m[f["name"]] = f["id"]

    def parse_q(lbl: str) -> Optional[int]:
        s = lbl.strip().lower().replace("q", "")
        try:
            return int(s)
        except Exception:
            return None

    # Sort numeric charges desc (q+2, q+1, q0, q-1, …) then unknowns
    return dict(sorted(m.items(), key=lambda x: (parse_q(x[0]) is None, -(parse_q(x[0]) or 0))))


def find_bulk_folder(compound_folder_id: str) -> Optional[str]:
    for f in list_children(compound_folder_id):
        if f["mimeType"] == "application/vnd.google-apps.folder" and f["name"].lower() == "bulk":
            return f["id"]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def scan_compound(compound_name: str, compound_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (bulk_df, defects_df) for one compound."""
    # Bulk
    bulk_id = find_bulk_folder(compound_id)
    if bulk_id:
        ebulk, src = parse_total_energy_for_folder(bulk_id)
        bulk_rows = [{
            "Compound": compound_name,
            "Total Energy (eV)": ebulk,
            "Source": src
        }]
    else:
        bulk_rows = [{
            "Compound": compound_name,
            "Total Energy (eV)": None,
            "Source": "not_found"
        }]
    bulk_df = pd.DataFrame(bulk_rows)

    # Defects/charges
    defects = discover_defects(compound_id)
    rows = []
    if not defects:
        rows.append({
            "Compound": compound_name,
            "Defect": "—",
            "Charge": "—",
            "Total Energy (eV)": None,
            "Source": "no_defect_folders"
        })
    else:
        for dname, did in defects.items():
            charges = discover_charge_states(did)
            if not charges:
                rows.append({
                    "Compound": compound_name,
                    "Defect": dname,
                    "Charge": "—",
                    "Total Energy (eV)": None,
                    "Source": "no_charge_state_folder"
                })
                continue
            for qlbl, qid in charges.items():
                try:
                    e, src = parse_total_energy_for_folder(qid)
                    rows.append({
                        "Compound": compound_name,
                        "Defect": dname,
                        "Charge": qlbl,
                        "Total Energy (eV)": e,
                        "Source": src
                    })
                except Exception as ex:
                    rows.append({
                        "Compound": compound_name,
                        "Defect": dname,
                        "Charge": qlbl,
                        "Total Energy (eV)": None,
                        "Source": f"error: {ex}"
                    })
    defects_df = pd.DataFrame(rows)
    return bulk_df, defects_df


# ─────────────────────────────────────────────────────────────────────────────
# UI: SIDEBAR — DATA SOURCE
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧪 DefectDB Browser (Drive) + Defect Analysis")

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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DRIVE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
compounds = st.session_state.get("compounds")
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
            except Exception as e:
                st.error(f"Error: {e}")

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
            except Exception as e:
                st.error(f"Error: {e}")

# Results
bulk_df = st.session_state.get("bulk_df")
defects_df = st.session_state.get("defects_df")

if bulk_df is not None or defects_df is not None:
    st.markdown("### 🧱 Bulk Energy (per compound)")
    if bulk_df is not None:
        st.dataframe(bulk_df, width="stretch")
        st.download_button("Download bulk energies (CSV)", bulk_df.to_csv(index=False).encode(), file_name="bulk_energies.csv")
    else:
        st.info("No bulk data yet. Click a scan button above.")

    st.markdown("### 🧬 Defects — Charge-State Energies")
    if defects_df is not None:
        st.dataframe(defects_df, width="stretch")
        st.download_button("Download defect energies (CSV)", defects_df.to_csv(index=False).encode(), file_name="defect_energies.csv")
    else:
        st.info("No defect data yet. Click a scan button above.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DEFECT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📈 Defect Analysis")

st.markdown(
    "Upload your **correction-energy table** (xlsx or csv) with columns like:"
    " `Compound, Defect, Type, Plot, Label, gap, VBM, Toten_pure, Toten_p2, Toten_p1, Toten_neut, Toten_m1, Toten_m2, "
    " Corr_p2, Corr_p1, Corr_neut, Corr_m1, Corr_m2, mu_A_rich, mu_med, mu_Te_rich`."
)

uploaded = st.file_uploader("Upload correction-energy file", type=["xlsx", "xls", "csv"])

def _coerce_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    s = s.replace(",", "").replace("−", "-")
    try:
        return float(s)
    except Exception:
        return np.nan

def _available_charge_terms(row):
    """Return list of (q, toten_col, corr_col) that exist and are not NaN."""
    opts = []
    for q, tcol, ccol in [
        (+2, "Toten_p2", "Corr_p2"),
        (+1, "Toten_p1", "Corr_p1"),
        ( 0, "Toten_neut", "Corr_neut"),
        (-1, "Toten_m1", "Corr_m1"),
        (-2, "Toten_m2", "Corr_m2"),
    ]:
        tval = _coerce_float(row.get(tcol))
        cval = _coerce_float(row.get(ccol))
        if not np.isnan(tval) and not np.isnan(cval):
            opts.append((q, tval, cval))
    return opts

def _pick_mu(row, chem_pot_key: str):
    # chem_pot_key in {"A-rich", "Medium", "Te-rich"}
    if chem_pot_key == "A-rich":
        return _coerce_float(row.get("mu_A_rich"))
    elif chem_pot_key == "Medium":
        return _coerce_float(row.get("mu_med"))
    elif chem_pot_key in {"Te-rich", "B-rich"}:
        # Support both names; many users label Te-rich as mu_Te_rich
        return _coerce_float(row.get("mu_Te_rich"))
    return np.nan

if uploaded:
    # Load table
    try:
        if uploaded.name.lower().endswith(".csv"):
            table = pd.read_csv(uploaded)
        else:
            table = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        table = None

    if table is not None:
        st.success(f"Loaded table with {len(table)} rows.")
        # Normalize column names (strip spaces)
        table.columns = [c.strip() for c in table.columns]
        if "Plot" in table.columns:
            table["Plot"] = table["Plot"].astype(str).str.strip()

        # Pick Compound for analysis
        compounds_in_table = sorted(table["Compound"].astype(str).unique())
        compound_for_plot = st.selectbox("Compound to analyze", compounds_in_table)

        # Filter to compound
        sub = table[table["Compound"].astype(str) == compound_for_plot].copy()
        # Which defects to analyze?
        all_defects = sorted(sub["Defect"].astype(str).unique())
        default_defects = sorted(sub[sub.get("Plot", "Y").str.upper().eq("Y")]["Defect"].astype(str).unique()) or all_defects
        chosen_defects = st.multiselect("Choose defects to analyze", all_defects, default=default_defects)

        # Chemistry condition (maps to mu_* columns)
        chem_pot_choice = st.selectbox("Chemical potential set", ["A-rich", "Medium", "Te-rich"])

        # Choose Action
        action = st.radio("Action", ["Plot formation energy vs Fermi level", "Download optimized structures from Drive"])

        if action == "Plot formation energy vs Fermi level":
            # global plot style
            plt.rcParams["font.size"] = 22
            # create figure
            fig, ax = plt.subplots(figsize=(7, 6))
            # for EF range, use band gap from first matching row (fallbacks handled)
            gap_val = _coerce_float(sub["gap"].iloc[0]) if len(sub) else np.nan
            if np.isnan(gap_val) or gap_val <= 0:
                st.warning("Bandgap (gap) missing or non-positive — defaulting to 1.5 eV.")
                gap_val = 1.5

            VBM_val = _coerce_float(sub["VBM"].iloc[0]) if len(sub) else 0.0
            EF = np.arange(0.0, gap_val + 1e-9, 0.01)

            # line color cycle (matplotlib default)
            for dname in chosen_defects:
                rows_d = sub[sub["Defect"].astype(str) == dname]
                if rows_d.empty:
                    st.info(f"{dname}: no rows in table.")
                    continue

                # pick label from first row; fall back to defect name
                label = rows_d.iloc[0].get("Label", dname)
                # compute lower envelope over available charge states
                # E_f(q, EF) = Toten_q - Toten_pure + mu + q*(EF + VBM) + Corr_q
                # build array per row, then take min across rows+charges
                # (if multiple rows for same defect exist)
                env = None
                any_valid = False
                for _, r in rows_d.iterrows():
                    toten_pure = _coerce_float(r.get("Toten_pure"))
                    mu = _pick_mu(r, chem_pot_choice)
                    if np.isnan(toten_pure) or np.isnan(mu):
                        continue
                    charges = _available_charge_terms(r)
                    if not charges:
                        continue
                    any_valid = True
                    # compute energies for every charge option, then take min at each EF
                    # stack columns per charge state
                    curves = []
                    for q, toten_q, corr_q in charges:
                        # E_f(EF) for this q
                        Ef_q = (toten_q - toten_pure + mu + q * (EF + VBM_val) + corr_q)
                        curves.append(Ef_q)
                    # lower envelope over charges
                    curve_min = np.min(np.vstack(curves), axis=0)
                    env = curve_min if env is None else np.minimum(env, curve_min)

                if not any_valid:
                    st.warning(f"{dname}: no usable charge-state energies found (check Toten_* / Corr_* / mu_* columns).")
                    continue

                ax.plot(EF, env, linewidth=3, label=str(label))

            # Decorations: band edges and shading
            ax.axvline(0.0, linestyle="dotted", color="black")
            ax.axvline(gap_val, linestyle="dotted", color="black")
            # shade regions outside band gap
            ax.fill_betweenx([0, 12], -10, 0, alpha=0.2, color="lightgrey")
            ax.fill_betweenx([0, 12], gap_val, gap_val+10, alpha=0.2, color="lightgrey")
            ax.fill_between([0, gap_val], 0, 0, alpha=0.15, color="lightyellow")  # subtle band area anchor

            ax.set_xlim(-0.1, gap_val + 0.1)
            ax.set_ylim(0, 10)
            ax.set_xlabel("Fermi Level (eV)")
            ax.set_ylabel("Defect Formation Energy (eV)")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
            st.pyplot(fig, clear_figure=True)

            # Offer download of the PNG
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            st.download_button("Download plot (PNG)", data=buf.getvalue(), file_name=f"{compound_for_plot}_defects_{chem_pot_choice}.png")

        else:
            # Download optimized structures from Drive
            if not compounds:
                st.info("First, scan your Drive root (sidebar) so we know where to look for structures.")
            else:
                # Choose the Drive compound that matches table compound (default to same name)
                drive_comp_names = list(compounds.keys())
                if compound_for_plot in drive_comp_names:
                    comp_drive_choice = compound_for_plot
                else:
                    comp_drive_choice = st.selectbox("Pick the matching Drive compound", drive_comp_names)

                comp_id = compounds.get(comp_drive_choice)
                if not comp_id:
                    st.error("Drive compound not found. Scan the root or pick another name.")
                else:
                    # list defects available in Drive for this compound
                    drive_defects = discover_defects(comp_id)
                    st.markdown("#### Download structures per defect / charge state")
                    for dname in chosen_defects:
                        did = drive_defects.get(dname)
                        if not did:
                            st.info(f"Drive: defect folder '{dname}' not found.")
                            continue
                        charges = discover_charge_states(did)
                        if not charges:
                            st.info(f"{dname}: no charge-state subfolders.")
                            continue

                        st.markdown(f"**{dname}**")
                        # Let user pick which charge states to attempt
                        charge_labels = list(charges.keys())
                        default_pick = charge_labels  # default to all
                        picked = st.multiselect(f"Charge states for {dname}", charge_labels, default=default_pick, key=f"pick_{dname}")

                        for qlbl in picked:
                            qid = charges[qlbl]
                            blob, fname = find_structure_file(qid)
                            if blob is None:
                                st.write(f"• {qlbl}: _structure not found_")
                            else:
                                st.download_button(
                                    label=f"Download {dname} {qlbl} → {fname}",
                                    data=blob,
                                    file_name=f"{comp_drive_choice}_{dname}_{qlbl}_{fname}",
                                    key=f"dl_{dname}_{qlbl}"
                                )


# ─────────────────────────────────────────────────────────────────────────────
# HELP
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Help / Expected Folder Structure"):
    st.markdown(
        """
**Folder structure expected (Google Drive):**
- Root Folder (the ID you paste in the sidebar)
  - `Compound_A/`
    - `Bulk/`  ← (optional) contains `OUTCAR(.gz)`, `vasprun.xml(.gz)`, or `OSZICAR(.gz)`
    - `V_Cd/`
      - `q0/`, `q+1/`, `q-1/`, ... each with one of: OUTCAR(.gz), vasprun.xml(.gz), OSZICAR(.gz)
    - `Cl_Te/`
      - `q0/`, `q+1/`, `q-1/`, ...
  - `Compound_B/`
    - ...

**Energy parsing priority:** `OUTCAR` → `vasprun.xml` → `OSZICAR`.

**Plotting formula (per row / charge q):**  
`E_f(q, EF) = Toten_q - Toten_pure + μ + q * (EF + VBM) + Corr_q`  
The app plots the **lower envelope** across all available charge-states and rows for a chosen defect.
"""
    )
