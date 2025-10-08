# app.py — DefectDB Browser (Drive) + Auto-built Defect Analysis
#
# FIXED: Charge mapping is STANDARD now:
#   Charged+N -> pN;  Charged-N -> mN;  Charged0/Neutral -> neut
#   So Toten_m2 comes from 'Charged-2', Toten_p2 from 'Charged+2', etc.
#
# WHAT THIS APP DOES
# - Browse Google Drive DefectDB: compounds → defects → charge-state folders
# - Read energies from OUTCAR(.gz) → vasprun.xml(.gz) → OSZICAR(.gz)
# - Load μ from ROOT/data.csv even if it ONLY has columns like:
#     V_Cd (Cd-rich), V_Cd (Te-rich), As_Te (Cd-rich), As_Te (Te-rich), Cl_Te (Cd-rich), Cl_Te (Te-rich)
#   (No Compound / VBM / Bandgap required; defaults used if missing)
# - Build a table with Toten_p2, Toten_p1, Toten_neut, Toten_m1, Toten_m2 (Corr_* = 0.0)
# - Plot defect formation energy vs Fermi level (lower envelope)
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
# .streamlit/secrets.toml must contain [gdrive_service_account] (service account JSON).

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
from googleapiclient.http import MediaIoBaseDownload

# ── Config & SSL ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="DefectDB Browser (Drive) + Analysis", layout="wide")
httplib2.CA_CERTS = certifi.where()
ssl.create_default_context(cafile=certifi.where())

ROOT_FOLDER_ID_DEFAULT = "1gYTtFpPIRCDWpLBW855RA6XwG0buifbi"

# Defaults if VBM/Bandgap not given anywhere
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
    """
    STANDARD mapping:
      'Charged+N' -> +N ; 'Charged-N' -> -N ; 'Charged0' -> 0
      'Neutral'/'neut'/'q0'/'0' -> 0
      'pN' -> +N ; 'mN' -> -N
      'q+N'/'q-N'/'q0' also supported
      plain '+1'/'-2'/'0' also supported
    """
    s = (label or "").strip().lower()

    # explicit neutral words
    if s in {"neutral", "neut"}:
        return 0

    # pN / mN
    m = re.match(r'^[pm]\s*(\d+)$', s)
    if m:
        n = int(m.group(1))
        return +n if s.startswith('p') else -n

    # q±N or qN
    m = re.match(r'^q\s*([+\-]?\d+)$', s)
    if m:
        return int(m.group(1))

    # plain ±N or 0
    m = re.match(r'^[+\-]?\d+$', s)
    if m:
        return int(s)

    # charged±N or charge±N or charged0
    m = re.match(r'^(charged|charge)\s*([+\-]?)\s*(\d+)$', s)
    if m:
        sign = m.group(2) or "0"
        n = int(m.group(3))
        if n == 0:
            return 0
        # STANDARD: '+' means +N; '-' means -N
        return +n if sign == '+' else -n

    # fallback: any integer in string
    m = re.search(r'([+\-]?\d+)', s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
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
    """
    Returns (VBM, gap, row_with_mu_columns_or_None).
    - If df has 'Compound' row for this compound, uses that row; tries 'VBM (eV)'/'Bandgap (eV)'.
    - Else treats df as GLOBAL μ table (no 'Compound') and returns first row; VBM/gap default.
    """
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
        # fallthrough to global
    row = params_df.iloc[0]
    return vbm, gap, row

def mu_from_params_row(row: Optional[pd.Series], defect_name: str, chem_pot: str) -> Optional[float]:
    """
    Works with μ-only CSV:
      'V_Cd (Cd-rich)', 'V_Cd (Te-rich)', ...
    """
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
    """
    Returns DataFrame columns:
      Compound, Defect, Type, Plot, Label, gap, VBM, Toten_pure,
      Toten_p2, Toten_p1, Toten_neut, Toten_m1, Toten_m2,
      Corr_p2, Corr_p1, Corr_neut, Corr_m1, Corr_m2, mu
    """
    vbm, gap, params_row = get_params_row_for_compound(root_params, compound)

    # Bulk energy
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
            if q is None:
                continue
            if q == +2: vals["Toten_p2"] = e
            elif q == +1: vals["Toten_p1"] = e
            elif q == 0:  vals["Toten_neut"] = e
            elif q == -1: vals["Toten_m1"] = e
            elif q == -2: vals["Toten_m2"] = e
            # ignore |q| > 2

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

# ── Plotting ─────────────────────────────────────────────────────────────────
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
    for q, tcol, ccol in [
        (+2,"Toten_p2","Corr_p2"),
        (+1,"Toten_p1","Corr_p1"),
        ( 0,"Toten_neut","Corr_neut"),
        (-1,"Toten_m1","Corr_m1"),
        (-2,"Toten_m2","Corr_m2"),
    ]:
        tq = _coerce_float(row.get(tcol)); cq = _coerce_float(row.get(ccol))
        if not np.isnan(tq) and not np.isnan(cq):
            out.append((q, tq, cq))
    return out

def plot_formation_energy(df_sub: pd.DataFrame, title: str):
    plt.rcParams["font.size"] = 22
    fig, ax = plt.subplots(figsize=(7,6))
    if df_sub.empty:
        ax.text(0.5,0.5,"No data", ha="center", va="center", transform=ax.transAxes)
        st.pyplot(fig, clear_figure=True); return

    gap = _coerce_float(df_sub["gap"].iloc[0]); gap = DEFAULT_GAP if (np.isnan(gap) or gap<=0) else gap
    vbm = _coerce_float(df_sub["VBM"].iloc[0]); vbm = DEFAULT_VBM if np.isnan(vbm) else vbm
    mu  = _coerce_float(df_sub["mu"].iloc[0]);  mu  = 0.0 if np.isnan(mu) else mu

    EF = np.arange(0.0, gap+1e-9, 0.01)

    for _, r in df_sub.iterrows():
        charges = _available_charges(r)
        if not charges: continue
        toten_pure = _coerce_float(r.get("Toten_pure"))
        if np.isnan(toten_pure): continue
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
            comps = discover_compounds(root_id)
            if not comps:
                st.error("No compound folders found (or missing permissions)."); st.stop()
            st.session_state["compounds"] = dict(sorted(comps.items(), key=lambda x: x[0].lower()))
            st.session_state["root_params"] = load_root_params(root_id)
            if st.session_state["root_params"] is None:
                st.warning("ROOT/data.csv not found — using defaults (VBM=0, gap=1.5) and μ=None.")
            else:
                st.success("Loaded ROOT/data.csv.")
            st.success(f"Found {len(comps)} compound folder(s).")
        except Exception as e:
            st.error(f"Error: {e}")

compounds = st.session_state.get("compounds")
root_params = st.session_state.get("root_params")

if compounds:
    st.subheader("📦 Compounds")
    overview = [{"Compound": c, "Has Bulk": "Yes" if find_bulk_folder(cid) else "No"} for c, cid in compounds.items()]
    st.dataframe(pd.DataFrame(overview), width="stretch")

    comp_sel = st.selectbox("Select a compound", list(compounds.keys()))
    comp_id = compounds[comp_sel]

    defects = discover_defects(comp_id)
    defect_names = sorted(defects.keys())
    st.markdown("### 🧬 Choose defects to analyze")
    chosen_defects = st.multiselect("Defects", defect_names, default=defect_names)

    chem_pot_choice = st.selectbox("Chemical potential set", ["Cd-rich", "Te-rich"])

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Build table & plot"):
            with st.spinner("Building table from Drive + data.csv..."):
                try:
                    table = build_correction_table_for_compound(
                        root_id, comp_sel, comp_id, chem_pot_choice, root_params,
                        restrict_defects=chosen_defects
                    )
                    if table.empty:
                        st.warning("No rows produced — check folder contents.")
                    else:
                        st.success("Constructed correction/energy table.")
                        st.dataframe(table, width="stretch")
                        st.download_button(
                            "Download table (CSV)",
                            table.to_csv(index=False).encode(),
                            file_name=f"{comp_sel}_corrections.csv"
                        )
                        sub = table[table["Defect"].isin(chosen_defects)].copy()
                        plot_formation_energy(sub, title=f"{comp_sel}_{chem_pot_choice}_formation_plot")
                except Exception as e:
                    st.error(f"Failed to build/plot: {e}")

    with colB:
        st.markdown("#### 📥 Download optimized structures")
        if st.button("List & download structures"):
            drive_defects = discover_defects(comp_id)
            for dname in chosen_defects:
                did = drive_defects.get(dname)
                if not did:
                    st.info(f"Drive: defect folder '{dname}' not found."); continue
                charges = discover_charge_states(did)
                if not charges:
                    st.info(f"{dname}: no charge-state subfolders."); continue
                # sort by q (desc)
                items = []
                for qlbl, qid in charges.items():
                    q = parse_charge_label_to_q(qlbl)
                    items.append((q, qlbl, qid))
                items.sort(key=lambda x: (x[0] is None, -(x[0] or 0)))
                st.markdown(f"**{dname}**")
                for q, qlbl, qid in items:
                    blob, fname = find_structure_file(qid)
                    if blob is None:
                        st.write(f"• {qlbl}: _structure not found_")
                    else:
                        st.download_button(
                            label=f"Download {dname} {qlbl} → {fname}",
                            data=blob,
                            file_name=f"{comp_sel}_{dname}_{qlbl}_{fname}",
                            key=f"dl_{dname}_{qlbl}"
                        )

with st.expander("ℹ️ Notes"):
    st.markdown(
        f"""
**Charge mapping (fixed):**  
`Charged+2 → Toten_p2`, `Charged+1 → Toten_p1`, `Charged0/Neutral → Toten_neut`, `Charged-1 → Toten_m1`, `Charged-2 → Toten_m2`.

**Global μ CSV supported:**  
If `data.csv` only has μ columns like `V_Cd (Cd-rich)`, `V_Cd (Te-rich)`, `As_Te (Cd-rich)`, `As_Te (Te-rich)`, `Cl_Te (Cd-rich)`, `Cl_Te (Te-rich)`, the app uses those for **all compounds**.  
Defaults: VBM={DEFAULT_VBM}, Bandgap={DEFAULT_GAP}.

**Formation energy:**  
`E_f(q, EF) = Toten_q − Toten_pure + μ + q·(EF + VBM) + Corr_q` (Corr_q defaults to 0.0).
"""
    )
