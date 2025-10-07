# app.py
# -------------------------------------------------------------
# Defect Explorer (Google Drive) — Public Streamlit App (DefectDB-aware)
# Adds: Bulk energy reader (handles .gz), dynamic DB browser
#
# Requirements (requirements.txt):
#   streamlit
#   google-api-python-client
#   google-auth
#   google-auth-httplib2
#   pandas
#   numpy
#   matplotlib
#   openpyxl         # for reading .xlsx
#   pymatgen         # for DOS/OUTCAR from vasprun.xml/OUTCAR
#
# Secrets (.streamlit/secrets.toml):
# [gdrive_service_account]
# type = "service_account"
# project_id = "..."
# private_key_id = "..."
# private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
# client_email = "SERVICE_ACCOUNT@PROJECT.iam.gserviceaccount.com"
# client_id = "..."
# token_uri = "https://oauth2.googleapis.com/token"
# -------------------------------------------------------------

import io
import re
import gzip
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# ---------- Streamlit page & style ----------
st.set_page_config(page_title="Defect Explorer (Google Drive)", layout="wide")
plt.rcParams["font.family"] = "Arial Narrow"
plt.rcParams["font.size"] = 24

# ---------- Google Drive helpers ----------
@st.cache_resource(show_spinner=False)
def get_drive_service():
    """Authenticate once and reuse the Drive client (read-only)."""
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gdrive_service_account"]),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _list_children(folder_id: str, mime_filter: Optional[str] = None) -> List[Dict]:
    """List direct children of a Drive folder. Optionally filter on mimeType."""
    service = get_drive_service()
    q = f"'{folder_id}' in parents and trashed = false"
    if mime_filter:
        q += f" and mimeType {mime_filter}"
    files = []
    page_token = None
    while True:
        resp = service.files().list(
            q=q,
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType, parents, modifiedTime, size)",
            pageToken=page_token,
            pageSize=1000,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files

def _find_by_name_in_parent(name: str, parent_id: str, mime_prefix: Optional[str] = None) -> Optional[Dict]:
    """Find FIRST item with exact name under a parent (optionally by mimeType contains 'folder')."""
    service = get_drive_service()
    q = f"name = '{name}' and '{parent_id}' in parents and trashed = false"
    if mime_prefix:
        q += f" and mimeType contains '{mime_prefix}'"
    resp = service.files().list(
        q=q, spaces="drive", fields="files(id, name, mimeType, parents)"
    ).execute()
    files = resp.get("files", [])
    return files[0] if files else None

@st.cache_data(show_spinner=False)
def resolve_path_to_id(path: str) -> Optional[str]:
    """
    Resolve Drive-like path (e.g., 'DefectDB' or 'Materials/DefectDB') to a folder ID.
    Searches from 'My Drive' root.
    """
    service = get_drive_service()
    about = service.about().get(fields="rootFolderId").execute()
    current_id = about["rootFolderId"]
    parts = [p for p in path.strip("/").split("/") if p]
    for part in parts:
        hit = _find_by_name_in_parent(part, current_id, mime_prefix="folder")
        if not hit:
            return None
        current_id = hit["id"]
    return current_id

@st.cache_data(show_spinner=False)
def download_file_bytes(file_id: str) -> bytes:
    """Download a file by ID and return raw bytes."""
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()

def read_excel_from_drive(file_id: str) -> pd.DataFrame:
    """Read .xlsx from Drive into a DataFrame."""
    data = download_file_bytes(file_id)
    return pd.read_excel(io.BytesIO(data))

def read_csv_from_drive(file_id: str) -> pd.DataFrame:
    data = download_file_bytes(file_id)
    return pd.read_csv(io.BytesIO(data))

# ---------- Top-level discovery ----------
@st.cache_data(show_spinner=False)
def discover_compounds_and_vbm_gap(root_folder_id: str) -> Tuple[List[str], Dict[str, Dict], Optional[pd.DataFrame]]:
    """
    Under root (e.g., DefectDB), find:
      - compound folders (children folders)
      - vbm_gap.csv file (if present)
    Returns:
      compounds: sorted list of compound names
      map_compound_to_info: { name: {'id': folder_id} }
      vbm_gap_df: DataFrame for VBM & Bandgap or None
    """
    items = _list_children(root_folder_id)
    compounds = []
    cmap: Dict[str, Dict] = {}
    vbm_gap_df = None

    for it in items:
        if it["mimeType"] == "application/vnd.google-apps.folder":
            compounds.append(it["name"])
            cmap[it["name"]] = {"id": it["id"]}
        elif it["name"].lower() == "vbm_gap.csv":
            try:
                vbm_gap_df = read_csv_from_drive(it["id"])
            except Exception:
                vbm_gap_df = None

    compounds.sort()
    return compounds, cmap, vbm_gap_df

# ---------- Defect discovery ----------
def discover_defects_and_assets(composition_folder_id: str) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Scan composition folder for:
      - Defect folders (e.g., V_Cd)
      - Excel (first .xlsx)
      - Charge folders (e.g., q+2, q0, q-1) each with vasprun.xml
    """
    defect_index: Dict[str, Dict] = {}
    defects: List[str] = []

    children = _list_children(composition_folder_id, mime_filter="contains 'folder'")
    for f in children:
        defect_name = f["name"]
        defect_folder_id = f["id"]

        files = _list_children(defect_folder_id)
        excel_id = None
        for g in files:
            if g["mimeType"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                excel_id = g["id"]
                break

        charges: Dict[str, str] = {}
        subfolders = [x for x in files if x["mimeType"] == "application/vnd.google-apps.folder"]
        for sf in subfolders:
            label = sf["name"].strip()  # "q+2", "q0", "q-1", etc.
            cc = _list_children(sf["id"])
            for h in cc:
                if h["name"].lower() == "vasprun.xml" or h["name"].lower() == "vasprun.xml.gz":
                    charges[label] = h["id"]

        defect_index[defect_name] = {
            "folder_id": defect_folder_id,
            "excel_id": excel_id,
            "charges": charges,
        }
        defects.append(defect_name)

    return sorted(defects), defect_index

# ---------- Bulk discovery ----------
def discover_bulk_folder(comp_folder_id: str) -> Optional[str]:
    """Return the 'Bulk' subfolder id if present."""
    kids = _list_children(comp_folder_id, mime_filter="contains 'folder'")
    for k in kids:
        if k["name"].lower() == "bulk":
            return k["id"]
    return None

def pick_bulk_file_ids(bulk_folder_id: str) -> Dict[str, Optional[str]]:
    """
    Inside Bulk/, pick preferred files:
      - OUTCAR(.gz) preferred
      - vasprun.xml(.gz) fallback
      - OSZICAR(.gz) fallback
    """
    files = _list_children(bulk_folder_id)
    ids = {"outcar": None, "vasprun": None, "oszicar": None}
    for f in files:
        name = f["name"].lower()
        if name == "outcar" or name == "outcar.gz":
            ids["outcar"] = f["id"]
        elif name == "vasprun.xml" or name == "vasprun.xml.gz":
            ids["vasprun"] = f["id"]
        elif name == "oszicar" or name == "oszicar.gz":
            ids["oszicar"] = f["id"]
    return ids

# ---------- Utilities for gz + parsers ----------
def maybe_gunzip(name: str, data: bytes) -> bytes:
    """If filename ends with .gz, gunzip in memory; else return as-is."""
    if name.lower().endswith(".gz"):
        return gzip.decompress(data)
    return data

def parse_outcar_total_energy(raw_bytes: bytes) -> Optional[float]:
    """Use pymatgen Outcar if available; fallback to regex if needed."""
    try:
        from pymatgen.io.vasp.outputs import Outcar
        with tempfile.NamedTemporaryFile(delete=True, suffix=".OUTCAR") as tmp:
            tmp.write(raw_bytes); tmp.flush()
            out = Outcar(tmp.name)
            # Outcar.final_energy (free energy, usually E0)
            if hasattr(out, "final_energy") and out.final_energy is not None:
                return float(out.final_energy)
    except Exception:
        pass
    # regex fallback: last "energy  without entropy=" or "free  energy   TOTEN  =" lines
    text = raw_bytes.decode(errors="ignore")
    # Try common patterns
    m = None
    for pat in [
        r"free\s+energy\s+TOTEN\s*=\s*([-\d\.Ee+]+)",
        r"energy\s+without\s+entropy\s*=\s*([-\d\.Ee+]+)"
    ]:
        mm = list(re.finditer(pat, text))
        if mm:
            m = mm[-1]
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def parse_vasprun_total_energy(raw_bytes: bytes) -> Optional[float]:
    """Use pymatgen Vasprun.final_energy."""
    try:
        from pymatgen.io.vasp.outputs import Vasprun
        with tempfile.NamedTemporaryFile(delete=True, suffix=".xml") as tmp:
            tmp.write(raw_bytes); tmp.flush()
            vr = Vasprun(tmp.name, parse_dos=False, parse_eigen=False)
            if hasattr(vr, "final_energy") and vr.final_energy is not None:
                return float(vr.final_energy)
    except Exception:
        return None
    return None

def parse_oszicar_total_energy(raw_bytes: bytes) -> Optional[float]:
    """
    Parse last line with E0=... from OSZICAR.
    Example: "... E0=  -5.123456E+02  d E = ..."
    """
    text = raw_bytes.decode(errors="ignore")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Scan from bottom for E0=
    for line in reversed(lines):
        m = re.search(r"E0\s*=\s*([-\d\.Ee+]+)", line)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None

def get_bulk_total_energy(bulk_folder_id: str) -> Tuple[Optional[float], str]:
    """
    Return (energy, source) where source is 'OUTCAR'/'vasprun.xml'/'OSZICAR' describing which file was used.
    """
    picks = pick_bulk_file_ids(bulk_folder_id)
    # Try OUTCAR
    if picks["outcar"]:
        raw = download_file_bytes(picks["outcar"])
        # We don't know actual name (Drive strips), assume OUTCAR or OUTCAR.gz by trying gunzip first
        for name_try in ["OUTCAR.gz", "OUTCAR"]:
            data = maybe_gunzip(name_try, raw)
            e = parse_outcar_total_energy(data)
            if e is not None:
                return e, "OUTCAR"
    # Try vasprun.xml
    if picks["vasprun"]:
        raw = download_file_bytes(picks["vasprun"])
        for name_try in ["vasprun.xml.gz", "vasprun.xml"]:
            data = maybe_gunzip(name_try, raw)
            e = parse_vasprun_total_energy(data)
            if e is not None:
                return e, "vasprun.xml"
    # Try OSZICAR
    if picks["oszicar"]:
        raw = download_file_bytes(picks["oszicar"])
        for name_try in ["OSZICAR.gz", "OSZICAR"]:
            data = maybe_gunzip(name_try, raw)
            e = parse_oszicar_total_energy(data)
            if e is not None:
                return e, "OSZICAR"
    return None, "not_found"

# ---------- Plotting: formation energy (uses optional VBM/gap overrides) ----------
def plot_defect_formation_energy(
    df: pd.DataFrame,
    chem_pot: str,
    title: str,
    vbm_override: Optional[float] = None,
    gap_override: Optional[float] = None,
) -> bytes:
    gap = float(gap_override if gap_override is not None else df["gap"].iloc[0])

    EF = np.arange(-0.5, gap + 0.5, 0.01)
    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(left=0.14, bottom=0.14, right=0.70, top=0.90)
    plt.title(title, fontsize=24, ha="center", va="top", y=1.08)

    colors = [
        "red", "b", "c", "g", "mediumpurple", "darkorange", "saddlebrown", "m",
        "darkkhaki", "dodgerblue", "grey", "salmon", "limegreen", "gold", "navy",
        "deeppink", "olive", "teal", "firebrick", "indigo", "peru", "turquoise",
        "coral", "chartreuse", "slateblue", "tomato", "orchid", "royalblue", "seagreen"
    ]

    count = 0
    for i in range(len(df)):
        if chem_pot == "A-rich":
            mu = df["mu_A_rich"].iloc[i]
        elif chem_pot == "Medium":
            mu = df["mu_med"].iloc[i]
        else:
            mu = df["mu_Te_rich"].iloc[i]

        VBM = float(vbm_override) if vbm_override is not None else df["VBM"].iloc[i]

        Form_en = np.array([
            min(
                df["Toten_p2"].iloc[i]   - df["Toten_pure"].iloc[i] + mu + 2 * (ef + VBM) + df["Corr_p2"].iloc[i],
                df["Toten_p1"].iloc[i]   - df["Toten_pure"].iloc[i] + mu + 1 * (ef + VBM) + df["Corr_p1"].iloc[i],
                df["Toten_neut"].iloc[i] - df["Toten_pure"].iloc[i] + mu + 0 * (ef + VBM) + df["Corr_neut"].iloc[i],
                df["Toten_m1"].iloc[i]   - df["Toten_pure"].iloc[i] + mu - 1 * (ef + VBM) + df["Corr_m1"].iloc[i],
                df["Toten_m2"].iloc[i]   - df["Toten_pure"].iloc[i] + mu - 2 * (ef + VBM) + df["Corr_m2"].iloc[i],
            )
            for ef in EF
        ])

        if str(df["Plot"].iloc[i]).strip().upper() == "Y":
            plt.plot(EF, Form_en, c=colors[count % len(colors)], ls="solid", lw=4, label=str(df["Label"].iloc[i]))
            count += 1

    plt.axvline(x=0, linestyle="dotted", color="black")
    plt.axvline(x=gap, linestyle="dotted", color="black")
    plt.fill_between(EF, -4.2, 0, color="grey", alpha=0.7)

    x1 = np.arange(-10, 0.01, 0.01)
    x2 = np.arange(gap, 10.0, 0.01)
    x3 = np.arange(0.0, gap, 0.01)
    plt.fill_between(x1, -100, 100, facecolor="lightgrey", alpha=0.3)
    plt.fill_between(x2, -100, 100, facecolor="lightgrey", alpha=0.3)
    plt.fill_between(x3, -100, 100, facecolor="lightyellow", alpha=0.3)

    plt.xlabel("Fermi Level (eV)", fontsize=24, labelpad=8)
    plt.ylabel("Defect Formation Energy (eV)", fontsize=24)
    plt.xticks([0.0, np.round(gap / 2, 2), np.round(gap, 2)], fontsize=24)
    plt.yticks([0, 2, 4, 6, 8, 10], fontsize=24)
    plt.xlim([-0.2, gap + 0.2])
    plt.ylim([0, 10])
    plt.legend(loc="center left", bbox_to_anchor=[1.03, 0.5], ncol=1, frameon=True, prop={"size": 18})

    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches="tight", dpi=450, format="png")
    plt.close()
    return buf.getvalue()

# ---------- DOS plotting ----------
def read_complete_dos_from_vasprun_bytes(xml_bytes: bytes):
    """Parse a vasprun.xml (bytes) and return pymatgen CompleteDos, aligned to efermi later."""
    from pymatgen.io.vasp.outputs import Vasprun
    with tempfile.NamedTemporaryFile(delete=True, suffix=".xml") as tmp:
        tmp.write(xml_bytes)
        tmp.flush()
        vr = Vasprun(tmp.name, parse_dos=True, parse_eigen=True)
    return vr.get_complete_dos()

def plot_total_dos(dos, title: str) -> bytes:
    energies = np.array(dos.energies) - dos.efermi
    densities = np.array(dos.densities["total"])
    plt.figure(figsize=(6, 4))
    plt.plot(energies, densities, lw=2)
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Energy - $E_F$ (eV)")
    plt.ylabel("DOS (states/eV)")
    plt.title(title, y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, dpi=300, bbox_inches="tight", format="png")
    plt.close()
    return buf.getvalue()

def _charge_sort_key(label: str) -> int:
    s = label.lower().strip().replace("q", "")
    try:
        return int(s)
    except ValueError:
        return -999999

# ---------- UI ----------
st.title("Defect Explorer (Google Drive)")

with st.sidebar:
    st.header("Data Source")
    root_path = st.text_input("Drive path to root (DefectDB)", value="DefectDB", help="Example: 'DefectDB' or 'Materials/DefectDB'")
    chem_pot = st.selectbox("Chemical potential", ["A-rich", "Medium", "B-rich"], index=0)
    scan_root = st.button("Scan Root")

if scan_root:
    try:
        root_id = resolve_path_to_id(root_path)
        if not root_id:
            st.error(f"Could not resolve path: {root_path}. Did you share this folder with the service account?")
        else:
            st.success("Root folder resolved. Discovering compounds & vbm_gap.csv…")
            compounds, cmap, vbm_gap_df = discover_compounds_and_vbm_gap(root_id)

            if not compounds:
                st.warning("No compound folders found in this root folder.")
            else:
                # ----- Database view: show compounds + bulk presence -----
                db_rows = []
                for comp in compounds:
                    comp_id = cmap[comp]["id"]
                    bulk_id = discover_bulk_folder(comp_id)
                    bulk_status = "Yes" if bulk_id else "No"
                    db_rows.append({"Compound": comp, "Has Bulk": bulk_status})
                st.subheader("Database Overview")
                st.dataframe(pd.DataFrame(db_rows))

                colA, colB = st.columns([1, 2], gap="large")

                with colA:
                    compound_sel = st.selectbox("Choose a compound", compounds, index=0)
                    comp_id = cmap[compound_sel]["id"]
                    st.caption(f"Selected compound: **{compound_sel}**")

                    # Pull VBM & Bandgap from CSV if available
                    vbm_val = None
                    gap_val = None
                    if vbm_gap_df is not None and {"Compound", "VBM", "Bandgap"}.issubset(set(vbm_gap_df.columns)):
                        v = vbm_gap_df[vbm_gap_df["Compound"].astype(str).str.strip() == compound_sel]
                        if not v.empty:
                            try:
                                vbm_val = float(v["VBM"].iloc[0])
                                gap_val = float(v["Bandgap"].iloc[0])
                            except Exception:
                                vbm_val, gap_val = None, None

                    # Discover defects inside this compound folder
                    defects, index = discover_defects_and_assets(comp_id)
                    if defects:
                        st.markdown(f"**Defects found:** {', '.join(defects)}")
                        defect_sel = st.selectbox("Choose a defect", defects, index=0)

                        st.subheader("Formation Energy vs Fermi Level")
                        excel_id = index[defect_sel].get("excel_id")
                        if not excel_id:
                            st.info("No Excel file found (e.g., `defect_v2.xlsx`). Place an .xlsx in this defect folder.")
                        else:
                            try:
                                df = read_excel_from_drive(excel_id)
                                title = f"{compound_sel} ({chem_pot}) — {defect_sel}"
                                png = plot_defect_formation_energy(
                                    df, chem_pot, title,
                                    vbm_override=vbm_val,
                                    gap_override=gap_val
                                )
                                st.image(png, caption="Formation energy plot", use_column_width=True)
                                st.download_button(
                                    "Download plot (PNG)",
                                    png,
                                    file_name=f"{compound_sel}_{defect_sel}_formation_energy.png"
                                )
                                with st.expander("Raw table"):
                                    st.dataframe(df)
                            except Exception as e:
                                st.error(f"Error reading/plotting Excel: {e}")
                    else:
                        st.info("No defect folders found in this compound.")

                    # ----- Bulk energy UI -----
                    st.subheader("Bulk Energy")
                    bulk_id = discover_bulk_folder(comp_id)
                    if not bulk_id:
                        st.info("No `Bulk/` folder found inside this compound.")
                    else:
                        if st.button("Read Bulk Total Energy", key=f"read_bulk_{compound_sel}"):
                            try:
                                energy, source = get_bulk_total_energy(bulk_id)
                                if energy is None:
                                    st.error("Could not parse bulk energy from OUTCAR/vasprun.xml/OSZICAR.")
                                else:
                                    st.success(f"Bulk total energy: **{energy:.6f} eV**  (source: {source})")
                            except Exception as e:
                                st.error(f"Error reading bulk energy: {e}")

                with colB:
                    st.subheader("Density of States (per charge state)")
                    if 'index' in locals() and 'defect_sel' in locals() and defect_sel in index:
                        charges: Dict[str, str] = index[defect_sel].get("charges", {})
                        if not charges:
                            st.info("No charge-state subfolders with `vasprun.xml` found.")
                        else:
                            for qlabel in sorted(charges.keys(), key=_charge_sort_key, reverse=True):
                                file_id = charges[qlabel]
                                with st.expander(f"DOS: {qlabel}"):
                                    try:
                                        raw = download_file_bytes(file_id)
                                        # handle .gz transparently
                                        data = maybe_gunzip("vasprun.xml.gz", raw) if False else raw
                                        # try gunzip first; if fails, fall back to raw
                                        try:
                                            data = gzip.decompress(raw)
                                        except Exception:
                                            data = raw
                                        dos = read_complete_dos_from_vasprun_bytes(data)
                                        dos_png = plot_total_dos(dos, f"{compound_sel} — {defect_sel} — {qlabel}")
                                        st.image(dos_png, use_column_width=True)
                                        st.download_button(
                                            f"Download DOS plot ({qlabel})",
                                            dos_png,
                                            file_name=f"{compound_sel}_{defect_sel}_{qlabel}_DOS.png",
                                        )
                                    except HttpError as he:
                                        st.error(f"Drive download error for {qlabel}: {he}")
                                    except Exception as e:
                                        st.error(f"Failed to parse/plot DOS for {qlabel}: {e}")
                    else:
                        st.caption("Select a compound and defect to view DOS.")

    except HttpError as he:
        st.error(f"Google Drive API error: {he}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
