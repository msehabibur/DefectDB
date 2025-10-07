# app.py
# -------------------------------------------------------------
# Defect Explorer (Google Drive) — Public Streamlit App
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
#   pymatgen         # for DOS from vasprun.xml
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
#
# Drive layout example:
# CdTe/
#   V_Cd/
#     defect_v2.xlsx
#     q+2/vasprun.xml
#     q+1/vasprun.xml
#     q0/vasprun.xml
#     q-1/vasprun.xml
#     q-2/vasprun.xml
#   V_Te/
#     ...
# -------------------------------------------------------------

import io
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload  # noqa: F401 (upload unused, but useful if you extend)

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
    Resolve Drive-like path (e.g., 'CdTe' or 'Materials/CdTe') to a folder ID.
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

# ---------- Domain discovery ----------
def discover_defects_and_assets(composition_folder_id: str) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Scan composition folder (e.g., CdTe) for:
      - Defect folders (e.g., V_Cd)
      - Excel (first .xlsx)
      - Charge folders (e.g., q+2, q0, q-1) each with vasprun.xml

    Returns:
      defects: sorted list of defect names
      index: {
        defect_name: {
          'folder_id': ...,
          'excel_id': ... or None,
          'charges': { 'q+2': file_id, 'q+1': file_id, 'q0': file_id, ... }
        }, ...
      }
    """
    defect_index: Dict[str, Dict] = {}
    defects: List[str] = []

    # list child folders (defects)
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
            # (Optional) add Google Sheets export logic here if needed

        charges: Dict[str, str] = {}
        subfolders = [x for x in files if x["mimeType"] == "application/vnd.google-apps.folder"]
        for sf in subfolders:
            label = sf["name"].strip()  # "q+2", "q0", "q-1", etc.
            # look inside each charge folder for vasprun.xml
            cc = _list_children(sf["id"])
            for h in cc:
                if h["name"].lower() == "vasprun.xml":
                    charges[label] = h["id"]

        defect_index[defect_name] = {
            "folder_id": defect_folder_id,
            "excel_id": excel_id,
            "charges": charges,
        }
        defects.append(defect_name)

    return sorted(defects), defect_index

# ---------- Formation energy plotting ----------
def plot_defect_formation_energy(df: pd.DataFrame, chem_pot: str, title: str) -> bytes:
    """
    Plots formation energy envelope vs Fermi level using the schema from user's code.
    Expected columns include:
      ['gap','mu_A_rich','mu_med','mu_Te_rich','Toten_p2','Toten_p1','Toten_neut',
       'Toten_m1','Toten_m2','Toten_pure','Corr_p2','Corr_p1','Corr_neut','Corr_m1',
       'Corr_m2','VBM','Defect','Label','Plot']
    """
    gap = float(df["gap"].iloc[0])

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
            # In your script, B-rich used mu_Te_rich
            mu = df["mu_Te_rich"].iloc[i]

        VBM = df["VBM"].iloc[i]

        # envelope across charge states (+2, +1, 0, -1, -2)
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

    # band edges & shading
    plt.axvline(x=0, linestyle="dotted", color="black")
    plt.axvline(x=gap, linestyle="dotted", color="black")

    # forbidden region (below 0 eV formation)
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
    """
    Normalize labels like 'q+2', 'q-1', 'q0' -> integer +2, -1, 0 for sorting.
    If already like '+2', '-1', '0', it also works.
    """
    s = label.lower().strip()
    s = s.replace("q", "")
    try:
        return int(s)
    except ValueError:
        # fallback: unknown label goes to end
        return -999999

# ---------- UI ----------
st.title("Defect Explorer (Google Drive)")

with st.sidebar:
    st.header("Data Source")
    drive_path = st.text_input("Drive path to composition folder", value="CdTe", help="Example: 'CdTe' or 'Materials/CdTe'")
    chem_pot = st.selectbox("Chemical potential", ["A-rich", "Medium", "B-rich"], index=0)
    scan = st.button("Scan Folder")

if scan:
    try:
        comp_id = resolve_path_to_id(drive_path)
        if not comp_id:
            st.error(f"Could not resolve path: {drive_path}. Did you share this folder with the service account?")
        else:
            st.success("Folder resolved. Discovering defects…")
            defects, index = discover_defects_and_assets(comp_id)

            if not defects:
                st.warning("No defect folders found in this composition folder.")
            else:
                st.write(f"**Found defects:** {', '.join(defects)}")

                col_left, col_right = st.columns([1, 2], gap="large")

                with col_left:
                    defect_sel = st.selectbox("Choose a defect", defects, index=0)

                    st.subheader("Formation Energy vs Fermi Level")
                    excel_id = index[defect_sel].get("excel_id")
                    if not excel_id:
                        st.info("No Excel file found (e.g., `defect_v2.xlsx`). Place an .xlsx in this defect folder.")
                    else:
                        try:
                            df = read_excel_from_drive(excel_id)
                            title = f"{drive_path} ({chem_pot}) — {defect_sel}"
                            png = plot_defect_formation_energy(df, chem_pot, title)
                            st.image(png, caption="Formation energy plot", use_column_width=True)
                            st.download_button("Download plot (PNG)", png, file_name=f"{defect_sel}_formation_energy.png")
                            with st.expander("Raw table"):
                                st.dataframe(df)
                        except Exception as e:
                            st.error(f"Error reading/plotting Excel: {e}")

                with col_right:
                    st.subheader("Density of States (per charge state)")
                    charges: Dict[str, str] = index[defect_sel].get("charges", {})
                    if not charges:
                        st.info("No charge-state subfolders with `vasprun.xml` found.")
                    else:
                        for qlabel in sorted(charges.keys(), key=_charge_sort_key, reverse=True):
                            file_id = charges[qlabel]
                            with st.expander(f"DOS: {qlabel}"):
                                try:
                                    xml_bytes = download_file_bytes(file_id)
                                    dos = read_complete_dos_from_vasprun_bytes(xml_bytes)
                                    dos_png = plot_total_dos(dos, f"{defect_sel} — {qlabel}")
                                    st.image(dos_png, use_column_width=True)
                                    st.download_button(
                                        f"Download DOS plot ({qlabel})",
                                        dos_png,
                                        file_name=f"{defect_sel}_{qlabel}_DOS.png",
                                    )
                                except HttpError as he:
                                    st.error(f"Drive download error for {qlabel}: {he}")
                                except Exception as e:
                                    st.error(f"Failed to parse/plot DOS for {qlabel}: {e}")

    except HttpError as he:
        st.error(f"Google Drive API error: {he}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

- Folder structure expected:
