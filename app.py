import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from typing import Dict, List, Optional, Tuple
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ---------- Streamlit page config ----------
st.set_page_config(page_title="Defect Explorer (Drive)", layout="wide")

# Global plot style to match your preferences
plt.rcParams["font.family"] = "Arial Narrow"
plt.rcParams["font.size"] = 24

# ---------- Google Drive helpers ----------
@st.cache_resource(show_spinner=False)
def get_drive_service():
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gdrive_service_account"]),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _list_children(folder_id: str, mime_filter: Optional[str] = None) -> List[Dict]:
    """List children of a Drive folder (1 level)."""
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
    """Find a single file/folder by exact name under a parent."""
    service = get_drive_service()
    q = f"name = '{name}' and '{parent_id}' in parents and trashed = false"
    if mime_prefix:
        q += f" and mimeType contains '{mime_prefix}'"
    resp = service.files().list(q=q, spaces="drive", fields="files(id, name, mimeType, parents)").execute()
    files = resp.get("files", [])
    return files[0] if files else None

@st.cache_data(show_spinner=False)
def resolve_path_to_id(path: str) -> Optional[str]:
    """
    Resolve a Drive-like path (e.g., 'CdTe' or 'Materials/CdTe') to a folder ID.
    Searches from 'My Drive' root.
    """
    service = get_drive_service()
    # Get 'root' folder id
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
    """Download a file from Drive and return bytes."""
    service = get_drive_service()
    req = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = googleapiclient.http.MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    return fh.getvalue()

def read_excel_from_drive(file_id: str) -> pd.DataFrame:
    data = download_file_bytes(file_id)
    return pd.read_excel(io.BytesIO(data))

# ---------- Domain discovery helpers ----------
def discover_defects_and_assets(composition_folder_id: str) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Scans the composition folder (e.g., CdTe) for defects and assets:
      composition/
        V_Cd/                 <-- defect folder
          (optional) defect_v2.xlsx
          q+2/ vasprun.xml
          q+1/ vasprun.xml
          q0/  vasprun.xml
          q-1/ vasprun.xml
          q-2/ vasprun.xml
    Returns:
      defects: list of defect names
      index: { defect_name: { 'folder_id':..., 'excel_id':..., 'charges': { '+2': file_id, ... } } }
    """
    defect_index: Dict[str, Dict] = {}
    defects: List[str] = []

    # list defect folders
    children = _list_children(composition_folder_id, mime_filter="contains 'folder'")
    for f in children:
        defect_name = f["name"]
        defect_folder_id = f["id"]

        # Excel file (first matching *.xlsx)
        files = _list_children(defect_folder_id)
        excel_id = None
        for g in files:
            if g["mimeType"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                excel_id = g["id"]
                break
            # also allow Google Sheets export (optional)
            if g["mimeType"] == "application/vnd.google-apps.spreadsheet":
                # you could export sheets via exportMedia; keeping to .xlsx for simplicity
                pass

        # charge subfolders and vasprun.xml
        charges = {}
        subfolders = [x for x in files if x["mimeType"] == "application/vnd.google-apps.folder"]
        for sf in subfolders:
            charge_label = sf["name"].strip()  # e.g., q+2, q0, q-1
            cc = _list_children(sf["id"])
            for h in cc:
                if h["name"].lower() == "vasprun.xml":
                    # save file id
                    charges[charge_label] = h["id"]

        defect_index[defect_name] = {
            "folder_id": defect_folder_id,
            "excel_id": excel_id,
            "charges": charges,
        }
        defects.append(defect_name)

    defects.sort()
    return defects, defect_index

# ---------- Plotting: Defect formation energy vs Fermi level ----------
def plot_defect_formation_energy(df: pd.DataFrame, chem_pot: str, title: str) -> bytes:
    # Expect the same columns you used in your code:
    # ['gap','mu_A_rich','mu_med','mu_Te_rich','Toten_p2','Toten_p1','Toten_neut','Toten_m1','Toten_m2',
    #  'Toten_pure','Corr_p2','Corr_p1','Corr_neut','Corr_m1','Corr_m2','VBM','Defect','Label','Plot']
    gap = float(df['gap'].iloc[0])

    EF = np.arange(-0.5, gap + 0.5, 0.01)
    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(left=0.14, bottom=0.14, right=0.70, top=0.90)
    plt.title(title, fontsize=24, ha='center', va='top', y=1.08)

    colors = [
        'red', 'b', 'c', 'g', 'mediumpurple', 'darkorange', 'saddlebrown', 'm',
        'darkkhaki', 'dodgerblue', 'grey', 'salmon', 'limegreen', 'gold', 'navy',
        'deeppink', 'olive', 'teal', 'firebrick', 'indigo', 'peru', 'turquoise',
        'coral', 'chartreuse', 'slateblue', 'tomato', 'orchid', 'royalblue', 'seagreen'
    ]

    count = 0
    for i in range(len(df)):
        if chem_pot == 'A-rich':
            mu = df['mu_A_rich'].iloc[i]
        elif chem_pot == 'Medium':
            mu = df['mu_med'].iloc[i]
        else:
            mu = df['mu_Te_rich'].iloc[i]

        VBM = df['VBM'].iloc[i]
        # Compute formation energy envelope across charge states
        Form_en = np.array([min(
            df['Toten_p2'].iloc[i]  - df['Toten_pure'].iloc[i] + mu + 2 * (ef + VBM) + df['Corr_p2'].iloc[i],
            df['Toten_p1'].iloc[i]  - df['Toten_pure'].iloc[i] + mu + 1 * (ef + VBM) + df['Corr_p1'].iloc[i],
            df['Toten_neut'].iloc[i]- df['Toten_pure'].iloc[i] + mu + 0 * (ef + VBM) + df['Corr_neut'].iloc[i],
            df['Toten_m1'].iloc[i]  - df['Toten_pure'].iloc[i] + mu - 1 * (ef + VBM) + df['Corr_m1'].iloc[i],
            df['Toten_m2'].iloc[i]  - df['Toten_pure'].iloc[i] + mu - 2 * (ef + VBM) + df['Corr_m2'].iloc[i]
        ) for ef in EF])

        if str(df['Plot'].iloc[i]).strip().upper() == 'Y':
            linestyle = 'solid'
            plt.plot(EF, Form_en, c=colors[count % len(colors)], ls=linestyle, lw=4, label=str(df['Label'].iloc[i]))
            count += 1

    # band edges, shading
    plt.axvline(x=0, linestyle='dotted', color='black')
    plt.axvline(x=gap, linestyle='dotted', color='black')
    plt.fill_between(EF, -4.2, 0, color='grey', alpha=0.7)

    x1 = np.arange(-10, 0.01, 0.01); plt.fill_between(x1, -100, 100, facecolor='lightgrey', alpha=0.3)
    x2 = np.arange(gap, 10.0, 0.01); plt.fill_between(x2, -100, 100, facecolor='lightgrey', alpha=0.3)
    x3 = np.arange(0.0, gap, 0.01);  plt.fill_between(x3, -100, 100, facecolor='lightyellow', alpha=0.3)

    plt.xlabel('Fermi Level (eV)', fontsize=24, labelpad=8)
    plt.ylabel('Defect Formation Energy (eV)', fontsize=24)
    plt.xticks([0.0, np.round(gap/2, 2), np.round(gap, 2)], fontsize=24)
    plt.yticks([0, 2, 4, 6, 8, 10], fontsize=24)
    plt.xlim([-0.2, gap + 0.2])
    plt.ylim([0, 10])
    plt.legend(loc='center left', bbox_to_anchor=[1.03, 0.5], ncol=1, frameon=True, prop={'size': 18})

    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', dpi=450, format='png')
    plt.close()
    return buf.getvalue()

# ---------- DOS plotting ----------
def read_complete_dos_from_vasprun_bytes(xml_bytes: bytes):
    # Lazy import pymatgen (heavy)
    from pymatgen.io.vasp.outputs import Vasprun
    fh = io.BytesIO(xml_bytes)
    # Vasprun can read from a file-like only if we write it to a temp file; fallback to temp.
    # Simpler approach: write to a BytesIO-backed temporary file on disk.
    import tempfile
    with tempfile.NamedTemporaryFile(delete=True, suffix=".xml") as tmp:
        tmp.write(xml_bytes); tmp.flush()
        vr = Vasprun(tmp.name, parse_dos=True, parse_eigen=True)
    return vr.get_complete_dos()

def plot_total_dos(dos, title: str) -> bytes:
    energies = np.array(dos.energies) - dos.efermi  # align E_F to 0
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

# ---------- UI ----------
st.title("Defect Explorer (Google Drive)")

with st.sidebar:
    st.header("Data Source")
    drive_path = st.text_input("Drive path to composition folder", value="CdTe")
    chem_pot = st.selectbox("Chemical potential condition", ["A-rich", "Medium", "B-rich"], index=0)
    st.caption("Path example: `CdTe` or `Materials/CdTe`")

    go = st.button("Scan Folder")

if go:
    try:
        comp_id = resolve_path_to_id(drive_path)
        if not comp_id:
            st.error(f"Could not resolve path: {drive_path}")
        else:
            st.success("Folder resolved. Discovering defects...")
            defects, index = discover_defects_and_assets(comp_id)

            if not defects:
                st.warning("No defect folders found.")
            else:
                st.write(f"**Found defects:** {', '.join(defects)}")

                col1, col2 = st.columns([1, 2], gap="large")

                with col1:
                    defect_sel = st.selectbox("Choose a defect", defects, index=0)

                    # Formation energy plot
                    st.subheader("Defect Formation Energy vs Fermi Level")
                    info = index[defect_sel]
                    excel_id = info.get("excel_id")
                    if not excel_id:
                        st.info("No Excel file found (e.g., `defect_v2.xlsx`). Place an .xlsx in this defect folder.")
                    else:
                        try:
                            df = read_excel_from_drive(excel_id)
                            title = f"{drive_path} ({chem_pot}) — {defect_sel}"
                            png = plot_defect_formation_energy(df, chem_pot, title)
                            st.image(png, caption="Formation energy plot", use_column_width=True)
                            st.download_button("Download plot (PNG)", png, file_name=f"{defect_sel}_formation_energy.png")
                            st.dataframe(df)  # show the raw table for transparency
                        except Exception as e:
                            st.error(f"Error reading/plotting Excel: {e}")

                with col2:
                    st.subheader("Density of States (per charge state)")
                    charges: Dict[str, str] = index[defect_sel].get("charges", {})
                    if not charges:
                        st.info("No charge-state subfolders with `vasprun.xml` found.")
                    else:
                        # Sort charge labels in canonical order
                        def _sort_key(q):
                            # normalize like q+2 -> +2, q-1 -> -1, q0 -> 0
                            s = q.lower().replace("q", "")
                            return int(s)
                        for qlabel in sorted(charges.keys(), key=_sort_key, reverse=True):
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
                                        file_name=f"{defect_sel}_{qlabel}_DOS.png"
                                    )
                                except HttpError as he:
                                    st.error(f"Drive download error for {qlabel}: {he}")
                                except Exception as e:
                                    st.error(f"Failed to parse/plot DOS for {qlabel}: {e}")

    except HttpError as he:
        st.error(f"Google Drive API error: {he}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# Helpful notes
st.markdown("""
**Notes**
- Organize your Drive like:
