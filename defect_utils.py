import io
import os
import ssl
import time
import httplib2
import pandas as pd
import numpy as np
from tqdm import tqdm
from rich import print
from uncertainties import ufloat
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# ─── APP DEFAULTS ─────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
ROOT_FOLDER_ID_DEFAULT = "1gYTtFpPIRCDWpLBW855RA6XwG0buifbi"  # ✅ your Drive root folder ID

# ──────────────────────────────────────────────────────────────
# ─── GOOGLE DRIVE SETUP ───────────────────────────────────────
# ──────────────────────────────────────────────────────────────
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2 import service_account
except ImportError:
    build = MediaIoBaseDownload = service_account = None
    print("[yellow]⚠️ Google API libraries not found. Running in offline mode.[/yellow]")

SERVICE_ACCOUNT_FILE = "service_account.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

drive_service = None
try:
    if service_account and os.path.exists(SERVICE_ACCOUNT_FILE):
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        drive_service = build("drive", "v3", credentials=credentials)
        print("[green]✅ Google Drive service initialized successfully.[/green]")
    else:
        print("[red]❌ service_account.json not found — please upload it.[/red]")
except Exception as e:
    print(f"[red]⚠️ Could not initialize Google Drive API: {e}[/red]")

# ──────────────────────────────────────────────────────────────
# ─── RETRY HELPER ─────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
def _with_retries(fn, *, tries: int = 3, base_delay: float = 0.8):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if attempt == tries - 1:
                raise
            wait_time = base_delay * (2 ** attempt)
            print(f"[yellow]Retry {attempt+1}/{tries}: {e} → waiting {wait_time:.1f}s[/yellow]")
            time.sleep(wait_time)

# ──────────────────────────────────────────────────────────────
# ─── DOWNLOAD STRUCTURE FROM DRIVE ────────────────────────────
# ──────────────────────────────────────────────────────────────
def download_bytes(file_id):
    """Download a file from Google Drive using secure HTTPS client."""
    if drive_service is None:
        print("[yellow]⚠️ Skipping Google Drive download (no credentials).[/yellow]")
        return b"", "mock_structure.cif"

    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("HTTP_PROXY", None)

    ssl_context = ssl.create_default_context()
    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2

    http = httplib2.Http(timeout=60, ca_certs=ssl.get_default_verify_paths().cafile)
    http.disable_ssl_certificate_validation = False

    request = drive_service.files().get_media(fileId=file_id)
    file_handle = io.BytesIO()
    downloader = MediaIoBaseDownload(file_handle, request)

    done = False
    while not done:
        def _do():
            return downloader.next_chunk(http=http)
        _, done = _with_retries(_do)

    return file_handle.getvalue()

# ──────────────────────────────────────────────────────────────
# ─── DISCOVERY HELPERS (RETURN DICTS) ─────────────────────────
# ──────────────────────────────────────────────────────────────
def discover_compounds(root_folder_id: Optional[str] = None) -> Dict[str, str]:
    """Return dict of compound name → folder ID."""
    if drive_service is None:
        return {
            "CdTe": "mock_id_CdTe",
            "CdSeTe": "mock_id_CdSeTe",
            "CdZnTe": "mock_id_CdZnTe"
        }

    folder_id = root_folder_id or ROOT_FOLDER_ID_DEFAULT
    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id, name)",
    ).execute()
    return {f["name"]: f["id"] for f in results.get("files", [])}


def discover_defects(compound_folder_id: Optional[str] = None) -> Dict[str, str]:
    """Return dict of defect name → folder ID."""
    if drive_service is None:
        return {
            "V_Cd": "mock_id_V_Cd",
            "V_Te": "mock_id_V_Te",
            "As_Te": "mock_id_As_Te",
            "Cl_Te": "mock_id_Cl_Te"
        }

    results = drive_service.files().list(
        q=f"'{compound_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id, name)",
    ).execute()
    return {f["name"]: f["id"] for f in results.get("files", [])}


def discover_charge_states(defect_folder_id: Optional[str] = None) -> Dict[str, str]:
    """Return dict of charge label → folder ID."""
    if drive_service is None:
        return {
            "q=0": "mock_id_q0",
            "q=+1": "mock_id_q+1",
            "q=-1": "mock_id_q-1",
            "q=+2": "mock_id_q+2",
            "q=-2": "mock_id_q-2",
        }

    results = drive_service.files().list(
        q=f"'{defect_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id, name)",
    ).execute()
    return {f["name"]: f["id"] for f in results.get("files", [])}

# ──────────────────────────────────────────────────────────────
# ─── PARSE CHARGE LABEL → INTEGER ─────────────────────────────
# ──────────────────────────────────────────────────────────────
def parse_charge_label_to_q(label: str) -> Optional[int]:
    """Parse folder/label like 'q=+1', 'charge_minus_2', 'q0' → integer."""
    label = label.lower().strip()
    if label.startswith("q="):
        try:
            return int(label.replace("q=", "").replace("+", ""))
        except ValueError:
            return None
    if "plus" in label:
        try:
            return int(label.split("plus_")[-1])
        except ValueError:
            return None
    if "minus" in label:
        try:
            return -int(label.split("minus_")[-1])
        except ValueError:
            return None
    if label in ["q0", "charge0", "neutral"]:
        return 0
    return None

# ──────────────────────────────────────────────────────────────
# ─── FIND STRUCTURE FILE ──────────────────────────────────────
# ──────────────────────────────────────────────────────────────
STRUCTURE_FILE_PRIORITY = [
    "CONTCAR", "POSCAR", "Relaxed.cif", "Final.cif", "structure.cif", "optimized.cif",
]

def find_structure_file(folder_id: str):
    """Find preferred structure file in a Drive folder."""
    if drive_service is None:
        print(f"[yellow]⚠️ Skipping structure lookup for folder_id={folder_id} (mock mode).[/yellow]")
        return b"", "mock_structure.cif"

    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name)"
    ).execute()

    files = results.get("files", [])
    files_by_name = {f["name"]: f for f in files}

    for candidate in STRUCTURE_FILE_PRIORITY:
        for name, meta in files_by_name.items():
            if name.lower() == candidate.lower():
                return download_bytes(meta["id"]), name

    if files_by_name:
        first = sorted(files_by_name.items())[0][1]
        return download_bytes(first["id"]), first["name"]

    raise FileNotFoundError(f"No structure file found in folder {folder_id}")

# ──────────────────────────────────────────────────────────────
# ─── DATA LOADING UTIL ────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
def load_csv_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print(f"[green]Loaded CSV file successfully: {path}[/green]")
        return df
    except Exception as e:
        print(f"[red]Failed to load CSV file: {e}[/red]")
        return pd.DataFrame()

# ──────────────────────────────────────────────────────────────
# ─── PLOTTING UTIL ────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
def plot_defect_levels(defect_data: pd.DataFrame, title: str = "Defect Formation Energy"):
    """Plot defect formation energies (Y = 0–5 eV, large fonts)."""
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.arange(len(defect_data)))

    for i, (_, row) in enumerate(defect_data.iterrows()):
        plt.plot(row["charge_states"], row["energies"], "-o", color=colors[i], label=row["defect"])

    plt.title(title, fontsize=18)
    plt.xlabel("Fermi Level (eV)", fontsize=16)
    plt.ylabel("Formation Energy (eV)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 5)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────
# ─── TEST ─────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.DataFrame({
        "defect": ["V_Cd", "V_Te"],
        "charge_states": [np.linspace(0, 1, 5), np.linspace(0, 1, 5)],
        "energies": [np.random.uniform(0, 4, 5), np.random.uniform(0, 4, 5)]
    })
    plot_defect_levels(df)
