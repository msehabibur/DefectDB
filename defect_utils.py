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
ROOT_FOLDER_ID_DEFAULT = None  # placeholder so app.py import succeeds

# ──────────────────────────────────────────────────────────────
# ─── GOOGLE DRIVE SETUP ───────────────────────────────────────
# ──────────────────────────────────────────────────────────────

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2 import service_account

    SERVICE_ACCOUNT_FILE = "service_account.json"
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

    if os.path.exists(SERVICE_ACCOUNT_FILE):
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        drive_service = build("drive", "v3", credentials=credentials)
        print("[green]✅ Google Drive service initialized successfully.[/green]")
    else:
        print("[yellow]⚠️ service_account.json not found; continuing without Drive access.[/yellow]")
        drive_service = None

except Exception as e:
    print(f"[red]⚠️ Could not initialize Google Drive API: {e}[/red]")
    drive_service = None

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
# ─── FIXED DOWNLOAD FUNCTION ──────────────────────────────────
# ──────────────────────────────────────────────────────────────

def download_bytes(file_id):
    """
    Download a file from Google Drive using a clean HTTPS client with proper SSL.
    """
    if drive_service is None:
        raise RuntimeError("Google Drive API not initialized (missing credentials).")

    # Remove proxy interference
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("HTTP_PROXY", None)

    # Force secure TLS ≥ 1.2
    ssl_context = ssl.create_default_context()
    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2

    # Rebuild HTTP client with secure context
    http = httplib2.Http(timeout=60, ca_certs=ssl.get_default_verify_paths().cafile)
    http.disable_ssl_certificate_validation = False

    # Prepare download request
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
# ─── DISCOVERY HELPERS (used by page_structures) ──────────────
# ──────────────────────────────────────────────────────────────

def discover_compounds(root_folder_id: Optional[str] = None) -> List[str]:
    """Return a list of compound names found under the root folder."""
    # Placeholder version — replace with real Drive traversal later
    return ["CdTe", "CdSeTe", "CdZnTe"]

def discover_defects(compound_folder_id: Optional[str] = None) -> List[str]:
    """Return a list of defects available for a given compound."""
    return ["V_Cd", "V_Te", "As_Te", "Cl_Te"]

def discover_charge_states(defect_folder_id: Optional[str] = None) -> List[int]:
    """Return charge states for a given defect."""
    return [-2, -1, 0, +1, +2]

# ──────────────────────────────────────────────────────────────
# ─── FILE SEARCHING UTILS ─────────────────────────────────────
# ──────────────────────────────────────────────────────────────

STRUCTURE_FILE_PRIORITY = [
    "CONTCAR",
    "POSCAR",
    "Relaxed.cif",
    "Final.cif",
    "structure.cif",
    "optimized.cif",
]

def find_structure_file(folder_id: str):
    """
    Find the preferred structure file (CONTCAR, POSCAR, etc.) in a given Drive folder.
    """
    if drive_service is None:
        raise RuntimeError("Google Drive API not initialized (missing credentials).")

    results = (
        drive_service.files()
        .list(q=f"'{folder_id}' in parents and trashed=false", fields="files(id, name)")
        .execute()
    )
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
    """
    Plot defect formation energies with larger font sizes and Y-limits (0–5 eV).
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.arange(len(defect_data)))

    for i, (_, row) in enumerate(defect_data.iterrows()):
        plt.plot(row["charge_states"], row["energies"], "-o", color=colors[i], label=row["defect"])

    plt.title(title, fontsize=18)
    plt.xlabel("Fermi Level (eV)", fontsize=16)
    plt.ylabel("Formation Energy (eV)", fontsize=16)

    # Larger tick fonts
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Limit Y-axis
    plt.ylim(0, 5)

    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────
# ─── EXAMPLE TEST ─────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.DataFrame({
        "defect": ["V_Cd", "V_Te"],
        "charge_states": [np.linspace(0, 1, 5), np.linspace(0, 1, 5)],
        "energies": [np.random.uniform(0, 4, 5), np.random.uniform(0, 4, 5)]
    })
    plot_defect_levels(df)
