import io
import os
import ssl
import time
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
MOCK_MODE = True  # ✅ True = run fully offline (no Google Drive API calls)

# ──────────────────────────────────────────────────────────────
# ─── GOOGLE DRIVE (SAFE MOCK INITIALIZATION) ──────────────────
# ──────────────────────────────────────────────────────────────
drive_service = None
if not MOCK_MODE:
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
    except Exception as e:
        print(f"[red]⚠️ Could not initialize Google Drive API: {e}[/red]")
        drive_service = None
else:
    print("[yellow]⚠️ Running in MOCK MODE — no Google Drive access.[/yellow]")

# ──────────────────────────────────────────────────────────────
# ─── RETRY HELPER (unused in mock mode) ───────────────────────
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
# ─── MOCK DOWNLOAD AND FIND STRUCTURE FILE ────────────────────
# ──────────────────────────────────────────────────────────────
def download_bytes(file_id):
    """
    Mock version for local/offline use — skip Google Drive entirely.
    """
    print(f"[yellow]⚠️ Skipping Google Drive download for file_id={file_id} (mock mode).[/yellow]")
    return b"", "mock_structure.cif"

def find_structure_file(folder_id: str):
    """
    Mock function — returns a placeholder structure instead of calling Drive API.
    """
    print(f"[yellow]⚠️ Skipping Google Drive structure lookup for folder_id={folder_id} (mock mode).[/yellow]")
    fake_bytes = b"PLACEHOLDER_CIF_DATA"
    return fake_bytes, "mock_structure.cif"

# ──────────────────────────────────────────────────────────────
# ─── DISCOVERY HELPERS (used by page_structures) ──────────────
# ──────────────────────────────────────────────────────────────
def discover_compounds(root_folder_id: Optional[str] = None) -> List[str]:
    """Return a list of compound names found under the root folder."""
    return ["CdTe", "CdSeTe", "CdZnTe"]

def discover_defects(compound_folder_id: Optional[str] = None) -> List[str]:
    """Return a list of defects available for a given compound."""
    return ["V_Cd", "V_Te", "As_Te", "Cl_Te"]

def discover_charge_states(defect_folder_id: Optional[str] = None) -> List[int]:
    """Return charge states for a given defect."""
    return [-2, -1, 0, +1, +2]

# ──────────────────────────────────────────────────────────────
# ─── DATA LOADING UTIL ────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
def load_csv_data(path: str) -> pd.DataFrame:
    """Safely load CSV data for defect plots or energy tables."""
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
# ─── TEST (SAFE LOCAL RUN) ────────────────────────────────────
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.DataFrame({
        "defect": ["V_Cd", "V_Te"],
        "charge_states": [np.linspace(0, 1, 5), np.linspace(0, 1, 5)],
        "energies": [np.random.uniform(0, 4, 5), np.random.uniform(0, 4, 5)]
    })
    plot_defect_levels(df)
