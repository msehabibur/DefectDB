"""Utility helpers for the DefectDB Streamlit application."""

import importlib.util
import io
import os
import re
import ssl
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def _resolve_rich_print():
    """Return ``rich.print`` when available, otherwise fall back to ``print``."""

    if importlib.util.find_spec("rich") is not None:  # pragma: no branch - deterministic check
        from rich import print as rich_print  # type: ignore

        return rich_print
    return print


console_print = _resolve_rich_print()

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
def download_bytes(file_id: str) -> Tuple[bytes, str]:
    """
    Mock version for local/offline use — skip Google Drive entirely.
    """
    console_print(
        f"[yellow]⚠️ Skipping Google Drive download for file_id={file_id} (mock mode).[/yellow]"
    )
    return b"", "mock_structure.cif"

def find_structure_file(folder_id: str) -> Tuple[bytes, str]:
    """
    Mock function — returns a placeholder structure instead of calling Drive API.
    """
    console_print(
        "[yellow]⚠️ Skipping Google Drive structure lookup for "
        f"folder_id={folder_id} (mock mode).[/yellow]"
    )
    fake_bytes = b"PLACEHOLDER_CIF_DATA"
    return fake_bytes, "mock_structure.cif"

# ──────────────────────────────────────────────────────────────
# ─── DISCOVERY HELPERS (used by page_structures) ──────────────
# ──────────────────────────────────────────────────────────────
MockFolderId = str


def discover_compounds(
    root_folder_id: Optional[str] = None,
) -> Dict[str, MockFolderId]:
    """Return a deterministic mapping of compound labels to mock folder IDs."""

    del root_folder_id  # unused in mock mode
    return {
        "CdTe": "compound_cdte",
        "CdSeTe": "compound_cdsxte",
        "CdZnTe": "compound_cdznte",
    }


def discover_defects(
    compound_folder_id: Optional[str] = None,
) -> Dict[str, MockFolderId]:
    """Return a deterministic mapping of defect labels to mock folder IDs."""

    del compound_folder_id  # unused in mock mode
    return {
        "V_Cd": "defect_v_cd",
        "V_Te": "defect_v_te",
        "As_Te": "defect_as_te",
        "Cl_Te": "defect_cl_te",
    }


def discover_charge_states(
    defect_folder_id: Optional[str] = None,
) -> Dict[str, MockFolderId]:
    """Return a deterministic mapping of charge labels to mock folder IDs."""

    del defect_folder_id  # unused in mock mode
    return {
        "q+2": "charge_plus_2",
        "q+1": "charge_plus_1",
        "q0": "charge_0",
        "q-1": "charge_minus_1",
        "q-2": "charge_minus_2",
    }


def parse_charge_label_to_q(label: str) -> Optional[int]:
    """Parse a charge label such as ``q+1`` or ``neutral`` into an integer value."""

    if not isinstance(label, str):
        return None

    stripped = label.strip()
    if not stripped:
        return None

    lowered = stripped.lower()
    if lowered in {"neutral", "neut", "qneutral"}:
        return 0

    if lowered in {"q0", "0", "+0", "-0"}:
        return 0

    match = re.search(r"([+-]?\d+)", stripped)
    if match:
        try:
            return int(match.group(1))
        except ValueError:  # pragma: no cover - defensive guard
            return None

    return None

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
def _iter_rows(defect_data: pd.DataFrame) -> Iterable[Tuple[Any, Any, Any]]:
    """Yield defect plotting tuples without forcing Matplotlib imports globally."""

    for _, row in defect_data.iterrows():
        yield row["defect"], row["charge_states"], row["energies"]


def plot_defect_levels(
    defect_data: pd.DataFrame, title: str = "Defect Formation Energy"
) -> None:
    """
    Plot defect formation energies with larger font sizes and Y-limits (0–5 eV).
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.arange(len(defect_data)))

    for i, (defect_name, charges, energies) in enumerate(_iter_rows(defect_data)):
        plt.plot(charges, energies, "-o", color=colors[i], label=defect_name)

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
