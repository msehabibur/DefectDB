# page_plotter.py
import io
from typing import Optional, Dict, Any

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure Matplotlib uses a non-interactive backend that works in headless
# environments (such as the execution sandbox for these exercises).  Using a
# GUI backend can lead to segmentation faults when Streamlit launches.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_VBM = 0.0
DEFAULT_GAP = 1.5

# ── Plotting Function (from your script) ─────────────────────────────────────
def _coerce_float(x):
    try:
        if x is None: return np.nan
        if isinstance(x, (int, float, np.floating)): return float(x)
        s = str(x).strip().replace(",", "").replace("−", "-")
        return float(s)
    except Exception:
        return np.nan

def plot_formation_energy(df_to_plot: pd.DataFrame, compound_name: str, chem_pot: str, chem_pot_col: str):
    """
    Generates the defect formation energy plot based on the provided DataFrame
    and plotting script logic.
    """
    plt.rc('font', family='sans-serif')
    fig, ax = plt.subplots(figsize=(5, 6))
    plt.subplots_adjust(left=0.14, bottom=0.14, right=0.70, top=0.90)

    if df_to_plot.empty:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center", transform=ax.transAxes)
        st.pyplot(fig, clear_figure=True); return

    # Get gap from the data (use default if missing)
    gap = _coerce_float(df_to_plot["gap"].iloc[0]); gap = DEFAULT_GAP if (np.isnan(gap) or gap<=0) else gap
    
    # Set title based on your format
    ax.set_title(f'{compound_name} ($\mu$ = {chem_pot})', fontsize=20, horizontalalignment='center', verticalalignment='top', y=1.08)

    # Use your color list
    colors = ['red', 'b', 'g', 'c', 'black', 'mediumpurple', 'darkorange', 'saddlebrown', 'm', 'darkkhaki', 'dodgerblue', 'grey', 'salmon']
    count = 0

    # Use your Fermi level range
    EF = np.arange(-0.5, gap + 0.5, 0.01)
    f = len(EF)

    # Store min/max for y-axis
    all_ymin, all_ymax = [], []

    for _, r in df_to_plot.iterrows():
        # Filter rows where Plot == 'Y'
        if r.get("Plot") != 'Y':
            continue

        # Get all values, defaulting to nan
        Toten_pure = _coerce_float(r.get("Toten_pure"))
        mu = _coerce_float(r.get(chem_pot_col)) # Use the selected mu column
        vbm = _coerce_float(r.get("VBM"))
        
        Toten_p2 = _coerce_float(r.get("Toten_p2"))
        Corr_p2 = _coerce_float(r.get("Corr_p2"))
        Toten_p1 = _coerce_float(r.get("Toten_p1"))
        Corr_p1 = _coerce_float(r.get("Corr_p1"))
        Toten_neut = _coerce_float(r.get("Toten_neut"))
        Corr_neut = _coerce_float(r.get("Corr_neut"))
        Toten_m1 = _coerce_float(r.get("Toten_m1"))
        Corr_m1 = _coerce_float(r.get("Corr_m1"))
        Toten_m2 = _coerce_float(r.get("Toten_m2"))
        Corr_m2 = _coerce_float(r.get("Corr_m2"))

        # Skip if pure, mu, or vbm is missing
        if any(np.isnan(v) for v in [Toten_pure, mu, vbm]):
            st.warning(f"Skipping defect {r.get('Defect')}: missing Toten_pure, mu, or VBM.")
            continue

        # Helper to calculate energy for a charge state, returning np.inf if data is missing
        def get_q_energy(toten_q, corr_q, q, ef_val, vbm_val):
            if np.isnan(toten_q) or np.isnan(corr_q):
                return np.inf
            return toten_q - Toten_pure + mu + q * (ef_val + vbm_val) + corr_q

        # Calculate the lower envelope (formation energy)
        Form_en = np.array([min(
            get_q_energy(Toten_p2, Corr_p2, 2, EF[j], vbm),
            get_q_energy(Toten_p1, Corr_p1, 1, EF[j], vbm),
            get_q_energy(Toten_neut, Corr_neut, 0, EF[j], vbm),
            get_q_energy(Toten_m1, Corr_m1, -1, EF[j], vbm),
            get_q_energy(Toten_m2, Corr_m2, -2, EF[j], vbm)
        ) for j in range(f)])
        
        # Filter out inf values for plotting and range calculation
        valid_indices = np.isfinite(Form_en)
        if not np.any(valid_indices):
            st.warning(f"Skipping defect {r.get('Defect')}: No valid charge states found.")
            continue
            
        all_ymin.append(np.min(Form_en[valid_indices]))
        all_ymax.append(np.max(Form_en[valid_indices]))

        ax.plot(EF[valid_indices], Form_en[valid_indices], c=colors[count % len(colors)], ls='solid', lw=4, label=r.get("Label"))
        count += 1

    # --- Apply your exact styling ---
    
    # Add vertical dotted lines at x=0 and x=gap
    ax.axvline(x=0, linestyle='dotted', color='black')
    ax.axvline(x=gap, linestyle='dotted', color='black')

    # Shaded area from Y = -2 to Y = 0
    ax.fill_between(EF, -2, 0, color='grey', alpha=1)

    # Shaded VB region
    x1 = np.arange(-10, 0.01,  0.01)
    ax.fill_between(x1, -100, 100, facecolor='lightgrey', alpha=0.3)
    # Shaded CB region
    x2 = np.arange(gap, 10.0,  0.01)
    ax.fill_between(x2, -100, 100, facecolor='lightgrey', alpha=0.3)
    # Shaded Gap region
    x3 = np.arange(0.0, gap,  0.01)
    ax.fill_between(x3, -100, 100, facecolor='lightyellow', alpha=0.3)

    ax.set_xlabel('Fermi Level (eV)', fontname='sans-serif', size=22, labelpad=8)
    ax.set_ylabel('Defect Formation Energy (eV)', fontname='sans-serif', size=22, labelpad=12)
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    
    # Set limits and ticks based on your script
    ax.set_xlim([-0.2, gap + 0.2])
    
    # Use calculated y-min/max if available, else default
    ymin = min(all_ymin) if all_ymin else -0.2
    ymax = max(all_ymax) if all_ymax else 1.5
    # Use your script's y-range logic
    ax.set_ylim([max(ymin, -0.2), min(ymax, 1.5)]) 
    
    ax.set_xticks([0.0, np.round(gap / 2, 2), np.round(gap, 2)])
    ax.set_yticks([0, 0.5, 1, 1.5])
    
    if count > 0:
        ax.legend(loc='center left', bbox_to_anchor=[1.03, 0.5], ncol=1, frameon=True, prop={'family': 'sans-serif', 'size': 22})
    
    st.pyplot(fig, clear_figure=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    
    # Update download button filename
    st.download_button(
        "Download plot (PNG)", 
        buf.getvalue(),
        file_name=f"{compound_name}_{chem_pot}_plot.png"
    )

    plt.close(fig)

# ── Main Page UI Function ──────────────────────────────────────────────────
def render_plotter_page(df: pd.DataFrame):
    st.header("📊 Plotter Controls")
    
    # Check for required columns based on the image
    required_cols = ['AB', 'Defect', 'Plot', 'gap', 'VBM', 'Toten_pure', 'mu_Cd_rich', 'mu_Te_rich']
    if not all(col in df.columns for col in required_cols):
        st.error(f"File is missing required columns. Make sure it contains: {', '.join(required_cols)}")
        return

    # 1. Ask user which compound (from 'AB' column)
    compound_list = df['AB'].unique()
    comp_sel = st.selectbox("1. Select a Compound", compound_list)
    
    # Filter dataframe by selected compound
    df_comp = df[df['AB'] == comp_sel].copy()

    # 2. Ask which chemical potential (from mu_ columns)
    mu_cols = {
        "Cd-rich": "mu_Cd_rich", 
        "Te-rich": "mu_Te_rich"
    }
    chem_pot_choice = st.selectbox("2. Select Chemical Potential", mu_cols.keys())
    chem_pot_col_name = mu_cols[chem_pot_choice]

    # 3. Ask which defects (from 'Defect' column)
    defect_list = sorted(df_comp['Defect'].unique())
    chosen_defects = st.multiselect("3. Select Defects to Plot", defect_list, default=[])

    if st.button("Generate Plot"):
        if not chosen_defects:
            st.warning("Please select at least one defect.")
            return

        # Filter the dataframe for the final plot
        df_plot = df_comp[df_comp['Defect'].isin(chosen_defects)].copy()
        
        if df_plot.empty:
            st.warning("No data found for the selected defects.")
        else:
            st.subheader(f"Formation Energy Plot: {comp_sel}")
            plot_formation_energy(
                df_plot, 
                compound_name=comp_sel, 
                chem_pot=chem_pot_choice,
                chem_pot_col=chem_pot_col_name
            )
            
            with st.expander("Show Filtered Data Used for Plot"):
                st.dataframe(df_plot)
