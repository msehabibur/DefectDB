import io
from typing import Optional, Dict, Any

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

# Use a headless backend for Streamlit environments
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_VBM = 0.0
DEFAULT_GAP = 1.5

# ── Helpers ──────────────────────────────────────────────────────────────────
def _coerce_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = str(x).strip().replace(",", "").replace("−", "-")
        return float(s)
    except Exception:
        return np.nan

def _format_compound_latex(compound_name: str) -> str:
    """Convert compound names to LaTeX format with subscripts."""
    import re

    if not compound_name:
        return compound_name

    try:
        pattern = r'(\d+\.?\d*)'

        def replace_numbers(match):
            num = match.group(1)
            try:
                if '.' in num or (len(num) == 1 and int(num) < 10):
                    return f'$_{{{num}}}$'
            except (ValueError, TypeError):
                pass
            return num

        parts = re.split(r'(?=[A-Z])', compound_name)
        formatted_parts = []

        for part in parts:
            if not part:
                continue
            match = re.match(r'([A-Z][a-z]?)(.*)$', part)
            if match:
                element = match.group(1)
                rest = match.group(2)
                if rest and re.match(r'^\d', rest):
                    rest = re.sub(pattern, replace_numbers, rest)
                formatted_parts.append(element + rest)
            else:
                formatted_parts.append(part)

        return ''.join(formatted_parts)
    except Exception:
        return compound_name

# ── Plotting Function ────────────────────────────────────────────────────────
def plot_formation_energy(df_to_plot: pd.DataFrame, compound_name: str, chem_pot: str, chem_pot_col: str):
    """
    Generates the defect formation energy plot based on the provided DataFrame.
    """
    plt.rc("font", family="sans-serif")
    fig, ax = plt.subplots(figsize=(5, 6))
    plt.subplots_adjust(left=0.14, bottom=0.14, right=0.70, top=0.90)

    if df_to_plot.empty:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center", transform=ax.transAxes)
        st.pyplot(fig, clear_figure=True)
        return

    # Gap
    gap = _coerce_float(df_to_plot["gap"].iloc[0])
    gap = DEFAULT_GAP if (np.isnan(gap) or gap <= 0) else gap

    # Format compound name with LaTeX subscripts
    formatted_name = _format_compound_latex(compound_name)
    ax.set_title(
        rf"{formatted_name} ($\mu$ = {chem_pot})",
        fontsize=20,
        horizontalalignment="center",
        verticalalignment="top",
        y=1.08,
    )

    colors = [
        "red",
        "b",
        "g",
        "c",
        "black",
        "mediumpurple",
        "darkorange",
        "saddlebrown",
        "m",
        "darkkhaki",
        "dodgerblue",
        "grey",
        "salmon",
    ]
    count = 0
    EF = np.arange(-0.5, gap + 0.5, 0.01)
    f = len(EF)
    all_ymin, all_ymax = [], []

    for _, r in df_to_plot.iterrows():
        if r.get("Plot") != "Y":
            continue

        Toten_pure = _coerce_float(r.get("Toten_pure"))
        mu = _coerce_float(r.get(chem_pot_col))
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

        if np.isnan(mu):
            st.info(f"⚠️ Formation energy not available for **{r.get('Defect')}** under this chemical potential condition.")
            continue

        if any(np.isnan(v) for v in [Toten_pure, vbm]):
            continue

        def get_q_energy(toten_q, corr_q, q, ef_val, vbm_val):
            if np.isnan(toten_q) or np.isnan(corr_q):
                return np.inf
            return toten_q - Toten_pure + mu + q * (ef_val + vbm_val) + corr_q

        Form_en = np.array(
            [
                min(
                    get_q_energy(Toten_p2, Corr_p2, 2, EF[j], vbm),
                    get_q_energy(Toten_p1, Corr_p1, 1, EF[j], vbm),
                    get_q_energy(Toten_neut, Corr_neut, 0, EF[j], vbm),
                    get_q_energy(Toten_m1, Corr_m1, -1, EF[j], vbm),
                    get_q_energy(Toten_m2, Corr_m2, -2, EF[j], vbm),
                )
                for j in range(f)
            ]
        )

        valid_indices = np.isfinite(Form_en)
        if not np.any(valid_indices):
            st.warning(f"Skipping defect {r.get('Defect')}: No valid charge states found.")
            continue

        all_ymin.append(np.min(Form_en[valid_indices]))
        all_ymax.append(np.max(Form_en[valid_indices]))

        ax.plot(
            EF[valid_indices],
            Form_en[valid_indices],
            c=colors[count % len(colors)],
            ls="solid",
            lw=4,
            label=r.get("Label"),
        )
        count += 1

    # ── Styling ───────────────────────────────────────────────────────────────
    ax.axvline(x=0, linestyle="dotted", color="black")
    ax.axvline(x=gap, linestyle="dotted", color="black")
    ax.fill_between(EF, -2, 0, color="grey", alpha=1)

    # Regions
    x1 = np.arange(-10, 0.01, 0.01)
    ax.fill_between(x1, -100, 100, facecolor="lightgrey", alpha=0.3)
    x2 = np.arange(gap, 10.0, 0.01)
    ax.fill_between(x2, -100, 100, facecolor="lightgrey", alpha=0.3)
    x3 = np.arange(0.0, gap, 0.01)
    ax.fill_between(x3, -100, 100, facecolor="lightyellow", alpha=0.3)

    ax.set_xlabel("Fermi Level (eV)", size=22, labelpad=8)
    ax.set_ylabel("Defect Formation Energy (eV)", size=22, labelpad=12)
    plt.rc("xtick", labelsize=22)
    plt.rc("ytick", labelsize=22)
    ax.set_xlim([-0.2, gap + 0.2])

    # Auto-scale Y-axis based on selected defects: 0 to (max + 0.5 eV)
    if all_ymax:
        ymax_limit = max(all_ymax) + 0.5
    else:
        ymax_limit = 1.5
    ax.set_ylim([0, ymax_limit])

    ax.set_xticks([0.0, np.round(gap / 2, 2), np.round(gap, 2)])
    # Dynamically set y-ticks based on the range
    if ymax_limit <= 2.0:
        ax.set_yticks(np.arange(0, ymax_limit + 0.5, 0.5))
    else:
        ax.set_yticks(np.arange(0, ymax_limit + 1, 1.0))

    if count > 0:
        ax.legend(
            loc="center left",
            bbox_to_anchor=[1.03, 0.5],
            ncol=1,
            frameon=True,
            prop={"family": "sans-serif", "size": 22},
        )

    st.pyplot(fig, clear_figure=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        "📥 Download plot (PNG)",
        buf.getvalue(),
        file_name=f"{compound_name}_{chem_pot}_plot.png",
    )

    plt.close(fig)

# ── UI Wrapper ───────────────────────────────────────────────────────────────
def render_plotter_page(df: pd.DataFrame):
    st.header("📊 Plotter Controls")

    required_cols = [
        "AB",
        "Defect",
        "Plot",
        "gap",
        "VBM",
        "Toten_pure",
        "mu_Cd_rich",
        "mu_Te_rich",
    ]
    if not all(col in df.columns for col in required_cols):
        st.error(f"File missing required columns: {', '.join(required_cols)}")
        return

    compound_list = df["AB"].unique()
    # Create formatted display names for compounds
    compound_display = {comp: _format_compound_latex(comp) for comp in compound_list}
    formatted_compounds = [compound_display[comp] for comp in compound_list]

    comp_sel_formatted = st.selectbox("1️⃣ Select a Compound", formatted_compounds)
    # Get the original compound name
    comp_sel = [k for k, v in compound_display.items() if v == comp_sel_formatted][0]

    df_comp = df[df["AB"] == comp_sel].copy()

    mu_cols = {"Cd-rich": "mu_Cd_rich", "Te-rich": "mu_Te_rich"}
    chem_pot_choice = st.selectbox("2️⃣ Select Chemical Potential", mu_cols.keys())
    chem_pot_col_name = mu_cols[chem_pot_choice]

    defect_list = sorted(df_comp["Defect"].unique())
    chosen_defects = st.multiselect("3️⃣ Select Defects to Plot", defect_list, default=[])

    if st.button("🚀 Generate Plot"):
        if not chosen_defects:
            st.warning("Please select at least one defect.")
            return

        df_plot = df_comp[df_comp["Defect"].isin(chosen_defects)].copy()

        if df_plot.empty:
            st.warning("No data found for the selected defects.")
        else:
            st.subheader(f"Formation Energy Plot: {comp_sel_formatted}")
            plot_formation_energy(
                df_plot,
                compound_name=comp_sel,
                chem_pot=chem_pot_choice,
                chem_pot_col=chem_pot_col_name,
            )

            with st.expander("Show Filtered Data Used for Plot"):
                st.dataframe(df_plot, width="stretch")
