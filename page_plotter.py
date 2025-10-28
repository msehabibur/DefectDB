# page_plotter.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from typing import Optional, List, Dict, Any

# Import necessary functions from the backend
from defect_utils import (
    build_correction_table_for_compound, 
    discover_defects,
    DEFAULT_GAP,
    DEFAULT_VBM
)

# ── Plotting Helpers ─────────────────────────────────────────────────────────
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

def plot_formation_energy(df_to_plot: pd.DataFrame, title: str):
    plt.rcParams["font.size"] = 22
    fig, ax = plt.subplots(figsize=(7,6))
    if df_to_plot.empty:
        ax.text(0.5,0.5,"No data", ha="center", va="center", transform=ax.transAxes)
        st.pyplot(fig, clear_figure=True); return

    gap = _coerce_float(df_to_plot["gap"].iloc[0]); gap = DEFAULT_GAP if (np.isnan(gap) or gap<=0) else gap
    vbm = _coerce_float(df_to_plot["VBM"].iloc[0]); vbm = DEFAULT_VBM if np.isnan(vbm) else vbm

    EF = np.arange(0.0, gap+1e-9, 0.01)
    
    # Track min/max for y-axis
    ymin, ymax = np.inf, -np.inf

    for _, r in df_to_plot.iterrows():
        charges = _available_charges(r)
        if not charges: continue
        
        toten_pure = _coerce_float(r.get("Toten_pure"))
        if np.isnan(toten_pure): continue
        
        mu = _coerce_float(r.get("mu")); mu = 0.0 if np.isnan(mu) else mu
        
        curves = []
        for q, tq, cq in charges:
            Ef_q = (tq - toten_pure + mu + q*(EF + vbm) + cq)
            curves.append(Ef_q)
        
        if not curves: continue
        
        env = np.min(np.vstack(curves), axis=0)
        ymin, ymax = min(ymin, np.min(env)), max(ymax, np.max(env))
        
        label = str(r.get("Label") or r.get("Defect"))
        ax.plot(EF, env, linewidth=3, label=label)

    if np.isinf(ymin): ymin = 0
    if np.isinf(ymax): ymax = 10
    if ymax < 5: ymax = 5
    if ymin > 0: ymin = 0
    
    yrange = ymax - ymin
    
    ax.axvline(0.0, linestyle="dotted", color="black")
    ax.axvline(gap, linestyle="dotted", color="black")
    ax.fill_betweenx([ymin - yrange*0.1, ymax + yrange*0.1], -10, 0, alpha=0.2, color="lightgrey")
    ax.fill_betweenx([ymin - yrange*0.1, ymax + yrange*0.1], gap, gap+10, alpha=0.2, color="lightgrey")
    
    ax.set_xlim(-0.1, gap+0.1)
    ax.set_ylim(ymin - yrange*0.05, ymax + yrange*0.05)
    
    ax.set_xlabel("Fermi Level (eV)")
    ax.set_ylabel("Defect Formation Energy (eV)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    
    st.pyplot(fig, clear_figure=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    st.download_button("Download plot (PNG)", buf.getvalue(), file_name=f"{title}.png")

# ── Tab UI Function ──────────────────────────────────────────────────────────
def render_plotter_tab(
    root_id: str, 
    compounds: Dict[str, str], 
    root_params: Optional[pd.DataFrame]
):
    st.subheader("📦 Select Compound and Conditions")
    
    c1, c2 = st.columns(2)
    with c1:
        comp_sel = st.selectbox(
            "Select a compound", 
            list(compounds.keys()), 
            key="plot_comp_sel"
        )
        comp_id = compounds[comp_sel]
    
    with c2:
        chem_pot_choice = st.selectbox(
            "Chemical potential set", 
            ["Cd-rich", "Te-rich"], 
            key="plot_chem_pot"
        )
    
    defects = discover_defects(comp_id)
    defect_names = sorted(defects.keys())
    st.markdown("### 🧬 Choose defects to analyze")
    chosen_defects = st.multiselect(
        "Defects", 
        defect_names, 
        default=defect_names, 
        key="plot_defect_multi"
    )

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
                    st.dataframe(table, use_container_width=True)
                    st.download_button(
                        "Download table (CSV)",
                        table.to_csv(index=False).encode(),
                        file_name=f"{comp_sel}_corrections.csv"
                    )
                    
                    st.markdown("---")
                    st.subheader("📊 Defect Formation Energy Plot")
                    sub = table[table["Defect"].isin(chosen_defects)].copy()
                    plot_formation_energy(sub, title=f"{comp_sel}_{chem_pot_choice}_formation_plot")
            
            except Exception as e:
                st.error(f"Failed to build/plot: {e}")
                st.exception(e) # Show full traceback
