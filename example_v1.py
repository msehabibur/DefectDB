#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 Universal Defect Simulator
========================================================================
M3GNet Universal Potential for Defect Analysis
"""

import warnings, os, io, base64, tempfile, json, requests, zipfile, shutil, re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from itertools import product

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

from pymatgen.core import Structure, Composition, Element, Lattice
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.vasp import Vasprun
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Try to import new MP API
try:
    from mp_api.client import MPRester as NewMPRester
    HAS_NEW_MP_API = True
except ImportError:
    HAS_NEW_MP_API = False

# M3GNet
from matgl import load_model
from matgl.ext.ase import M3GNetCalculator, Relaxer

# ASE
from ase.optimize import BFGS
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize import view
from ase.io import write

# Doped for chemical potentials
try:
    from doped.chemical_potentials import CompetingPhasesAnalyzer
    HAS_DOPED = True
except ImportError:
    HAS_DOPED = False

warnings.filterwarnings("ignore")

# =====================================================================================
# CONFIGURATION
# =====================================================================================
UDS_UPLOAD_DIR = Path.home() / "uds_uploads"
UDS_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

UDS_MODELS_DIR = Path.home() / "uds_models"
UDS_MODELS_DIR.mkdir(parents=True, exist_ok=True)

UDS_CHEMPOT_CACHE = Path.home() / "uds_chempot_cache"
UDS_CHEMPOT_CACHE.mkdir(parents=True, exist_ok=True)

UDS_ANIONS = {"O", "S", "Se", "Te"}

# Light Color Palette (softer colors)
UDS_COLORS = {
    'header_bg': '#6b7280',
    'header_text': '#ffffff',
    'panel_bg': '#f9fafb',
    'panel_border': '#e5e7eb',
    'text_primary': '#374151',
    'text_secondary': '#6b7280',
    'success_bg': '#f0f4f8',
    'success_border': '#cbd5e0',
    'error_bg': '#fef2f2',
    'error_border': '#fecaca',
    'warning_bg': '#fffbeb',
    'warning_border': '#fde68a',
    'bar_fill': '#a8b8d8',      # Soft blue
    'bar_edge': '#7a8fb5',       # Darker blue
    'grid_line': '#e5e7eb',
    'scatter_1': '#b8cfe5',      # Light blue
    'scatter_2': '#c4d6b0',      # Light green  
    'scatter_3': '#f4c7ab',      # Light coral
}

# Periodic Table
PERIODIC_TABLE = [
    ['H', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'He'],
    ['Li', 'Be', '', '', '', '', '', '', '', '', '', '', 'B', 'C', 'N', 'O', 'F', 'Ne'],
    ['Na', 'Mg', '', '', '', '', '', '', '', '', '', '', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
    ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
    ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'],
    ['Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']
]

# =====================================================================================
# SIMPLE LOGGING
# =====================================================================================

def log_info(text, output_widget=None):
    """Log info message"""
    msg = f"<span style='color: #4b5563;'>{text}</span>"
    if output_widget:
        with output_widget:
            display(HTML(msg))
    else:
        display(HTML(msg))

def log_success(text, output_widget=None):
    """Log success message"""
    msg = f"<span style='color: #065f46; font-weight: bold;'>{text}</span>"
    if output_widget:
        with output_widget:
            display(HTML(msg))
    else:
        display(HTML(msg))

def log_warning(text, output_widget=None):
    """Log warning message"""
    msg = f"<span style='color: #92400e;'>{text}</span>"
    if output_widget:
        with output_widget:
            display(HTML(msg))
    else:
        display(HTML(msg))

def log_error(text, output_widget=None):
    """Log error message"""
    msg = f"<span style='color: #991b1b; font-weight: bold;'>{text}</span>"
    if output_widget:
        with output_widget:
            display(HTML(msg))
    else:
        display(HTML(msg))

def log_section(text, output_widget=None):
    """Log section header with box-drawing characters"""
    # Calculate padding for centering
    box_width = 50
    text_len = len(text)
    padding = (box_width - text_len - 2) // 2
    
    top = "╔" + "═" * (box_width - 2) + "╗"
    middle = "║" + " " * padding + text + " " * (box_width - text_len - padding - 2) + "║"
    bottom = "╚" + "═" * (box_width - 2) + "╝"
    
    msg = f"<div style='color: #1e40af; font-family: monospace; font-weight: bold; margin: 10px 0;'>" \
          f"{top}<br>{middle}<br>{bottom}</div>"
    
    if output_widget:
        with output_widget:
            display(HTML(msg))
    else:
        display(HTML(msg))

def log_plain(text, output_widget=None):
    """Log plain text"""
    msg = f"<span style='color: #6b7280;'>{text}</span>"
    if output_widget:
        with output_widget:
            display(HTML(msg))
    else:
        display(HTML(msg))

# =====================================================================================
# STRUCTURE UTILITIES
# =====================================================================================

def get_conventional_structure(structure):
    """Convert primitive to conventional cell if needed"""
    try:
        analyzer = SpacegroupAnalyzer(structure)
        conv_struct = analyzer.get_conventional_standard_structure()
        return conv_struct
    except:
        return structure

def get_structure_info(structure):
    """Get detailed structure information"""
    try:
        analyzer = SpacegroupAnalyzer(structure)
        primitive = analyzer.get_primitive_standard_structure()
        conventional = analyzer.get_conventional_standard_structure()
        
        is_primitive = len(structure) == len(primitive)
        
        return {
            'is_primitive': is_primitive,
            'n_atoms': len(structure),
            'n_atoms_primitive': len(primitive),
            'n_atoms_conventional': len(conventional),
            'space_group': analyzer.get_space_group_symbol(),
            'lattice_type': analyzer.get_lattice_type()
        }
    except:
        return {
            'is_primitive': False,
            'n_atoms': len(structure),
            'space_group': 'Unknown',
            'lattice_type': 'Unknown'
        }

# =====================================================================================
# DEFECT UTILITIES
# =====================================================================================

def get_central_atom_idx(struct, symbol, excluded_indices=None):
    """Find the atom nearest to cell center"""
    if excluded_indices is None:
        excluded_indices = []
    center = struct.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
    min_dist, best_idx = float("inf"), -1
    for i, site in enumerate(struct):
        if i in excluded_indices:
            continue
        if site.specie.symbol != symbol:
            continue
        d = np.linalg.norm(site.coords - center)
        if d < min_dist:
            min_dist, best_idx = d, i
    if best_idx < 0:
        raise ValueError(f"No available atom '{symbol}' found near cell center.")
    return best_idx

def get_min_pbc_distance(struct, cart_coords):
    """Get minimum distance to any atom considering PBC"""
    frac = struct.lattice.get_fractional_coords(cart_coords)
    mind = float("inf")
    for site in struct:
        df = frac - site.frac_coords
        df -= np.round(df)
        d = np.linalg.norm(struct.lattice.get_cartesian_coords(df))
        if d < mind:
            mind = d
    return float(mind)

def add_interstitial(struct, symbol):
    """Add interstitial at position with maximum distance from other atoms"""
    best_pos, best_d = None, -1.0
    for f in product(np.linspace(0.3, 0.7, 7), repeat=3):
        cart = struct.lattice.get_cartesian_coords(f)
        d = get_min_pbc_distance(struct, cart)
        if d > best_d:
            best_d, best_pos = d, cart
    if best_pos is None:
        raise ValueError(f"Could not place interstitial {symbol}.")
    struct.append(symbol, best_pos, coords_are_cartesian=True)

def create_defect_structure(bulk, defect_str):
    """Create defect structure from notation (V_Cd, Cu_i, As_Te, Cl_Te+V_Cd)"""
    s = bulk.copy()
    used_indices = []
    species_in_struct = sorted({site.specie.symbol for site in s})

    for raw in defect_str.split("+"):
        part = raw.strip()
        if not part:
            continue

        # Vacancy: V_X
        if part.startswith("V_"):
            host = part[2:]
            if host not in species_in_struct:
                raise ValueError(f"Vacancy target '{host}' not in structure. Available: {species_in_struct}")
            idx = get_central_atom_idx(s, host, excluded_indices=used_indices)
            s.remove_sites([idx])
            used_indices.append(idx)
            continue

        # Interstitial: X_i
        if part.endswith("_i"):
            elem = part[:-2]
            if not elem:
                raise ValueError(f"Invalid interstitial: '{part}'")
            add_interstitial(s, elem)
            continue

        # Substitution: A_B
        if "_" in part:
            A, B = part.split("_", 1)
            if B not in species_in_struct:
                raise ValueError(f"Substitution site '{B}' not in structure. Available: {species_in_struct}")
            idx = get_central_atom_idx(s, B, excluded_indices=used_indices)
            s.replace(idx, A)
            used_indices.append(idx)
            continue

        raise ValueError(f"Unknown defect notation: '{part}'")

    return s

def validate_supercell_size(structure, min_size=15.0):
    """Validate that supercell is at least min_size Angstroms in all directions"""
    a, b, c = structure.lattice.abc
    
    issues = []
    multipliers = [1, 1, 1]
    
    if a < min_size:
        mult = int(np.ceil(min_size / a))
        multipliers[0] = mult
        issues.append(f"a={a:.2f}Å (needs {mult}x)")
    
    if b < min_size:
        mult = int(np.ceil(min_size / b))
        multipliers[1] = mult
        issues.append(f"b={b:.2f}Å (needs {mult}x)")
    
    if c < min_size:
        mult = int(np.ceil(min_size / c))
        multipliers[2] = mult
        issues.append(f"c={c:.2f}Å (needs {mult}x)")
    
    if issues:
        msg = f"Cell too small: {', '.join(issues)}. Minimum: {min_size}Å in all directions."
        return False, msg, multipliers
    
    msg = f"Cell size OK: a={a:.2f}Å, b={b:.2f}Å, c={c:.2f}Å (all ≥ {min_size}Å)"
    return True, msg, multipliers

def extract_impurity_elements_from_defects(defect_list, bulk_elements):
    """Extract impurity elements from defect notation"""
    impurity_elements = []
    
    for defect_str in defect_list:
        # Parse each component of defect (handle complexes like As_Te+Cl_i)
        for part in defect_str.replace('+', ' ').split():
            part = part.strip()
            
            # Interstitial: X_i
            if part.endswith('_i'):
                elem = part[:-2]
                if elem not in bulk_elements and elem not in impurity_elements:
                    impurity_elements.append(elem)
            
            # Substitution: A_B (dopant A is impurity if not in bulk)
            elif '_' in part and not part.startswith('V_'):
                dopant = part.split('_')[0]
                if dopant not in bulk_elements and dopant not in impurity_elements:
                    impurity_elements.append(dopant)
    
    return impurity_elements

# =====================================================================================
# 3D STRUCTURE VIEWER
# =====================================================================================

def visualize_structure_3d(structure, title="Structure", output_widget=None):
    """Visualize structure in 3D with color-coded atoms"""
    try:
        from ase.io import write
        import plotly.graph_objects as go
        
        # Get atom colors (Jmol colors)
        element_colors = {
            'H': '#FFFFFF', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
            'F': '#90E050', 'Na': '#AB5CF2', 'Mg': '#8AFF00', 'Al': '#BFA6A6',
            'Si': '#F0C8A0', 'P': '#FF8000', 'S': '#FFFF30', 'Cl': '#1FF01F',
            'K': '#8F40D4', 'Ca': '#3DFF00', 'Ti': '#BFC2C7', 'V': '#A6A6AB',
            'Cr': '#8A99C7', 'Mn': '#9C7AC7', 'Fe': '#E06633', 'Co': '#F090A0',
            'Ni': '#50D050', 'Cu': '#C88033', 'Zn': '#7D80B0', 'Ga': '#C28F8F',
            'Ge': '#668F8F', 'As': '#BD80E3', 'Se': '#FFA100', 'Br': '#A62929',
            'Cd': '#FFD98F', 'In': '#A67573', 'Sn': '#668080', 'Sb': '#9E63B5',
            'Te': '#D47A00', 'I': '#940094', 'Ba': '#00C900', 'Pb': '#575961',
            'Bi': '#9E4FB5'
        }
        
        # Default color for unknown elements
        default_color = '#FF1493'
        
        # Prepare data for plotting
        positions = np.array([site.coords for site in structure])
        symbols = [site.specie.symbol for site in structure]
        colors = [element_colors.get(sym, default_color) for sym in symbols]
        
        # Get unique elements for legend
        unique_elements = sorted(set(symbols))
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add atoms by element type for legend
        for elem in unique_elements:
            mask = np.array([s == elem for s in symbols])
            pos_elem = positions[mask]
            
            fig.add_trace(go.Scatter3d(
                x=pos_elem[:, 0],
                y=pos_elem[:, 1],
                z=pos_elem[:, 2],
                mode='markers',
                name=elem,
                marker=dict(
                    size=8,
                    color=element_colors.get(elem, default_color),
                    line=dict(color='#000000', width=1)
                ),
                hovertemplate=f'<b>{elem}</b><br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>z: %{{z:.2f}}<extra></extra>'
            ))
        
        # Add unit cell edges
        a, b, c = structure.lattice.matrix
        origin = np.array([0, 0, 0])
        
        # Define unit cell edges
        edges = [
            [origin, a], [origin, b], [origin, c],
            [a, a+b], [a, a+c], [b, b+a], [b, b+c],
            [c, c+a], [c, c+b], [a+b, a+b+c],
            [a+c, a+b+c], [b+c, a+b+c]
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[edge[0][0], edge[1][0]],
                y=[edge[0][1], edge[1][1]],
                z=[edge[0][2], edge[1][2]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=22, color='#374151', family='Arial, sans-serif')
            ),
            scene=dict(
                xaxis=dict(title='X (Å)', backgroundcolor='#f9fafb', gridcolor='#e5e7eb'),
                yaxis=dict(title='Y (Å)', backgroundcolor='#f9fafb', gridcolor='#e5e7eb'),
                zaxis=dict(title='Z (Å)', backgroundcolor='#f9fafb', gridcolor='#e5e7eb'),
                aspectmode='data'
            ),
            legend=dict(
                title=dict(text='Elements', font=dict(size=18)),
                font=dict(size=16),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#e5e7eb',
                borderwidth=2
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor='#fafafa'
        )
        
        if output_widget:
            with output_widget:
                display(fig)
        else:
            display(fig)
            
    except Exception as e:
        log_error(f"3D visualization failed: {e}", output_widget)

# =====================================================================================
# MATERIALS PROJECT UTILITIES
# =====================================================================================

def search_mp_structures(query, api_key=None, log_func=print):
    """Search Materials Project and return ALL matching structures"""
    if not HAS_NEW_MP_API:
        raise ImportError("The new mp-api package is required.\nInstall with: pip install mp-api")
    
    if api_key is None:
        api_key = os.environ.get('MP_API_KEY', '')
    
    if not api_key:
        raise ValueError("MP API key required!")
    
    try:
        with NewMPRester(api_key) as mpr:
            # Direct ID lookup
            if query.startswith('mp-') or query.startswith('mvc-'):
                log_func(f"Direct lookup: {query}")
                docs = mpr.summary.search(
                    material_ids=[query],
                    fields=["material_id", "formula_pretty", "energy_per_atom", "structure"]
                )
                
                if not docs:
                    raise ValueError(f"Material {query} not found")
                
                doc = docs[0]
                return [(doc.material_id, doc.formula_pretty, doc.energy_per_atom, doc.structure)]
            
            # Formula search
            log_func(f"Searching Materials Project for: {query}")
            
            docs = mpr.summary.search(
                formula=query,
                fields=["material_id", "formula_pretty", "energy_per_atom", "structure"]
            )
            
            if not docs:
                raise ValueError(f"No materials found for '{query}'")
            
            log_func(f"Found {len(docs)} structures")
            
            structures_list = []
            for doc in docs[:50]:  # Increased from 20
                try:
                    structures_list.append((
                        doc.material_id,
                        doc.formula_pretty,
                        doc.energy_per_atom,
                        doc.structure
                    ))
                except:
                    pass
            
            structures_list.sort(key=lambda x: x[2])
            
            return structures_list
            
    except Exception as e:
        raise Exception(f"Materials Project search failed: {e}")

def search_compounds_by_elements(elements, api_key=None, log_func=print):
    """Search for all compounds containing the specified elements"""
    if not HAS_NEW_MP_API:
        raise ImportError("The new mp-api package is required.\nInstall with: pip install mp-api")
    
    if api_key is None:
        api_key = os.environ.get('MP_API_KEY', '')
    
    if not api_key:
        raise ValueError("MP API key required!")
    
    try:
        with NewMPRester(api_key) as mpr:
            element_str = '-'.join(sorted(elements))
            log_func(f"Searching for compounds with elements: {element_str}")
            
            chemsys = '-'.join(sorted(elements))
            docs = mpr.summary.search(
                chemsys=chemsys,
                fields=["material_id", "formula_pretty", "energy_per_atom", "structure"]
            )
            
            if not docs:
                log_func(f"No compounds found with elements: {elements}")
                return []
            
            log_func(f"Found {len(docs)} compounds")
            
            structures_list = []
            for doc in docs[:50]:  # Increased from 30
                try:
                    struct_elements = set(str(el) for el in doc.structure.composition.elements)
                    if struct_elements.issubset(set(elements)):
                        structures_list.append((
                            doc.material_id,
                            doc.formula_pretty,
                            doc.energy_per_atom,
                            doc.structure
                        ))
                except:
                    pass
            
            structures_list.sort(key=lambda x: (len(x[3].composition.elements), x[2]))
            
            return structures_list
            
    except Exception as e:
        log_func(f"Error searching compounds: {e}")
        return []

# =====================================================================================
# CHEMICAL POTENTIAL CALCULATION
# =====================================================================================

def canonical_formula(composition):
    """Convert to canonical formula (anions last, hide 1s)"""
    try:
        comp = Composition(composition)
        integer_form, _ = comp.get_integer_formula_and_factor()
        
        if isinstance(integer_form, str):
            comp_int = Composition(integer_form)
        else:
            comp_int = comp
        
        elements = sorted(comp_int.elements, 
                         key=lambda el: (1 if str(el) in UDS_ANIONS else 0, el.Z))
        
        parts = []
        for el in elements:
            amt = int(round(comp_int[el]))
            parts.append(f"{el}{'' if amt == 1 else amt}")
        
        return "".join(parts)
    except Exception:
        return composition

def calculate_chemical_potentials_simple(
    bulk_structure,
    bulk_energy,
    impurity_elements=None,
    api_key=None,
    log_widget=None
):
    """Simplified chemical potential calculation using pymatgen PhaseDiagram"""
    log_section("CHEMICAL POTENTIAL CALCULATION", log_widget)
    log_plain("", log_widget)
    
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
    
    try:
        comp = bulk_structure.composition
        bulk_elements = sorted([str(el) for el in comp.elements])
        
        # Add impurity elements
        all_elements = bulk_elements.copy()
        if impurity_elements:
            all_elements = sorted(list(set(all_elements + impurity_elements)))
            log_info(f"Bulk elements: {bulk_elements}", log_widget)
            log_info(f"Impurity elements: {impurity_elements}", log_widget)
            log_info(f"Total chemical system: {all_elements}", log_widget)
        else:
            log_info(f"Elements: {all_elements}", log_widget)
        
        # Get competing phases from MP
        if not HAS_NEW_MP_API or not api_key:
            log_warning("No MP API - using placeholder chemical potentials", log_widget)
            bulk_E_per_atom = bulk_energy / len(bulk_structure)
            chempots = {el: bulk_E_per_atom / len(all_elements) for el in all_elements}
            
            chempot_dict = {"Placeholder": chempots}
            return chempot_dict, None
        
        log_info("Searching Materials Project for competing phases...", log_widget)
        
        with NewMPRester(api_key) as mpr:
            chemsys = '-'.join(all_elements)
            log_plain(f"Chemical system: {chemsys}", log_widget)
            
            try:
                entries = mpr.get_entries_in_chemsys(all_elements)
                log_success(f"Retrieved {len(entries)} entries from MP", log_widget)
            except:
                log_warning("Using fallback entry retrieval...", log_widget)
                docs = mpr.summary.search(
                    chemsys=chemsys,
                    fields=["material_id"]
                )
                
                entries = []
                for doc in docs:
                    try:
                        entry = mpr.get_entry_by_material_id(doc.material_id)
                        entries.append(entry)
                    except:
                        pass
                
                log_success(f"Retrieved {len(entries)} entries", log_widget)
        
        # Add bulk entry
        bulk_entry = ComputedStructureEntry(bulk_structure, bulk_energy)
        all_entries = entries + [bulk_entry]
        
        # Convert to PDEntry
        pd_entries = []
        for entry in all_entries:
            pd_entry = PDEntry(entry.composition, entry.energy)
            pd_entries.append(pd_entry)
        
        # Create phase diagram
        phase_diagram = PhaseDiagram(pd_entries)
        
        log_success("Phase diagram created", log_widget)
        log_plain(f"Total entries: {len(pd_entries)}, Stable entries: {len(phase_diagram.stable_entries)}", log_widget)
        
        # Get elemental references
        elem_refs = {}
        for elem_str in all_elements:
            elem_entries = [e for e in all_entries if 
                           len(e.composition.elements) == 1 and 
                           str(e.composition.elements[0]) == elem_str]
            if elem_entries:
                elem_refs[elem_str] = elem_entries[0].energy_per_atom
            else:
                log_warning(f"No elemental reference for {elem_str}, using default", log_widget)
                elem_refs[elem_str] = bulk_energy / len(bulk_structure) / len(all_elements)
        
        log_info("Elemental references:", log_widget)
        for elem, E in elem_refs.items():
            log_plain(f"  E({elem}) = {E:.4f} eV/atom", log_widget)
        
        # Calculate chemical potential limits
        chempot_dict = {}
        
        bulk_E_per_atom = bulk_energy / len(bulk_structure)
        bulk_comp_dict = comp.get_el_amt_dict()
        total_atoms = sum(bulk_comp_dict.values())
        
        if len(bulk_elements) == 2:
            # Binary system - simple limits
            elem_A, elem_B = sorted(bulk_elements)
            
            x_A = bulk_comp_dict.get(elem_A, 0) / total_atoms
            x_B = bulk_comp_dict.get(elem_B, 0) / total_atoms
            
            # A-rich limit
            mu_A_rich = elem_refs[elem_A]
            mu_B_rich = (bulk_E_per_atom - x_A * mu_A_rich) / x_B
            
            limit_A = {elem_A: mu_A_rich, elem_B: mu_B_rich}
            
            # Add impurity chemical potentials at elemental reference
            if impurity_elements:
                for imp in impurity_elements:
                    limit_A[imp] = elem_refs.get(imp, 0.0)
            
            chempot_dict[f"{elem_A}-rich"] = limit_A
            
            # B-rich limit
            mu_B_rich2 = elem_refs[elem_B]
            mu_A_rich2 = (bulk_E_per_atom - x_B * mu_B_rich2) / x_A
            
            limit_B = {elem_A: mu_A_rich2, elem_B: mu_B_rich2}
            
            if impurity_elements:
                for imp in impurity_elements:
                    limit_B[imp] = elem_refs.get(imp, 0.0)
            
            chempot_dict[f"{elem_B}-rich"] = limit_B
            
            log_success(f"Calculated {len(chempot_dict)} chemical potential limits (binary system)", log_widget)
            
        else:
            # Multi-component - use stable phases
            log_info("Multi-component system - using phase diagram vertices", log_widget)
            
            # ✅ FIX: Convert set to list before slicing
            stable_entries_list = list(phase_diagram.stable_entries)
            
            for i, stable_entry in enumerate(stable_entries_list[:10]):
                mu_dict = {}
                for elem in all_elements:
                    mu_dict[elem] = elem_refs.get(elem, 0.0)
                
                chempot_dict[f"Limit_{i+1}"] = mu_dict
            
            log_success(f"Created {len(chempot_dict)} limits", log_widget)
        
        # Display results
        log_section("CHEMICAL POTENTIAL LIMITS", log_widget)
        
        # Create summary table
        rows = []
        for limit_name, chempots in chempot_dict.items():
            log_info(f"{limit_name}:", log_widget)
            row = {'Limit': limit_name}
            
            for elem in all_elements:
                mu = chempots.get(elem, None)
                if mu is not None:
                    log_plain(f"  μ({elem}) = {mu:.4f} eV", log_widget)
                    row[f'μ({elem})'] = mu
                else:
                    log_plain(f"  μ({elem}) = N/A", log_widget)
                    row[f'μ({elem})'] = None
            
            rows.append(row)
        
        chempot_table = pd.DataFrame(rows)
        
        log_success("Chemical potential calculation complete!", log_widget)
        
        return chempot_dict, chempot_table
        
    except Exception as e:
        log_error(f"Chemical potential calculation failed: {e}", log_widget)
        import traceback
        traceback.print_exc()
        return None, None

# =====================================================================================
# M3GNET MODEL MANAGEMENT
# =====================================================================================

class M3GNetModelManager:
    """Manage M3GNet universal potential loading"""
    
    _model_cache = None
    _calculator_cache = None
    
    # Updated model name for 2025.1 version from materialyzeai
    MODEL_NAME = "M3GNet-MatPES-PBE-v2025.1-PES"
    GITHUB_BASE_URL = "https://raw.githubusercontent.com/materialyzeai/matgl/main/pretrained_models"
    REQUIRED_FILES = ["model.pt", "model.json", "state.pt"]  # Correct files!
    
    @classmethod
    def check_model_files_exist(cls):
        """Check if M3GNet model files exist locally"""
        model_dir = Path.home() / ".cache" / "matgl" / cls.MODEL_NAME
        
        # Check for all required files
        all_exist = all((model_dir / f).exists() for f in cls.REQUIRED_FILES)
        
        return all_exist, model_dir
    
    @classmethod
    def download_model_files(cls, log_widget=None):
        """Download model files from GitHub"""
        model_dir = Path.home() / ".cache" / "matgl" / cls.MODEL_NAME
        model_dir.mkdir(parents=True, exist_ok=True)
        
        log_info(f"Downloading M3GNet model to: {model_dir}", log_widget)
        log_plain("", log_widget)
        
        for filename in cls.REQUIRED_FILES:
            file_path = model_dir / filename
            
            if file_path.exists():
                log_success(f"{filename} already exists", log_widget)
                continue
            
            url = f"{cls.GITHUB_BASE_URL}/{cls.MODEL_NAME}/{filename}"
            
            try:
                log_info(f"Downloading {filename}...", log_widget)
                
                import requests
                response = requests.get(url, timeout=120)
                
                if response.status_code == 200:
                    file_path.write_bytes(response.content)
                    size_mb = len(response.content) / (1024 * 1024)
                    log_success(f"{filename} downloaded ({size_mb:.1f} MB)", log_widget)
                else:
                    raise Exception(f"HTTP {response.status_code}")
                    
            except Exception as e:
                log_error(f"Failed to download {filename}: {e}", log_widget)
                raise
        
        log_plain("", log_widget)
        log_success("All model files downloaded successfully!", log_widget)
    
    @classmethod
    def load_m3gnet(cls, log_widget=None):
        """Load M3GNet universal potential with auto-download"""
        if cls._calculator_cache is not None:
            log_success("Using cached M3GNet calculator", log_widget)
            return cls._calculator_cache
        
        try:
            # Check if model files exist
            files_exist, model_dir = cls.check_model_files_exist()
            
            if not files_exist:
                log_warning("M3GNet model files not found - downloading...", log_widget)
                log_plain("", log_widget)
                
                try:
                    cls.download_model_files(log_widget)
                except Exception as download_error:
                    log_error("Automatic download failed!", log_widget)
                    log_plain("", log_widget)
                    log_warning("MANUAL DOWNLOAD REQUIRED", log_widget)
                    log_plain("", log_widget)
                    log_plain("Run these commands in your terminal:", log_widget)
                    log_plain("", log_widget)
                    log_plain(f"  mkdir -p ~/.cache/matgl/{cls.MODEL_NAME}", log_widget)
                    log_plain(f"  cd ~/.cache/matgl/{cls.MODEL_NAME}", log_widget)
                    log_plain("", log_widget)
                    for filename in cls.REQUIRED_FILES:
                        log_plain(f"  wget {cls.GITHUB_BASE_URL}/{cls.MODEL_NAME}/{filename}", log_widget)
                    log_plain("", log_widget)
                    log_success("After downloading, click the button again!", log_widget)
                    raise RuntimeError("M3GNet model download failed")
            
            log_info(f"Loading M3GNet Universal Potential ({cls.MODEL_NAME})...", log_widget)
            log_plain(f"Model location: {model_dir}", log_widget)
            
            if cls._model_cache is None:
                potential = load_model(str(model_dir))
                cls._model_cache = potential
            
            calculator = M3GNetCalculator(potential=cls._model_cache)
            cls._calculator_cache = calculator
            
            log_success("M3GNet loaded successfully", log_widget)
            return calculator
            
        except Exception as e:
            if "M3GNet model download failed" in str(e):
                raise
            log_error(f"M3GNet loading failed: {e}", log_widget)
            raise

# =====================================================================================
# STRUCTURE OPERATIONS
# =====================================================================================

def relax_structure_m3gnet(structure, fmax=0.05, steps=500, log_widget=None):
    """Relax structure with M3GNet"""
    try:
        log_info("Converting structure to ASE...", log_widget)
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        
        calc = M3GNetModelManager.load_m3gnet(log_widget=log_widget)
        atoms.calc = calc
        
        log_info(f"Relaxing structure (fmax={fmax}, max_steps={steps})...", log_widget)
        optimizer = BFGS(atoms, logfile=None)
        optimizer.run(fmax=fmax, steps=steps)
        
        E_final = atoms.get_potential_energy()
        relaxed_struct = adaptor.get_structure(atoms)
        
        log_success(f"Relaxation complete: E = {E_final:.6f} eV", log_widget)
        
        return {
            'energy': E_final,
            'energy_per_atom': E_final / len(atoms),
            'structure': relaxed_struct,
            'n_steps': optimizer.nsteps
        }
    except Exception as e:
        log_error(f"Relaxation failed: {e}", log_widget)
        raise

# =====================================================================================
# MOLECULAR DYNAMICS
# =====================================================================================

def run_md_simulation(structure, temperature=300, steps=10000, timestep=2.0, 
                     ensemble='NVT', friction=0.01, T_init=None, log_widget=None, plot_output=None):
    """Run molecular dynamics simulation with improved controls"""
    try:
        log_section("MOLECULAR DYNAMICS SIMULATION", log_widget)
        log_plain("", log_widget)
        log_plain(f"Temperature: {temperature} K", log_widget)
        log_plain(f"Steps: {steps}", log_widget)
        log_plain(f"Timestep: {timestep} fs", log_widget)
        log_plain(f"Ensemble: {ensemble}", log_widget)
        if ensemble == 'Langevin':
            log_plain(f"Friction: {friction} fs⁻¹", log_widget)
        if T_init is not None:
            log_plain(f"Initial Temperature: {T_init} K", log_widget)
        log_plain("", log_widget)
        
        # Convert to ASE
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        
        # Load M3GNet calculator
        calc = M3GNetModelManager.load_m3gnet(log_widget=log_widget)
        atoms.calc = calc
        
        # Pre-relax with relaxed convergence
        log_info("Pre-relaxing structure...", log_widget)
        relaxer = BFGS(atoms, logfile=None)
        try:
            relaxer.run(fmax=0.05, steps=100)
        except:
            pass
        
        # Initialize velocities
        init_temp = T_init if T_init is not None else temperature
        MaxwellBoltzmannDistribution(atoms, temperature_K=init_temp)
        Stationary(atoms)
        ZeroRotation(atoms)
        
        # Setup MD
        if ensemble == 'NVT':
            log_info("Using NVT Berendsen thermostat...", log_widget)
            dyn = NVTBerendsen(atoms, timestep * units.fs, temperature, taut=100*units.fs)
        elif ensemble == 'NPT':
            log_info("Using NPT Berendsen barostat...", log_widget)
            dyn = NPTBerendsen(atoms, timestep * units.fs, temperature, 
                              pressure_au=0.0, taut=100*units.fs, taup=1000*units.fs, 
                              compressibility=4.5e-5)
        elif ensemble == 'Langevin':
            log_info("Using Langevin thermostat...", log_widget)
            dyn = Langevin(atoms, timestep * units.fs, temperature_K=temperature, 
                          friction=friction)
        else:  # NVE
            log_info("Using microcanonical ensemble (NVE)...", log_widget)
            dyn = VelocityVerlet(atoms, timestep * units.fs)
        
        # Storage for trajectory
        trajectory_data = {
            'time': [],
            'energy': [],
            'kinetic': [],
            'potential': [],
            'temperature': [],
            'positions': []
        }
        
        # Setup live plotting if requested
        fig, axes = None, None
        lines = {}
        
        if plot_output is not None:
            with plot_output:
                clear_output(wait=True)
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
                fig.suptitle("MD Simulation", fontsize=22, fontweight='bold')
                axes = {'epot': ax1, 'ekin': ax2, 'etot': ax3, 'temp': ax4}
                
                lines['epot'], = ax1.plot([], [], 'b-', linewidth=2.5)
                lines['ekin'], = ax2.plot([], [], 'r-', linewidth=2.5)
                lines['etot'], = ax3.plot([], [], 'g-', linewidth=2.5)
                lines['temp'], = ax4.plot([], [], 'm-', linewidth=2.5)
                
                # Y-axis labels
                ax1.set_ylabel('Potential Energy (eV)', fontsize=18, fontweight='bold')
                ax2.set_ylabel('Kinetic Energy (eV)', fontsize=18, fontweight='bold')
                ax3.set_ylabel('Total Energy (eV)', fontsize=18, fontweight='bold')
                ax4.set_ylabel('Temperature (K)', fontsize=18, fontweight='bold')
                
                # X-axis labels
                ax1.set_xlabel('Step', fontsize=18, fontweight='bold')
                ax2.set_xlabel('Step', fontsize=18, fontweight='bold')
                ax3.set_xlabel('Step', fontsize=18, fontweight='bold')
                ax4.set_xlabel('Step', fontsize=18, fontweight='bold')
                
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=16)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                display(fig)
                plt.close(fig)
        
        def update_plot():
            if plot_output is None or len(trajectory_data['time']) < 2:
                return
            
            df = pd.DataFrame({
                'step': trajectory_data['time'],
                'E_pot': trajectory_data['potential'],
                'E_kin': trajectory_data['kinetic'],
                'E_tot': trajectory_data['energy'],
                'T': trajectory_data['temperature']
            })
            
            lines['epot'].set_data(df['step'], df['E_pot'])
            lines['ekin'].set_data(df['step'], df['E_kin'])
            lines['etot'].set_data(df['step'], df['E_tot'])
            lines['temp'].set_data(df['step'], df['T'])
            
            for ax in axes.values():
                ax.relim()
                ax.autoscale_view()
            
            with plot_output:
                clear_output(wait=True)
                display(fig)
        
        # Determine logging interval based on total steps
        if steps <= 1000:
            log_interval = 100
        elif steps <= 5000:
            log_interval = 500
        else:
            log_interval = 1000
        
        # Run MD
        log_info(f"Running MD for {steps} steps (logging every {log_interval} steps)...", log_widget)
        
        for i in range(steps):
            dyn.run(1)
            
            if i % 100 == 0:
                E_kin = atoms.get_kinetic_energy()
                E_pot = atoms.get_potential_energy()
                E_tot = E_kin + E_pot
                T = atoms.get_temperature()
                
                trajectory_data['time'].append(i)  # FIXED: Store step number, not time
                trajectory_data['energy'].append(E_tot)
                trajectory_data['kinetic'].append(E_kin)
                trajectory_data['potential'].append(E_pot)
                trajectory_data['temperature'].append(T)
                trajectory_data['positions'].append(atoms.get_positions().copy())
                
                # Safety check
                if i > 1000:
                    if abs(E_pot) > 1e4 or T > 20 * temperature:
                        raise RuntimeError(f"MD UNSTABLE! Epot={E_pot:.2f}, T={T:.1f}K")
                
                # Log at determined interval
                if i % log_interval == 0:
                    log_plain(f"Step {i:6d} | E_pot={E_pot:10.5f} eV | T={T:8.2f} K", log_widget)
                    update_plot()
        
        # Log final step
        E_kin = atoms.get_kinetic_energy()
        E_pot = atoms.get_potential_energy()
        T = atoms.get_temperature()
        log_plain(f"Step {steps:6d} (FINAL) | E_pot={E_pot:10.5f} eV | T={T:8.2f} K", log_widget)
        
        log_success("MD simulation complete!", log_widget)
        
        final_struct = adaptor.get_structure(atoms)
        
        return {
            'energy_log': trajectory_data,
            'final_structure': final_struct,
            'atoms': atoms
        }
        
    except Exception as e:
        log_error(f"MD simulation failed: {e}", log_widget)
        import traceback
        traceback.print_exc()
        raise

# =====================================================================================
# DEFECT FORMATION ENERGY
# =====================================================================================

def calculate_defect_formation_energy_silent(
    E_defect,
    E_bulk,
    chemical_potentials,
    defect_notation
):
    """Calculate defect formation energy for neutral charge state - SILENT version"""
    
    # Parse defect to determine stoichiometry change
    delta_n = {}
    
    for part in defect_notation.split('+'):
        part = part.strip()
        
        if part.startswith('V_'):  # Vacancy
            element = part[2:]
            delta_n[element] = delta_n.get(element, 0) + 1
            
        elif part.endswith('_i'):  # Interstitial
            element = part[:-2]
            delta_n[element] = delta_n.get(element, 0) - 1
            
        elif '_' in part:  # Substitution
            dopant, host = part.split('_', 1)
            delta_n[host] = delta_n.get(host, 0) + 1
            delta_n[dopant] = delta_n.get(dopant, 0) - 1
    
    # Calculate chemical potential contribution
    mu_sum = 0.0
    
    for element, n in delta_n.items():
        if element in chemical_potentials:
            mu = chemical_potentials[element]
            mu_sum += n * mu
    
    # Formation energy
    H_f = E_defect - E_bulk + mu_sum
    
    return H_f

# =====================================================================================
# FILE DOWNLOAD UTILITIES (Document 2 Style)
# =====================================================================================

def create_download_package(uds_state):
    """Create ZIP package with all results including MD"""
    
    # Get compound name for filename
    compound_name = "results"
    if uds_state.get('bulk_structure'):
        comp = uds_state['bulk_structure'].composition.reduced_formula
        compound_name = comp.replace(' ', '_')
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create main results folder
    main_folder = f"{compound_name}_results"
    main_path = Path(temp_dir) / main_folder
    main_path.mkdir()
    
    files_created = []
    
    # 1. Save bulk structure if available
    if uds_state.get('bulk_structure'):
        bulk_cif = main_path / "bulk_structure.cif"
        CifWriter(uds_state['bulk_structure']).write_file(str(bulk_cif))
        files_created.append("bulk_structure.cif")
    
    # 2. Save defect structures if available
    if uds_state.get('defect_structures'):
        defects_folder = main_path / "defect_structures"
        defects_folder.mkdir()
        
        for defect_name, struct in uds_state['defect_structures'].items():
            # Keep underscore in filename (not hyphen)
            safe_name = defect_name.replace('+', '_plus_')
            cif_path = defects_folder / f"{safe_name}.cif"
            CifWriter(struct).write_file(str(cif_path))
            files_created.append(f"defect_structures/{safe_name}.cif")
    
    # 3. Create comprehensive CSV with all data
    if uds_state.get('defect_energies'):
        rows = []
        for defect_str, E_defect in uds_state['defect_energies'].items():
            row = {'Defect': defect_str, 'E_defect (eV)': E_defect}
            
            if uds_state.get('bulk_energy'):
                H_f = calculate_defect_formation_energy_silent(
                    E_defect,
                    uds_state['bulk_energy'],
                    uds_state.get('chemical_potentials', {}),
                    defect_str
                )
                row['ΔH_f (eV)'] = H_f
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if 'ΔH_f (eV)' in df.columns:
            df = df.sort_values('ΔH_f (eV)')
        
        # Format defect names with subscript in CSV
        def format_defect_name(defect_str):
            parts = []
            for part in defect_str.split('+'):
                if part.startswith('V_'):
                    parts.append(f"V<sub>{part[2:]}</sub>")
                elif part.endswith('_i'):
                    parts.append(f"{part[:-2]}<sub>i</sub>")
                elif '_' in part:
                    A, B = part.split('_', 1)
                    parts.append(f"{A}<sub>{B}</sub>")
                else:
                    parts.append(part)
            return '+'.join(parts)
        
        df['Defect_Formatted'] = df['Defect'].apply(format_defect_name)
        
        csv_path = main_path / "formation_energies.csv"
        df.to_csv(csv_path, index=False)
        files_created.append("formation_energies.csv")
    
    # 4. Save MD results if available
    if uds_state.get('md_results'):
        md_folder = main_path / "md_simulations"
        md_folder.mkdir()
        
        for md_key, md_data in uds_state['md_results'].items():
            # Create subfolder for each MD simulation
            safe_key = md_key.replace('+', '_plus_').replace('/', '_')
            md_subfolder = md_folder / safe_key
            md_subfolder.mkdir()
            
            # Save energy trajectory
            if 'energy_log' in md_data:
                energy_df = pd.DataFrame({
                    'Step': md_data['energy_log']['time'],
                    'E_total (eV)': md_data['energy_log']['energy'],
                    'E_kinetic (eV)': md_data['energy_log']['kinetic'],
                    'E_potential (eV)': md_data['energy_log']['potential'],
                    'Temperature (K)': md_data['energy_log']['temperature']
                })
                csv_path = md_subfolder / "trajectory.csv"
                energy_df.to_csv(csv_path, index=False)
                files_created.append(f"md_simulations/{safe_key}/trajectory.csv")
            
            # Save final structure
            if 'final_structure' in md_data:
                cif_path = md_subfolder / "final_structure.cif"
                CifWriter(md_data['final_structure']).write_file(str(cif_path))
                files_created.append(f"md_simulations/{safe_key}/final_structure.cif")
    
    # 5. Create README file
    readme_content = f"""Universal Defect Simulator Results
=====================================
Compound: {compound_name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Files Included:
{chr(10).join(f'- {f}' for f in files_created)}

Description:
- bulk_structure.cif: Relaxed bulk structure
- defect_structures/: Individual defect structures
- formation_energies.csv: Defect formation energies
- md_simulations/: MD trajectory data and final structures
  - Each subfolder contains:
    - trajectory.csv: Energy and temperature vs step
    - final_structure.cif: Structure after MD equilibration

Defect Notation:
V_X: Vacancy of element X
X_i: Interstitial of element X
A_B: Substitution of B by A
A_B+C_i: Complex defect

MD Data:
- Step: MD step number
- E_total: Total energy (kinetic + potential)
- E_kinetic: Kinetic energy
- E_potential: Potential energy
- Temperature: Instantaneous temperature
"""
    
    readme_path = main_path / "README.txt"
    readme_path.write_text(readme_content)
    
    # Create ZIP file
    zip_filename = f"{compound_name}_results.zip"
    zip_path = Path(temp_dir) / zip_filename
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in main_path.rglob("*"):
            if file.is_file():
                arcname = str(file.relative_to(main_path))
                zipf.write(file, arcname)
    
    # Read ZIP file
    with open(zip_path, 'rb') as f:
        zip_data = f.read()
    
    # Clean up temp files
    shutil.rmtree(temp_dir)
    
    return zip_data, zip_filename




uds_header = widgets.HTML(f"""
<div style="background: linear-gradient(135deg, #9ca3af 0%, #6b7280 100%);
            padding: 25px; border-radius: 15px; margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
    <div style="color: white; font-size: 36px; font-weight: 900; text-align: center; margin-bottom: 10px;">
        🔬 Universal Defect Simulator
    </div>
    <div style="color: #e5e7eb; font-size: 18px; text-align: center;">
        M3GNet Universal Potential
    </div>
</div>
""")

uds_guidelines = widgets.HTML("""
<div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
            padding: 20px; border-radius: 12px; margin-bottom: 20px;
            border: 2px solid #e5e7eb; border-left: 4px solid #9ca3af;">
    <div style="font-size: 18px; font-weight: bold; color:#4b5563; margin-bottom: 15px;">
        <span style="font-size: 22px; margin-right: 8px;">🗺️</span>
        Workflow Guide
    </div>
    <div style="color: #6b7280; line-height: 1.8; font-size: 14px;">
        <div style="margin: 8px 0;">
            <span style="font-weight: bold; color: #6b7280;">📥 Phase 1:</span>
            Structure Setup (MP / Periodic Table / CIF Upload)
        </div>
        <div style="margin: 8px 0;">
            <span style="font-weight: bold; color: #6b7280;">❄️ Phase 2:</span>
            Bulk Relaxation at 0K
        </div>
        <div style="margin: 8px 0;">
            <span style="font-weight: bold; color: #6b7280;">⚛️ Phase 3:</span>
            Defect Relaxation at 0K (Formation Energies)
        </div>
        <div style="margin: 8px 0;">
            <span style="font-weight: bold; color: #6b7280;">🔥 Phase 4:</span>
            MD Simulations (Bulk + Defects at Finite T)
        </div>
    </div>
</div>
""")

# =====================================================================================
# PERIODIC TABLE WIDGET
# =====================================================================================

def create_periodic_table_widget():
    """Create an interactive periodic table for element selection"""
    
    uds_selected_elements = []
    
    element_buttons = {}
    
    def on_element_click(b):
        element = b.description
        if element in uds_selected_elements:
            uds_selected_elements.remove(element)
            b.style.button_color = None
        else:
            uds_selected_elements.append(element)
            b.style.button_color = '#d1d5db'
        
        uds_selected_elements_display.value = ' '.join(sorted(uds_selected_elements))
    
    rows = []
    for row in PERIODIC_TABLE:
        row_buttons = []
        for elem in row:
            if elem:
                btn = widgets.Button(
                    description=elem,
                    layout=widgets.Layout(width='45px', height='35px'),
                    tooltip=elem,
                    style={'button_color': None}
                )
                btn.on_click(on_element_click)
                element_buttons[elem] = btn
                row_buttons.append(btn)
            else:
                row_buttons.append(widgets.HTML(value='', layout=widgets.Layout(width='45px', height='35px')))
        
        rows.append(widgets.HBox(row_buttons, layout=widgets.Layout(margin='2px')))
    
    ptable_grid = widgets.VBox(rows)
    
    uds_selected_elements_display = widgets.Text(
        value='',
        placeholder='Selected elements will appear here',
        description='Selected:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%')
    )
    
    clear_btn = widgets.Button(
        description='Clear Selection',
        button_style='',
        layout=widgets.Layout(width='150px')
    )
    
    def on_clear(b):
        uds_selected_elements.clear()
        for btn in element_buttons.values():
            btn.style.button_color = None
        uds_selected_elements_display.value = ''
    
    clear_btn.on_click(on_clear)
    
    return widgets.VBox([
        widgets.HTML("<b>Select Elements:</b>"),
        ptable_grid,
        uds_selected_elements_display,
        clear_btn
    ]), uds_selected_elements

uds_periodic_table_widget, uds_selected_elements_list = create_periodic_table_widget()

# =====================================================================================
# INPUT METHOD SELECTION
# =====================================================================================

uds_input_method = widgets.ToggleButtons(
    options=[
        ('🔬 Materials Project', 'mp'),
        ('🔢 Periodic Table', 'ptable'),
        ('📁 Upload CIF', 'cif')
    ],
    value='mp',
    description='Input:',
    style={'description_width': '80px'}
)

# Materials Project panel
uds_mp_api_key = widgets.Text(
    placeholder='Enter MP API key (or set MP_API_KEY env var)',
    description='API Key:',
    style={'description_width': '80px'},
    layout=widgets.Layout(width='100%')
)

uds_mp_search = widgets.Text(
    placeholder='e.g., mp-149, CdTe, Si',
    description='Search:',
    style={'description_width': '80px'},
    layout=widgets.Layout(width='100%')
)

uds_mp_button = widgets.Button(
    description='🔍 Search MP',
    button_style='',
    layout=widgets.Layout(width='200px')
)

uds_mp_results = widgets.SelectMultiple(
    options=[],
    description='Results:',
    disabled=False,
    layout=widgets.Layout(width='100%', height='200px')
)

uds_mp_load_button = widgets.Button(
    description='✓ Load Selected',
    button_style='',
    layout=widgets.Layout(width='200px')
)

uds_view_3d_button = widgets.Button(
    description='🔬 View 3D',
    button_style='info',
    layout=widgets.Layout(width='200px')
)

uds_mp_panel = widgets.VBox([
    widgets.HTML("<b>Materials Project Search</b>"),
    uds_mp_api_key,
    uds_mp_search,
    uds_mp_button,
    widgets.HTML("<div style='margin-top:10px;'><b>Search Results:</b></div>"),
    uds_mp_results,
    widgets.HBox([uds_mp_load_button, uds_view_3d_button])
])

# Periodic Table panel
uds_ptable_search_button = widgets.Button(
    description='🔍 Search Compounds',
    button_style='',
    layout=widgets.Layout(width='200px')
)

uds_ptable_results = widgets.SelectMultiple(
    options=[],
    description='Compounds:',
    disabled=False,
    layout=widgets.Layout(width='100%', height='200px')
)

uds_ptable_load_button = widgets.Button(
    description='✓ Load Selected',
    button_style='',
    layout=widgets.Layout(width='200px')
)

uds_ptable_panel = widgets.VBox([
    uds_periodic_table_widget,
    widgets.HTML("<div style='margin-top:15px;'><i>Select elements above, then click 'Search Compounds'</i></div>"),
    uds_ptable_search_button,
    widgets.HTML("<div style='margin-top:10px;'><b>Found Compounds:</b></div>"),
    uds_ptable_results,
    uds_ptable_load_button
])

# CIF upload panel
uds_cif_upload = widgets.FileUpload(
    accept='.cif',
    multiple=False,
    description='Upload CIF:',
    style={'description_width': '100px'}
)

uds_clear_upload_button = widgets.Button(
    description='🗑️ Clear Uploaded Files',
    button_style='',
    layout=widgets.Layout(width='200px')
)

uds_cif_panel = widgets.VBox([
    widgets.HTML("<b>Upload CIF File</b>"),
    uds_cif_upload,
    uds_clear_upload_button
])

# Dynamic panel switcher
uds_input_panel = widgets.VBox([uds_mp_panel])

def uds_on_input_method_change(change):
    method = change['new']
    if method == 'mp':
        uds_input_panel.children = [uds_mp_panel]
    elif method == 'ptable':
        uds_input_panel.children = [uds_ptable_panel]
    elif method == 'cif':
        uds_input_panel.children = [uds_cif_panel]

uds_input_method.observe(uds_on_input_method_change, names='value')

# Structure display
uds_structure_info = widgets.HTML(f"""
<div style="padding: 15px; background: {UDS_COLORS['panel_bg']}; 
            border-radius: 8px; border-left: 4px solid {UDS_COLORS['panel_border']};">
    <b>Loaded Structure:</b> None
</div>
""")

# Supercell multipliers
uds_supercell_info = widgets.HTML("")

uds_supercell_a = widgets.IntText(value=1, description='a:', style={'description_width': '30px'}, layout=widgets.Layout(width='100px'))
uds_supercell_b = widgets.IntText(value=1, description='b:', style={'description_width': '30px'}, layout=widgets.Layout(width='100px'))
uds_supercell_c = widgets.IntText(value=1, description='c:', style={'description_width': '30px'}, layout=widgets.Layout(width='100px'))

uds_supercell_create = widgets.Button(
    description='Create Supercell',
    button_style='',
    layout=widgets.Layout(width='200px')
)

uds_supercell_panel = widgets.VBox([
    uds_supercell_info,
    widgets.HTML("<b>Supercell Multipliers:</b>"),
    widgets.HBox([uds_supercell_a, uds_supercell_b, uds_supercell_c]),
    uds_supercell_create
])

# Phase 2: Bulk Relaxation
uds_phase2_fmax = widgets.FloatSlider(
    value=0.05,
    min=0.01,
    max=0.2,
    step=0.01,
    description='Fmax (eV/Å):',
    readout_format='.3f',
    style={'description_width': '100px'}
)

uds_phase2_steps = widgets.IntSlider(
    value=500,
    min=50,
    max=2000,
    step=50,
    description='Max Steps:',
    style={'description_width': '100px'}
)

uds_phase2_button = widgets.Button(
    description='🧊 Relax Bulk (0K)',
    button_style='',
    layout=widgets.Layout(width='100%', height='50px'),
    style={'font_weight': 'bold'},
    disabled=True
)

# Phase 3: Defect Relaxation
uds_defect_list = widgets.Textarea(
    value='',
    placeholder='Enter defects (one per line):\nV_Cd  → Vacancy (+μ_Cd)\nAs_i  → Interstitial (-μ_As)\nAs_Te → Substitution (+μ_Te - μ_As)\nAs_Te+Cl_i → Complex defect',
    description='Defects:',
    style={'description_width': '80px'},
    layout=widgets.Layout(width='100%', height='150px')
)

uds_phase3_fmax = widgets.FloatSlider(
    value=0.05,
    min=0.01,
    max=0.2,
    step=0.01,
    description='Fmax (eV/Å):',
    readout_format='.3f',
    style={'description_width': '100px'}
)

uds_phase3_steps = widgets.IntSlider(
    value=500,
    min=50,
    max=2000,
    step=50,
    description='Max Steps:',
    style={'description_width': '100px'}
)

uds_phase3_button = widgets.Button(
    description='⚛️ Relax Defects (0K)',
    button_style='',
    layout=widgets.Layout(width='100%', height='50px'),
    style={'font_weight': 'bold'},
    disabled=True
)

# Phase 4: MD Simulations (IMPROVED)
uds_md_target = widgets.RadioButtons(
    options=['Bulk Structure', 'Defect Structures', 'Both'],
    value='Bulk Structure',
    description='Target:',
    style={'description_width': '100px'}
)

uds_md_T_init = widgets.IntText(
    value=300,
    description='T_init (K):',
    style={'description_width': '100px'},
    layout=widgets.Layout(width='200px')
)

uds_md_T_final = widgets.IntText(
    value=300,
    description='T_final (K):',
    style={'description_width': '100px'},
    layout=widgets.Layout(width='200px')
)

uds_md_steps = widgets.IntSlider(
    value=10000,
    min=100,  # Changed from 1000 to 100
    max=100000,
    step=100,
    description='MD Steps:',
    style={'description_width': '100px'}
)

uds_md_timestep = widgets.FloatSlider(
    value=2.0,
    min=0.5,
    max=5.0,
    step=0.5,
    description='Timestep (fs):',
    style={'description_width': '100px'}
)

uds_md_ensemble = widgets.Dropdown(
    options=['NVT', 'NVE', 'NPT', 'Langevin'],
    value='NVT',
    description='Ensemble:',
    style={'description_width': '100px'}
)

uds_md_friction = widgets.FloatSlider(
    value=0.01,
    min=0.001,
    max=0.1,
    step=0.001,
    description='Friction (fs⁻¹):',
    readout_format='.3f',
    style={'description_width': '100px'}
)

uds_phase4_button = widgets.Button(
    description='🔥 Run MD Simulations',
    button_style='',
    layout=widgets.Layout(width='100%', height='50px'),
    style={'font_weight': 'bold'},
    disabled=True
)

# Progress and status
uds_progress = widgets.IntProgress(
    value=0, min=0, max=1,
    description="Progress:",
    style={'bar_color': '#9ca3af', 'description_width': '70px'}
)

uds_status = widgets.HTML(
    f"<div style='padding: 10px; background: {UDS_COLORS['panel_bg']}; border-radius: 8px;'>"
    "<b>Status:</b> Ready - Load structure to begin</div>"
)

# Output areas
uds_log_output = widgets.Output(
    layout={'border': f'2px solid {UDS_COLORS["panel_border"]}', 'border_radius': '8px',
            'padding': '15px', 'background_color': '#fafafa',
            'height': '400px', 'overflow': 'auto'}
)

uds_results_output = widgets.Output(
    layout={'border': f'2px solid {UDS_COLORS["panel_border"]}', 'border_radius': '8px',
            'padding': '10px'}
)

uds_download_output = widgets.Output()

# Global state
uds_global_state = {
    'structure': None,
    'structure_list': [],
    'bulk_energy': None,
    'bulk_structure': None,
    'defect_energies': {},
    'defect_structures': {},
    'chemical_potentials': None,
    'all_chempot_limits': None,
    'chempot_table': None,
    'uploaded_cif_data': None,
    'md_results': {}
}

# ===================================================================================
# EVENT HANDLERS
# =====================================================================================

def uds_on_mp_search_clicked(b):
    """Search Materials Project"""
    uds_log_output.clear_output(wait=False)
    
    with uds_log_output:
        query = uds_mp_search.value.strip()
        if not query:
            log_error("Enter a search term!")
            return
        
        api_key = uds_mp_api_key.value.strip() or None
        
        try:
            results = search_mp_structures(query, api_key, log_func=lambda x: log_info(x))
            
            uds_global_state['structure_list'] = results
            
            options = []
            for mp_id, formula, energy, struct in results[:12]:  # Limit to 12
                label = f"{mp_id}: {formula} (E={energy:.4f} eV/atom)"
                options.append((label, mp_id))
            
            uds_mp_results.options = options
            
            log_success(f"Found {len(results)} structures (showing top 12)")
            log_info("Select one and click 'Load Selected'")
            
        except Exception as e:
            log_error(f"Search failed: {e}")
            import traceback
            traceback.print_exc()

uds_mp_button.on_click(uds_on_mp_search_clicked)

def uds_on_mp_load_clicked(b):
    """Load selected MP structure"""
    uds_log_output.clear_output(wait=False)
    
    with uds_log_output:
        selected = uds_mp_results.value
        if not selected:
            log_error("Select a structure first!")
            return
        
        mp_id = selected[0] if isinstance(selected, tuple) else selected
        
        for mid, formula, energy, struct in uds_global_state['structure_list']:
            if mid == mp_id:
                # Convert to conventional if primitive
                struct_info = get_structure_info(struct)
                
                if struct_info['is_primitive']:
                    log_info("Converting primitive cell to conventional...")
                    struct = get_conventional_structure(struct)
                
                uds_global_state['structure'] = struct
                
                is_valid, msg, mult = validate_supercell_size(struct, min_size=15.0)
                
                if is_valid:
                    log_success(msg)
                else:
                    log_warning(msg)
                
                comp = struct.composition.reduced_formula
                sg = struct.get_space_group_info()[0]
                a, b, c = struct.lattice.abc
                
                uds_structure_info.value = f"""
                <div style="padding: 15px; background: {UDS_COLORS['success_bg']}; 
                            border-radius: 8px; border-left: 4px solid {UDS_COLORS['success_border']};">
                    <b>✓ Loaded:</b> {comp} ({mp_id})<br>
                    <b>Atoms:</b> {len(struct)}<br>
                    <b>Spacegroup:</b> {sg}<br>
                    <b>Lattice:</b> a={a:.2f}Å, b={b:.2f}Å, c={c:.2f}Å
                </div>
                """
                
                if is_valid:
                    uds_supercell_info.value = f"""
                    <div style='padding: 10px; background: {UDS_COLORS['success_bg']}; 
                                border-radius: 6px; margin-bottom: 10px; 
                                border-left: 3px solid {UDS_COLORS['success_border']}'>
                        ✅ Cell size OK
                    </div>
                    """
                    uds_phase2_button.disabled = False
                    uds_status.value = f"<div style='padding: 10px; background: {UDS_COLORS['success_bg']}; border-radius: 8px;'><b>Status:</b> Structure loaded - Ready for Phase 2</div>"
                else:
                    uds_supercell_info.value = f"""
                    <div style='padding: 10px; background: {UDS_COLORS['warning_bg']}; 
                                border-radius: 6px; margin-bottom: 10px;
                                border-left: 3px solid {UDS_COLORS['warning_border']}'>
                        ⚠️ {msg}<br>
                        <b>Suggested multipliers set below!</b>
                    </div>
                    """
                    uds_supercell_a.value = mult[0]
                    uds_supercell_b.value = mult[1]
                    uds_supercell_c.value = mult[2]
                    uds_status.value = f"<div style='padding: 10px; background: {UDS_COLORS['warning_bg']}; border-radius: 8px;'><b>Status:</b> Create supercell first</div>"
                
                break

uds_mp_load_button.on_click(uds_on_mp_load_clicked)

def uds_on_view_3d_clicked(b):
    """View structure in 3D"""
    uds_results_output.clear_output(wait=False)
    
    with uds_results_output:
        if uds_global_state['structure'] is None:
            log_error("Load a structure first!")
            return
        
        struct = uds_global_state['structure']
        comp = struct.composition.reduced_formula
        visualize_structure_3d(struct, title=f"Structure: {comp}", output_widget=uds_results_output)

uds_view_3d_button.on_click(uds_on_view_3d_clicked)

def uds_on_ptable_search_clicked(b):
    """Search for compounds by elements"""
    uds_log_output.clear_output(wait=False)
    
    with uds_log_output:
        if not uds_selected_elements_list:
            log_error("Select elements first from the periodic table!")
            return
        
        api_key = uds_mp_api_key.value.strip() or None
        
        try:
            results = search_compounds_by_elements(uds_selected_elements_list, api_key, log_func=lambda x: log_info(x))
            
            if not results:
                log_error(f"No compounds found with elements: {uds_selected_elements_list}")
                return
            
            uds_global_state['structure_list'] = results
            
            options = []
            for mp_id, formula, energy, struct in results[:12]:  # Limit to 12
                n_elem = len(struct.composition.elements)
                label = f"{formula} ({mp_id}, {n_elem} elements, E={energy:.4f} eV/atom)"
                options.append((label, mp_id))
            
            uds_ptable_results.options = options
            
            log_success(f"Found {len(results)} compounds (showing top 12)")
            
        except Exception as e:
            log_error(f"Search failed: {e}")

uds_ptable_search_button.on_click(uds_on_ptable_search_clicked)

def uds_on_ptable_load_clicked(b):
    """Load selected compound from periodic table search"""
    uds_log_output.clear_output(wait=False)
    
    with uds_log_output:
        selected = uds_ptable_results.value
        if not selected:
            log_error("Select a compound first!")
            return
        
        mp_id = selected[0] if isinstance(selected, tuple) else selected
        
        for mid, formula, energy, struct in uds_global_state['structure_list']:
            if mid == mp_id:
                # Convert to conventional if primitive
                struct_info = get_structure_info(struct)
                
                if struct_info['is_primitive']:
                    log_info("Converting primitive cell to conventional...")
                    struct = get_conventional_structure(struct)
                
                uds_global_state['structure'] = struct
                
                is_valid, msg, mult = validate_supercell_size(struct, min_size=15.0)
                
                if is_valid:
                    log_success(msg)
                else:
                    log_warning(msg)
                
                comp = struct.composition.reduced_formula
                sg = struct.get_space_group_info()[0]
                a, b, c = struct.lattice.abc
                
                uds_structure_info.value = f"""
                <div style="padding: 15px; background: {UDS_COLORS['success_bg']}; 
                            border-radius: 8px; border-left: 4px solid {UDS_COLORS['success_border']};">
                    <b>✓ Loaded:</b> {comp} ({mp_id})<br>
                    <b>Atoms:</b> {len(struct)}<br>
                    <b>Spacegroup:</b> {sg}<br>
                    <b>Lattice:</b> a={a:.2f}Å, b={b:.2f}Å, c={c:.2f}Å
                </div>
                """
                
                if not is_valid:
                    uds_supercell_info.value = f"""
                    <div style='padding: 10px; background: {UDS_COLORS['warning_bg']}; 
                                border-radius: 6px; border-left: 3px solid {UDS_COLORS['warning_border']}'>
                        ⚠️ {msg}<br><b>Suggested multipliers set!</b>
                    </div>
                    """
                    uds_supercell_a.value = mult[0]
                    uds_supercell_b.value = mult[1]
                    uds_supercell_c.value = mult[2]
                else:
                    uds_phase2_button.disabled = False
                    uds_status.value = f"<div style='padding: 10px; background: {UDS_COLORS['success_bg']}; border-radius: 8px;'><b>Status:</b> Ready for Phase 2</div>"
                
                break

uds_ptable_load_button.on_click(uds_on_ptable_load_clicked)

def uds_on_cif_upload_change(change):
    """Handle CIF upload"""
    uds_log_output.clear_output(wait=False)
    
    with uds_log_output:
        if not uds_cif_upload.value:
            return
        
        try:
            uploaded = list(uds_cif_upload.value.values())[0]
            content = uploaded['content']
            
            log_info("Parsing CIF file...")
            cif_string = content.decode('utf-8')
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
                f.write(cif_string)
                temp_path = f.name
            
            from pymatgen.io.cif import CifParser
            parser = CifParser(temp_path)
            struct = parser.get_structures()[0]
            
            os.unlink(temp_path)
            
            # Convert to conventional if primitive
            struct_info = get_structure_info(struct)
            
            if struct_info['is_primitive']:
                log_info("Converting primitive cell to conventional...")
                struct = get_conventional_structure(struct)
            
            uds_global_state['structure'] = struct
            uds_global_state['uploaded_cif_data'] = content
            
            comp = struct.composition.reduced_formula
            sg = struct.get_space_group_info()[0]
            a, b, c = struct.lattice.abc
            
            log_success("CIF loaded successfully!")
            log_plain(f"Formula: {comp}")
            log_plain(f"Atoms: {len(struct)}")
            log_plain(f"Lattice: a={a:.2f}Å, b={b:.2f}Å, c={c:.2f}Å")
            
            uds_structure_info.value = f"""
            <div style="padding: 15px; background: {UDS_COLORS['success_bg']}; 
                        border-radius: 8px; border-left: 4px solid {UDS_COLORS['success_border']};">
                <b>✓ Loaded from CIF:</b> {comp}<br>
                <b>Atoms:</b> {len(struct)}<br>
                <b>Spacegroup:</b> {sg}<br>
                <b>Lattice:</b> a={a:.2f}Å, b={b:.2f}Å, c={c:.2f}Å
            </div>
            """
            
            is_valid, msg, mult = validate_supercell_size(struct, min_size=15.0)
            
            if not is_valid:
                log_warning(msg)
                uds_supercell_a.value = mult[0]
                uds_supercell_b.value = mult[1]
                uds_supercell_c.value = mult[2]
                
                uds_supercell_info.value = f"""
                <div style='padding: 10px; background: {UDS_COLORS['warning_bg']}; 
                            border-radius: 6px; border-left: 3px solid {UDS_COLORS['warning_border']}'>
                    ⚠️ {msg}<br>
                    <b>Click 'Create Supercell' below!</b>
                </div>
                """
            else:
                log_success(msg)
                uds_phase2_button.disabled = False
                uds_status.value = f"<div style='padding: 10px; background: {UDS_COLORS['success_bg']}; border-radius: 8px;'><b>Status:</b> Ready for Phase 2</div>"
            
        except Exception as e:
            log_error(f"CIF upload failed: {e}")
            import traceback
            traceback.print_exc()

uds_cif_upload.observe(uds_on_cif_upload_change, names='value')

def uds_on_clear_upload_clicked(b):
    """Clear uploaded CIF files"""
    uds_log_output.clear_output(wait=False)
    
    with uds_log_output:
        uds_cif_upload.value.clear()
        uds_cif_upload._counter = 0
        
        uds_global_state['uploaded_cif_data'] = None
        
        log_success("Uploaded files cleared!")
        
        uds_structure_info.value = f"""
        <div style="padding: 15px; background: {UDS_COLORS['panel_bg']}; 
                    border-radius: 8px; border-left: 4px solid {UDS_COLORS['panel_border']};">
            <b>Loaded Structure:</b> None
        </div>
        """
        
        uds_supercell_info.value = ""

uds_clear_upload_button.on_click(uds_on_clear_upload_clicked)

def uds_on_supercell_create_clicked(b):
    """Create supercell"""
    uds_log_output.clear_output(wait=False)
    
    with uds_log_output:
        if uds_global_state['structure'] is None:
            log_error("Load a structure first!")
            return
        
        try:
            struct = uds_global_state['structure']
            mult = [uds_supercell_a.value, uds_supercell_b.value, uds_supercell_c.value]
            
            log_info(f"Creating {mult[0]}×{mult[1]}×{mult[2]} supercell...")
            
            supercell = struct * mult
            
            is_valid, msg, _ = validate_supercell_size(supercell, min_size=15.0)
            
            if is_valid:
                log_success(msg)
            else:
                log_warning(msg)
            
            if is_valid:
                uds_global_state['structure'] = supercell
                
                comp = supercell.composition.reduced_formula
                a, b, c = supercell.lattice.abc
                
                uds_structure_info.value = f"""
                <div style="padding: 15px; background: {UDS_COLORS['success_bg']}; 
                            border-radius: 8px; border-left: 4px solid {UDS_COLORS['success_border']};">
                    <b>✓ Supercell Created:</b> {comp}<br>
                    <b>Atoms:</b> {len(supercell)}<br>
                    <b>Lattice:</b> a={a:.2f}Å, b={b:.2f}Å, c={c:.2f}Å
                </div>
                """
                
                uds_supercell_info.value = f"""
                <div style='padding: 10px; background: {UDS_COLORS['success_bg']}; 
                            border-radius: 6px; border-left: 3px solid {UDS_COLORS['success_border']}'>
                    ✅ Supercell is valid!
                </div>
                """
                
                uds_phase2_button.disabled = False
                uds_status.value = f"<div style='padding: 10px; background: {UDS_COLORS['success_bg']}; border-radius: 8px;'><b>Status:</b> Ready for Phase 2</div>"
                
                log_success("Supercell created successfully!")
            else:
                log_error("Supercell still too small! Increase multipliers.")
            
        except Exception as e:
            log_error(f"Supercell creation failed: {e}")

uds_supercell_create.on_click(uds_on_supercell_create_clicked)

def uds_on_phase2_clicked(b):
    """Phase 2: Bulk Relaxation"""
    global uds_global_state
    
    uds_log_output.clear_output(wait=False)
    uds_results_output.clear_output(wait=False)
    
    with uds_log_output:
        log_section("PHASE 2: BULK RELAXATION (0K)")
        log_plain("")
        
        if uds_global_state['structure'] is None:
            log_error("No structure loaded!")
            return
        
        struct = uds_global_state['structure']
        
        is_valid, msg, _ = validate_supercell_size(struct, min_size=15.0)
        
        if is_valid:
            log_success(msg)
        else:
            log_error(msg)
        
        if not is_valid:
            log_error("Cannot proceed - create larger supercell first!")
            return
        
        log_info(f"Structure: {struct.composition.reduced_formula} ({len(struct)} atoms)")
        
        try:
            bulk_result = relax_structure_m3gnet(
                struct,
                fmax=uds_phase2_fmax.value,
                steps=uds_phase2_steps.value,
                log_widget=uds_log_output
            )
            
            E_bulk = bulk_result['energy']
            E_bulk_per_atom = bulk_result['energy_per_atom']
            bulk_opt = bulk_result['structure']
            
            uds_global_state['bulk_energy'] = E_bulk
            uds_global_state['bulk_structure'] = bulk_opt
            
            log_success(f"Bulk Energy: {E_bulk:.6f} eV ({E_bulk_per_atom:.6f} eV/atom)")
            log_success("Phase 2 complete!")
            
            # View structure
            with uds_results_output:
                visualize_structure_3d(bulk_opt, title="Relaxed Bulk Structure", output_widget=uds_results_output)
            
            uds_phase3_button.disabled = False
            uds_status.value = f"<div style='padding: 10px; background: {UDS_COLORS['success_bg']}; border-radius: 8px;'><b>Status:</b> Phase 2 complete - Ready for Phase 3</div>"
            
        except Exception as e:
            log_error(f"Bulk relaxation failed: {e}")
            import traceback
            traceback.print_exc()

uds_phase2_button.on_click(uds_on_phase2_clicked)
def uds_on_phase3_clicked(b):
    """Phase 3: Defect Relaxation and Formation Energies"""
    global uds_global_state
    
    uds_log_output.clear_output(wait=False)
    uds_results_output.clear_output(wait=False)
    
    with uds_log_output:
        log_section("PHASE 3: DEFECT RELAXATION (0K)")
        log_plain("")
        
        if uds_global_state['bulk_structure'] is None:
            log_error("Run Phase 2 first!")
            return
        
        bulk_opt = uds_global_state['bulk_structure']
        E_bulk = uds_global_state['bulk_energy']
        
        # Get defects
        defect_text = uds_defect_list.value.strip()
        if not defect_text:
            log_error("No defects specified!")
            return
        
        defects = [line.strip() for line in defect_text.split('\n') 
                   if line.strip() and not line.strip().startswith('#')]
        
        log_info(f"Defects to process: {len(defects)}")
        
        # Step 1: Relax defects
        log_section("STEP 1: DEFECT STRUCTURE RELAXATION")
        defect_energies = {}
        defect_structures = {}
        
        for i, defect_str in enumerate(defects, 1):
            log_info(f"[{i}/{len(defects)}] Processing: {defect_str}")
            try:
                defect_struct = create_defect_structure(bulk_opt, defect_str)
                defect_struct.perturb(0.05)
                defect_result = relax_structure_m3gnet(
                    defect_struct,
                    fmax=uds_phase3_fmax.value,
                    steps=uds_phase3_steps.value,
                    log_widget=uds_log_output
                )
                E_defect = defect_result['energy']
                defect_energies[defect_str] = E_defect
                defect_structures[defect_str] = defect_result['structure']
                log_success(f"{defect_str}: E = {E_defect:.6f} eV")
            except Exception as e:
                log_error(f"Failed for {defect_str}: {e}")
        
        uds_global_state['defect_energies'] = defect_energies
        uds_global_state['defect_structures'] = defect_structures
        
        # Step 2: Chemical Potentials
        log_section("STEP 2: CHEMICAL POTENTIALS")
        bulk_elements = set(str(el) for el in bulk_opt.composition.elements)
        impurity_elements = extract_impurity_elements_from_defects(defects, bulk_elements)
        api_key = uds_mp_api_key.value.strip() or None
        
        try:
            chempot_dict, chempot_table = calculate_chemical_potentials_simple(
                bulk_opt, E_bulk, impurity_elements=impurity_elements if impurity_elements else None,
                api_key=api_key, log_widget=uds_log_output
            )
            if chempot_dict is None: raise ValueError("Chem pot failed")
            uds_global_state['all_chempot_limits'] = chempot_dict
            uds_global_state['chempot_table'] = chempot_table
            first_limit = list(chempot_dict.keys())[0]
            chempots = chempot_dict[first_limit]
        except Exception as e:
            log_warning("Using placeholder chemical potentials...")
            all_elements = list(bulk_elements) + (impurity_elements if impurity_elements else [])
            E_bulk_per_atom = E_bulk / len(bulk_opt)
            chempots = {el: E_bulk_per_atom / len(all_elements) for el in all_elements}
            chempot_table = None
        
        uds_global_state['chemical_potentials'] = chempots
        
        # Step 3: Formation Energies
        log_section("STEP 3: DEFECT FORMATION ENERGIES")
        formation_energies = {}
        for defect_str, E_defect in defect_energies.items():
            H_f = calculate_defect_formation_energy_silent(E_defect, E_bulk, chempots, defect_str)
            formation_energies[defect_str] = H_f
        
        # Display results
        with uds_results_output:
            clear_output(wait=False)
            log_section("PHASE 3 RESULTS")
            if chempot_table is not None:
                log_section("CHEMICAL POTENTIAL LIMITS")
                display(chempot_table)
            
            # --- PLOTTING SECTION ---
            fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
            ax.set_facecolor('white')
            
            defects_sorted = sorted(formation_energies.keys(), key=lambda x: formation_energies[x])
            H_f_values = [formation_energies[d] for d in defects_sorted]
            x_pos = np.arange(len(defects_sorted))
            
            # Scatter plot
            ax.scatter(x_pos, H_f_values, s=250, color='#3b82f6', edgecolor='#1d4ed8', linewidth=2, zorder=3)
            
            # X-Axis formatting - Use original defect strings with underscores
            tick_labels = defects_sorted  # Keep original notation like V_Cd, As_i, etc.
            ax.set_xticks(x_pos)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=26)
            
            # Labels
            ax.set_ylabel('Defect Formation Energy (eV)', fontsize=26, fontweight='bold', labelpad=15)
            ax.set_xlabel('Defect Type', fontsize=26, fontweight='bold', labelpad=15)
            ax.set_title(f'Defect Formation Energies', fontsize=28, fontweight='bold', pad=20)
            
            # Styling
            ax.tick_params(axis='both', labelsize=26, width=2, length=8)
            
            # Grid and Spines
            ax.grid(True, axis='y', linestyle='--', linewidth=1, alpha=0.6)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            
            plt.tight_layout()
            plt.show()
        
        uds_phase4_button.disabled = False
        uds_status.value = f"<div style='padding: 10px; background: {UDS_COLORS['success_bg']}; border-radius: 8px;'><b>Status:</b> Phase 3 complete</div>"
        
        # Enable download - NO AUTO-CLICK
        with uds_download_output:
            clear_output(wait=False)
            try:
                zip_data, zip_filename = create_download_package(uds_global_state)
                b64 = base64.b64encode(zip_data).decode()
                
                display(HTML(f"""
                    <div style='margin: 20px 0;'>
                        <div style='color: #065f46; margin-bottom: 10px; font-size: 16px;'>✓ All results packaged successfully!</div>
                        <a href="data:application/zip;base64,{b64}"
                           download="{zip_filename}"
                           style="display:inline-block; background:#10b981; color:white; padding:12px 24px;
                                  border-radius:8px; text-decoration:none; font-weight:bold; font-size:16px;">
                            📥 Download {zip_filename}
                        </a>
                    </div>
                """))
                log_success("Download ready!", uds_log_output)
            except Exception as e:
                log_error(f"Download preparation failed: {e}", uds_log_output)

uds_phase3_button.on_click(uds_on_phase3_clicked)
def uds_on_phase4_clicked(b):
    """Phase 4: MD Simulations"""
    global uds_global_state
    
    uds_log_output.clear_output(wait=False)
    uds_results_output.clear_output(wait=False)
    
    with uds_log_output:
        log_section("PHASE 4: MOLECULAR DYNAMICS SIMULATIONS")
        log_plain("")
        
        target = uds_md_target.value
        T_init = uds_md_T_init.value
        T_final = uds_md_T_final.value
        
        if target in ['Bulk Structure', 'Both']:
            if uds_global_state['bulk_structure'] is None:
                log_error("No bulk structure available!")
                return
            
            log_section("MD ON BULK STRUCTURE")
            log_plain("")
            
            try:
                md_result = run_md_simulation(
                    uds_global_state['bulk_structure'],
                    temperature=T_final,
                    steps=uds_md_steps.value,
                    timestep=uds_md_timestep.value,
                    ensemble=uds_md_ensemble.value,
                    friction=uds_md_friction.value,
                    T_init=T_init if T_init != T_final else None,
                    log_widget=uds_log_output,
                    plot_output=uds_results_output
                )
                
                uds_global_state['md_results']['bulk'] = md_result
                
                log_success("Bulk MD complete!")
                
            except Exception as e:
                log_error(f"Bulk MD failed: {e}")
        
        if target in ['Defect Structures', 'Both']:
            if not uds_global_state['defect_structures']:
                log_error("No defect structures available! Run Phase 3 first.")
                return
            
            log_section("MD ON DEFECT STRUCTURES")
            log_plain("")
            
            for defect_str, defect_struct in uds_global_state['defect_structures'].items():
                log_info(f"MD for defect: {defect_str}")
                
                try:
                    md_result = run_md_simulation(
                        defect_struct,
                        temperature=T_final,
                        steps=uds_md_steps.value,
                        timestep=uds_md_timestep.value,
                        ensemble=uds_md_ensemble.value,
                        friction=uds_md_friction.value,
                        T_init=T_init if T_init != T_final else None,
                        log_widget=uds_log_output,
                        plot_output=uds_results_output
                    )
                    
                    uds_global_state['md_results'][defect_str] = md_result
                    
                    log_success(f"MD for {defect_str} complete!")
                    
                except Exception as e:
                    log_error(f"MD for {defect_str} failed: {e}")
        
        log_success("PHASE 4 COMPLETE - ALL MD SIMULATIONS DONE!")
        
        uds_status.value = f"<div style='padding: 10px; background: {UDS_COLORS['success_bg']}; border-radius: 8px;'><b>Status:</b> All phases complete!</div>"
        
        # Update download with MD results
        with uds_download_output:
            clear_output(wait=False)
            try:
                zip_data, zip_filename = create_download_package(uds_global_state)
                b64 = base64.b64encode(zip_data).decode()
                
                display(HTML(f"""
                    <div style='margin: 20px 0;'>
                        <div style='color: #065f46; margin-bottom: 10px; font-size: 16px;'>✓ All results (including MD) packaged successfully!</div>
                        <a href="data:application/zip;base64,{b64}"
                           download="{zip_filename}"
                           style="display:inline-block; background:#10b981; color:white; padding:12px 24px;
                                  border-radius:8px; text-decoration:none; font-weight:bold; font-size:16px;">
                            📥 Download {zip_filename}
                        </a>
                    </div>
                """))
                log_success("Download updated with MD results!", uds_log_output)
            except Exception as e:
                log_error(f"Download preparation failed: {e}", uds_log_output)

uds_phase4_button.on_click(uds_on_phase4_clicked)

# =====================================================================================
# LAYOUT
# =====================================================================================

uds_left_panel = widgets.VBox([
    widgets.HTML(f"<h3 style='color:{UDS_COLORS['text_primary']};'>📥 Phase 1: Structure Setup</h3>"),
    uds_input_method,
    uds_input_panel,
    uds_structure_info,
    
    widgets.HTML("<hr style='margin: 20px 0; border-color: " + UDS_COLORS['panel_border'] + ";'>"),
    widgets.HTML(f"<div style='font-size:12px; color:{UDS_COLORS['text_secondary']}; margin-bottom:10px;'>"
                 "<i>Minimum: 20Å × 20Å × 20Å</i></div>"),
    uds_supercell_panel,
    
    widgets.HTML("<hr style='margin: 20px 0; border-color: " + UDS_COLORS['panel_border'] + ";'>"),
    widgets.HTML(f"<h3 style='color:{UDS_COLORS['text_primary']};'>❄️ Phase 2: Bulk Relaxation (0K)</h3>"),
    uds_phase2_fmax,
    uds_phase2_steps,
    uds_phase2_button,
    
    widgets.HTML("<hr style='margin: 20px 0; border-color: " + UDS_COLORS['panel_border'] + ";'>"),
    widgets.HTML(f"<h3 style='color:{UDS_COLORS['text_primary']};'>⚛️ Phase 3: Defect Analysis (0K)</h3>"),
    uds_defect_list,
    uds_phase3_fmax,
    uds_phase3_steps,
    uds_phase3_button,
    
    widgets.HTML("<hr style='margin: 20px 0; border-color: " + UDS_COLORS['panel_border'] + ";'>"),
    widgets.HTML(f"<h3 style='color:{UDS_COLORS['text_primary']};'>🔥 Phase 4: MD Simulations</h3>"),
    uds_md_target,
    widgets.HBox([uds_md_T_init, uds_md_T_final]),
    uds_md_steps,
    uds_md_timestep,
    uds_md_ensemble,
    uds_md_friction,
    uds_phase4_button,
    
    widgets.HTML("<hr style='margin: 20px 0; border-color: " + UDS_COLORS['panel_border'] + ";'>"),
    uds_progress,
    uds_status,
    widgets.HTML("<hr style='margin: 20px 0; border-color: " + UDS_COLORS['panel_border'] + ";'>"),
    widgets.HTML(f"<h3 style='color:{UDS_COLORS['text_primary']};'>📥 Downloads</h3>"),
    uds_download_output
], layout=widgets.Layout(width='48%', padding='15px'))

uds_right_panel = widgets.VBox([
    widgets.HTML(f"<h3 style='color:{UDS_COLORS['text_primary']};'>📟 Execution Log</h3>"),
    uds_log_output,
    widgets.HTML(f"<h3 style='color:{UDS_COLORS['text_primary']}; margin-top: 20px;'>📊 Results & Visualizations</h3>"),
    uds_results_output
], layout=widgets.Layout(width='50%', padding='15px'))

uds_main_layout = widgets.VBox([
    uds_header,
    uds_guidelines,
    widgets.HBox([uds_left_panel, uds_right_panel])
])

# Display
display(HTML("""
<style>
    /* CLASSIC JUPYTER NOTEBOOK FIXES */
    .container { width:100% !important; }
    .output_subarea {
        max-width: 100% !important;
    }
    .output_scroll {
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        box-shadow: none !important;
    }

    /* JUPYTERLAB FIXES */
    .jp-OutputArea-output {
        max-height: none !important;
        overflow-y: visible !important;
    }
    .jp-OutputArea-child {
        max-height: none !important;
    }
    .jp-Cell-outputArea {
        max-height: none !important;
        overflow: visible !important;
    }
    
    /* WIDGET SPECIFIC */
    .widget-box { 
        max-height: none !important; 
    }
</style>
"""))

display(uds_main_layout)
