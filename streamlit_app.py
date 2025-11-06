# =====================================================
# streamlit_app.py (TURBO ULTRA v2)
# Interstitial-site finder — BLAZING FAST 1D scanning
# Enhanced: a≥1.0, free placement, linked ratios, CSV export
# =====================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import csv

from interstitial_engine import (
    LatticeParams,
    Sublattice,
    max_multiplicity_for_scale,
    find_threshold_s_for_N,
    lattice_vectors,
    bravais_basis,
    frac_to_cart,
)

# Chemistry radii database (Å)
ANION_RADII: Dict[str, float] = {
    "O": 1.38, "S": 1.84, "Se": 1.98, "F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20,
}
METAL_RADII: Dict[str, Dict[int, float]] = {
    "Li": {1: 0.76}, "Na": {1: 1.02}, "K": {1: 1.38}, "Rb": {1: 1.52}, "Cs": {1: 1.67},
    "Be": {2: 0.59}, "Mg": {2: 0.72}, "Ca": {2: 1.00}, "Sr": {2: 1.18}, "Ba": {2: 1.35},
    "Al": {3: 0.535}, "Ga": {3: 0.62}, "In": {3: 0.80}, "Tl": {1: 1.59, 3: 0.885},
    "Si": {4: 0.40}, "Ge": {4: 0.53}, "Sn": {2: 1.18, 4: 0.69}, "Pb": {2: 1.19, 4: 0.775},
    "Ti": {3: 0.67, 4: 0.605}, "Zr": {4: 0.72}, "Hf": {4: 0.71},
    "V": {2: 0.79, 3: 0.64, 4: 0.58, 5: 0.54},
    "Nb": {5: 0.64}, "Ta": {5: 0.64},
    "Mo": {4: 0.65, 5: 0.61, 6: 0.59}, "W": {4: 0.66, 5: 0.62, 6: 0.60},
    "Sc": {3: 0.745}, "Y": {3: 0.90},
    "Cr": {2: 0.80, 3: 0.615, 6: 0.52},
    "Mn": {2: 0.83, 3: 0.645, 4: 0.67},
    "Fe": {2: 0.78, 3: 0.645},
    "Co": {2: 0.745, 3: 0.61},
    "Ni": {2: 0.69, 3: 0.56},
    "Cu": {1: 0.91, 2: 0.73}, "Zn": {2: 0.74},
    "Cd": {2: 0.95}, "Hg": {2: 1.16},
    "Ru": {3: 0.68, 4: 0.62}, "Rh": {3: 0.665, 4: 0.60}, "Pd": {2: 0.86, 4: 0.615},
    "Ag": {1: 1.15}, "Au": {1: 1.37, 3: 0.85},
    "Pt": {2: 0.80, 4: 0.625},
    "La": {3: 1.032}, "Ce": {3: 1.01, 4: 0.87}, "Pr": {3: 0.99}, "Nd": {3: 0.983},
    "Sm": {3: 0.958}, "Eu": {2: 1.25, 3: 0.947}, "Gd": {3: 0.938}, "Tb": {3: 0.923},
    "Dy": {3: 0.912}, "Ho": {3: 0.901}, "Er": {3: 0.89}, "Tm": {3: 0.88},
    "Yb": {2: 1.16, 3: 0.868}, "Lu": {3: 0.861},
    "B": {3: 0.27}, "P": {5: 0.52}, "As": {5: 0.60}, "Sb": {5: 0.74}, "Bi": {3: 1.03, 5: 0.76},
}

def norm_el(sym: str) -> str:
    sym = sym.strip()
    return sym[0].upper() + sym[1:].lower() if sym else sym

# Wyckoff presets
Wyck = {
    "cubic_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "1b (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "3c (0,1/2,1/2)":      {"type": "fixed", "xyz": (0.0, 0.5, 0.5)},
        "3d (1/2,0,0)":        {"type": "fixed", "xyz": (0.5, 0.0, 0.0)},
        "6e (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "cubic_F": {
        "4a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "4b (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "8c (1/4,1/4,1/4)":    {"type": "fixed", "xyz": (0.25,0.25,0.25)},
        "24d (0,1/4,1/4)":     {"type": "fixed", "xyz": (0.0, 0.25,0.25)},
        "24e (x,0,0)":         {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "32f (x,x,x)":         {"type": "free",  "xyz": ("x","x","x")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "cubic_I": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "6c (0,1/2,0)":        {"type": "fixed", "xyz": (0.0, 0.5, 0.0)},
        "6d (1/2,0,0)":        {"type": "fixed", "xyz": (0.5, 0.0, 0.0)},
        "12e (x,0,0)":         {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "tetragonal_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "1b (0,0,1/2)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.5)},
        "1c (1/2,1/2,0)":      {"type": "fixed", "xyz": (0.5, 0.5, 0.0)},
        "1d (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "2e (0,1/2,0)":        {"type": "fixed", "xyz": (0.0, 0.5, 0.0)},
        "2f (0,1/2,1/2)":      {"type": "fixed", "xyz": (0.0, 0.5, 0.5)},
        "2g (1/2,0,0)":        {"type": "fixed", "xyz": (0.5, 0.0, 0.0)},
        "2h (1/2,0,1/2)":      {"type": "fixed", "xyz": (0.5, 0.0, 0.5)},
        "2i (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "2j (x,x,0)":          {"type": "free",  "xyz": ("x","x",0.0)},
        "2k (x,x,z)":          {"type": "free",  "xyz": ("x","x","z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "tetragonal_I": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (0,0,1/2)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.5)},
        "4d (0,1/2,1/4)":      {"type": "fixed", "xyz": (0.0, 0.5, 0.25)},
        "4e (0,0,z)":          {"type": "free",  "xyz": (0.0, 0.0,"z")},
        "8g (0,1/2,z)":        {"type": "free",  "xyz": (0.0, 0.5,"z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "orthorhombic_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "1b (0,0,1/2)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.5)},
        "1c (0,1/2,0)":        {"type": "fixed", "xyz": (0.0, 0.5, 0.0)},
        "1d (1/2,0,0)":        {"type": "fixed", "xyz": (0.5, 0.0, 0.0)},
        "2e (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "2f (0,y,0)":          {"type": "free",  "xyz": (0.0,"y", 0.0)},
        "2g (0,0,z)":          {"type": "free",  "xyz": (0.0, 0.0,"z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "orthorhombic_C": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (0,1/2,0)":        {"type": "fixed", "xyz": (0.0, 0.5, 0.0)},
        "4g (0,y,0)":          {"type": "free",  "xyz": (0.0,"y", 0.0)},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "orthorhombic_I": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "4e (0,0,z)":          {"type": "free",  "xyz": (0.0, 0.0,"z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "orthorhombic_F": {
        "4a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "4b (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "8f (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "hexagonal_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "1b (0,0,1/2)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.5)},
        "2c (1/3,2/3,0)":      {"type": "fixed", "xyz": (1/3, 2/3, 0.0)},
        "2d (1/3,2/3,1/2)":    {"type": "fixed", "xyz": (1/3, 2/3, 0.5)},
        "2e (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "2f (x,x,0)":          {"type": "free",  "xyz": ("x","x", 0.0)},
        "2g (x,x,z)":          {"type": "free",  "xyz": ("x","x","z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "hexagonal_HCP": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (0,0,1/4)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.25)},
        "2c (1/3,2/3,1/4)":    {"type": "fixed", "xyz": (1/3, 2/3, 0.25)},
        "2d (1/3,2/3,3/4)":    {"type": "fixed", "xyz": (1/3, 2/3, 0.75)},
        "4f (1/3,2/3,z)":      {"type": "free",  "xyz": (1/3, 2/3, "z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "rhombohedral_R": {
        "3a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "3b (0,0,1/2)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.5)},
        "6c (0,0,z)":          {"type": "free",  "xyz": (0.0, 0.0, "z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "monoclinic_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2e (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "monoclinic_C": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "4i (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "triclinic_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "1b (x,y,z)":          {"type": "free",  "xyz": ("x","y","z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
}

st.set_page_config(page_title="Interstitial-site finder — TURBO ULTRA v2", layout="wide")
st.title("Interstitial-site finder — BLAZING FAST 1D Scan")

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if "run_count" not in st.session_state:
    st.session_state.run_count = 0
if "last_scan_data" not in st.session_state:
    st.session_state.last_scan_data = None

# =====================================================
# GLOBAL LATTICE PARAMS
# =====================================================
st.header("Global Lattice")
col_global = st.columns(3)
with col_global[0]:
    global_bravais = st.selectbox(
        "Bravais type",
        ["cubic_P", "cubic_I", "cubic_F", "tetragonal_P", "tetragonal_I",
         "orthorhombic_P", "orthorhombic_C", "orthorhombic_I", "orthorhombic_F",
         "hexagonal_P", "hexagonal_HCP", "rhombohedral_R", "monoclinic_P", "monoclinic_C", "triclinic_P"],
        key="global_bravais"
    )
with col_global[1]:
    a = st.number_input("a (Å)", 1.0, 20.0, 5.0, 0.1, key="global_a")
with col_global[2]:
    repeat = st.number_input("Supercell repeat", 1, 5, 1, key="global_repeat")

col_ratios = st.columns(3)
with col_ratios[0]:
    b_ratio = st.number_input("b/a", 0.5, 3.0, 1.0, 0.05, key="global_b_ratio")
with col_ratios[1]:
    c_ratio = st.number_input("c/a", 0.5, 3.0, 1.0, 0.05, key="global_c_ratio")
with col_ratios[2]:
    gamma = st.number_input("γ (°)", 60.0, 120.0, 90.0, 1.0, key="global_gamma")

col_angles = st.columns(2)
with col_angles[0]:
    alpha = st.number_input("α (°)", 60.0, 120.0, 90.0, 1.0, key="global_alpha")
with col_angles[1]:
    beta = st.number_input("β (°)", 60.0, 120.0, 90.0, 1.0, key="global_beta")

p = LatticeParams(a=a, b_ratio=b_ratio, c_ratio=c_ratio, alpha=alpha, beta=beta, gamma=gamma)
struct_params = ["a", "b_ratio", "c_ratio", "alpha", "beta", "gamma"]

# =====================================================
# SUBLATTICES
# =====================================================
st.header("Sublattices")
num_subs = st.number_input("Number of sublattices", 1, 6, 1, key="num_subs")

# Linked ratios UI
st.subheader("Linked Ratios (for synchronized scanning)")
linked_pairs = []
for i in range(int(num_subs) - 1):
    col_link = st.columns(3)
    with col_link[0]:
        link_check = st.checkbox(f"Link Sub{i+1} ↔ Sub{i+2}", value=False, key=f"link_{i}_{i+1}")
    with col_link[1]:
        if link_check:
            link_type = st.selectbox(f"Link type", ["ratio_a", "ratio_b"], key=f"link_type_{i}_{i+1}")
            if link_type == "ratio_a":
                st.caption(f"Sub{i+2} α = Sub{i+1} α × factor")
            else:
                st.caption(f"Sub{i+2} b/a = Sub{i+1} b/a")
        else:
            link_type = None
    with col_link[2]:
        if link_check:
            factor = st.number_input(f"Factor", 0.1, 10.0, 1.0, 0.1, key=f"link_factor_{i}_{i+1}")
        else:
            factor = 1.0
    if link_check:
        linked_pairs.append({"from": i, "to": i+1, "type": link_type, "factor": factor})

subs = []
for i in range(int(num_subs)):
    st.subheader(f"Sublattice {i+1}")
    col_sub = st.columns(5)
    with col_sub[0]:
        sub_bravais = st.selectbox(f"Bravais {i+1}", list(Wyck.keys()), key=f"sub_bravais_{i}")
    with col_sub[1]:
        sub_name = st.text_input(f"Name {i+1}", value=f"Sub{i+1}", key=f"sub_name_{i}")
    with col_sub[2]:
        preset_label = st.selectbox(f"Preset {i+1}", list(Wyck.get(sub_bravais, {}).keys()), key=f"sub_preset_{i}")
    with col_sub[3]:
        chem_mode = st.checkbox("Chemistry mode", value=False, key=f"sub_chem_{i}")
    with col_sub[4]:
        visible = st.checkbox("Visible", value=True, key=f"sub_visible_{i}")

    preset = Wyck[sub_bravais].get(preset_label, {"type": "fixed", "xyz": (0, 0, 0)})
    
    if chem_mode:
        col_chem = st.columns(2)
        with col_chem[0]:
            metal = st.selectbox("Metal", list(METAL_RADII.keys()), key=f"sub_metal_{i}")
            charge = st.selectbox("Charge", list(METAL_RADII[metal].keys()), key=f"sub_charge_{i}")
        with col_chem[1]:
            anion = st.selectbox("Anion", list(ANION_RADII.keys()), key=f"sub_anion_{i}")
        r_metal = METAL_RADII[metal][charge]
        r_anion = ANION_RADII[anion]
        alpha_ratio = r_metal + r_anion
    else:
        alpha_ratio = st.number_input(f"α (Å) {i+1}", 0.01, 5.0, 1.0, 0.1, key=f"sub_alpha_{i}")

    if preset["type"] == "free3d":
        # Free 3D placement
        col_free3d = st.columns(3)
        with col_free3d[0]:
            fx = st.number_input(f"x {i+1}", 0.0, 1.0, 0.0, 0.01, key=f"sub_free3d_x_{i}")
        with col_free3d[1]:
            fy = st.number_input(f"y {i+1}", 0.0, 1.0, 0.0, 0.01, key=f"sub_free3d_y_{i}")
        with col_free3d[2]:
            fz = st.number_input(f"z {i+1}", 0.0, 1.0, 0.0, 0.01, key=f"sub_free3d_z_{i}")
        offset_frac = (fx, fy, fz)
    elif preset["type"] == "free":
        offset = []
        for j, coord in enumerate(preset["xyz"]):
            if isinstance(coord, str):
                val = st.number_input(f"  {coord} {i+1}", 0.0, 1.0, 0.5, 0.05, key=f"sub_free_{i}_{j}")
                offset.append(val)
            else:
                offset.append(float(coord))
        offset_frac = tuple(offset)
    else:
        offset_frac = tuple(float(x) if isinstance(x, (int, float)) else 0.0 for x in preset["xyz"])

    subs.append(Sublattice(sub_name, sub_bravais, offset_frac, alpha_ratio, visible))

# =====================================================
# SCAN CONTROLS
# =====================================================
st.header("1D Parameter Scan")
scan_col = st.columns(4)
with scan_col[0]:
    scan_target = st.selectbox(
        "Scan parameter",
        struct_params + [f"alpha(Sub{i+1})" for i in range(int(num_subs))],
        key="scan_target"
    )
    label = f"Parameter {scan_target}"
    if scan_target in struct_params:
        label = scan_target
    else:
        try:
            idx = int(scan_target.split("Sub")[1].split(")")[0].strip()) - 1
            label = f"α (Sub {idx+1})"
        except:
            label = scan_target

with scan_col[1]:
    vmin = st.number_input("Min", -10.0, 10.0, 0.5, 0.1, key="vmin")
with scan_col[2]:
    vmax = st.number_input("Max", 0.0, 20.0, 2.0, 0.1, key="vmax")
with scan_col[3]:
    steps = st.number_input("Points", 3, 100, 20, 1, key="steps")

scan_col2 = st.columns(4)
with scan_col2[0]:
    Ns_str = st.text_input("Multiplicities (comma-sep)", "2, 3, 4, 5, 6", key="Ns_input")
    Ns = [int(x.strip()) for x in Ns_str.split(",") if x.strip()]
with scan_col2[1]:
    srange_min = st.number_input("s_min", 0.001, 1.0, 0.01, 0.001, key="srange_min")
with scan_col2[2]:
    srange_max = st.number_input("s_max", 0.1, 2.0, 0.9, 0.05, key="srange_max")
with scan_col2[3]:
    tol_inside = st.number_input("Tolerance", 1e-4, 1e-2, 1e-3, 1e-4, key="tol_inside", format="%.1e")

scan_col3 = st.columns(3)
with scan_col3[0]:
    cluster_eps = st.number_input("Cluster ε (×a)", 0.01, 0.5, 0.1, 0.01, key="cluster_eps")
with scan_col3[1]:
    max_workers = st.number_input("Parallel workers", 1, 16, 4, 1, key="max_workers")
with scan_col3[2]:
    st.write("")

def clone_params(base: LatticeParams, **kw) -> LatticeParams:
    d = dict(a=base.a, b_ratio=base.b_ratio, c_ratio=base.c_ratio,
             alpha=base.alpha, beta=base.beta, gamma=base.gamma)
    d.update(kw)
    return LatticeParams(**d)

def apply_alpha_scan(subs_in: List[Sublattice], mode: str, val: float, linked_pairs: List[Dict]) -> List[Sublattice]:
    """Apply alpha scan with optional linked ratio updates"""
    out: List[Sublattice] = []
    if mode == "alpha_scalar (all sublattices)":
        for s in subs_in:
            out.append(Sublattice(s.name, s.bravais, s.offset_frac, alpha_ratio=s.alpha_ratio * val, visible=s.visible))
    elif mode.startswith("alpha(Sub"):
        try:
            idx = int(mode.split("Sub")[1].split(")")[0].strip()) - 1
        except Exception:
            idx = -1
        for i, s in enumerate(subs_in):
            new_alpha = s.alpha_ratio * val if i == idx else s.alpha_ratio
            out.append(Sublattice(s.name, s.bravais, s.offset_frac, alpha_ratio=new_alpha, visible=s.visible))
    else:
        out = [Sublattice(s.name, s.bravais, s.offset_frac, s.alpha_ratio, s.visible) for s in subs_in]
    
    # Apply linked ratio updates
    for link in linked_pairs:
        from_idx = link["from"]
        to_idx = link["to"]
        if from_idx < len(out):
            from_sub = out[from_idx]
            to_sub = out[to_idx]
            if link["type"] == "ratio_a":
                new_alpha = from_sub.alpha_ratio * link["factor"]
                out[to_idx] = Sublattice(to_sub.name, to_sub.bravais, to_sub.offset_frac, 
                                        alpha_ratio=new_alpha, visible=to_sub.visible)
    
    return out

# =====================================================
# OPTIMIZED PARALLEL SCAN
# =====================================================
def compute_curve_point(idx: int, val: float, N: int, is_struct_param: bool) -> Tuple[int, int, Optional[float]]:
    """Compute one point on one curve. Returns (idx, N, s_star)"""
    try:
        if is_struct_param:
            p_i = clone_params(p, **{scan_target: float(val)})
            subs_i = subs
        else:
            p_i = p
            subs_i = apply_alpha_scan(subs, scan_target, float(val), linked_pairs)

        s_star, _ = find_threshold_s_for_N(
            N, subs_i, p_i, repeat,
            s_min=srange_min, s_max=srange_max,
            k_samples_coarse=4, k_samples_fine=8,
            tol_inside=tol_inside,
            cluster_eps=cluster_eps * p_i.a,
            max_iter=20
        )
        return idx, N, s_star
    except Exception as e:
        st.error(f"Error at index {idx}, N={N}: {e}")
        return idx, N, None

if st.button("⚡ Run Parallel Scan", use_container_width=True):
    st.session_state.run_count += 1
    xs = np.linspace(vmin, vmax, int(steps))
    curves = {N: np.full(xs.shape, np.nan) for N in Ns}

    progress_bar = st.progress(0, text="Initializing…")
    status_text = st.empty()

    # Parallel execution
    is_struct = scan_target in struct_params
    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        futures = {}
        for i, val in enumerate(xs):
            for N in Ns:
                task_id = (i, N)
                fut = executor.submit(compute_curve_point, i, val, N, is_struct)
                futures[fut] = task_id

        completed = 0
        total_tasks = len(futures)
        for future in as_completed(futures):
            completed += 1
            try:
                idx, N, s_star = future.result()
                curves[N][idx] = s_star if s_star is not None else np.nan
            except Exception as e:
                st.warning(f"Task failed: {e}")
            
            pct = int((completed / total_tasks) * 100)
            progress_bar.progress(pct, text=f"Scanning… {completed}/{total_tasks} ({pct}%)")

    progress_bar.empty()
    status_text.empty()

    # Store scan data for export
    st.session_state.last_scan_data = {
        "xs": xs,
        "curves": curves,
        "label": label,
        "scan_target": scan_target,
    }

    # Plot results (lines only, no markers)
    fig, ax = plt.subplots(figsize=(10, 6))
    for N in Ns:
        valid = ~np.isnan(curves[N])
        if np.any(valid):
            ax.plot(xs[valid], curves[N][valid], label=f"N={N}", linewidth=2)
    ax.set_xlabel(label)
    ax.set_ylabel("s*_N (min radius scale)")
    ax.set_title("Threshold Radii vs Parameter (Parallel Scan)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True, clear_figure=True)

    st.success(f"✓ Scan complete! ({st.session_state.run_count} runs)")

# =====================================================
# EXPORT SCAN DATA
# =====================================================
if st.session_state.last_scan_data:
    st.divider()
    st.subheader("Export Scan Data")
    
    col_exp = st.columns(3)
    with col_exp[0]:
        export_format = st.selectbox("Format", ["CSV", "TSV", "JSON"], key="export_format")
    with col_exp[1]:
        st.write("")
    with col_exp[2]:
        export_btn = st.button("Download")
    
    if export_btn:
        data = st.session_state.last_scan_data
        xs = data["xs"]
        curves = data["curves"]
        
        if export_format == "CSV" or export_format == "TSV":
            delimiter = "," if export_format == "CSV" else "\t"
            output = io.StringIO()
            writer = csv.writer(output, delimiter=delimiter)
            
            # Header
            header = [data["label"]] + [f"N={N}" for N in sorted(curves.keys())]
            writer.writerow(header)
            
            # Data rows
            for i, x_val in enumerate(xs):
                row = [str(x_val)]
                for N in sorted(curves.keys()):
                    s_val = curves[N][i]
                    row.append(str(s_val) if not np.isnan(s_val) else "")
                writer.writerow(row)
            
            filename = f"scan_{data['scan_target']}.{'csv' if export_format == 'CSV' else 'tsv'}"
            st.download_button(
                label=f"📥 Download {export_format}",
                data=output.getvalue(),
                file_name=filename,
                mime="text/plain",
                key="download_csv"
            )
        
        elif export_format == "JSON":
            import json
            json_data = {
                "scan_parameter": data["scan_target"],
                "parameter_label": data["label"],
                "x_values": xs.tolist(),
                "curves": {str(N): [float(v) if not np.isnan(v) else None for v in curves[N]] 
                          for N in curves}
            }
            filename = f"scan_{data['scan_target']}.json"
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(json_data, indent=2),
                file_name=filename,
                mime="application/json",
                key="download_json"
            )

st.caption(
    "Global lattice type/metrics apply to all sublattices. Free placement allows (x,y,z) entry. "
    "Linked ratios enable synchronized multi-sublattice scanning. Chemistry mode sets α_i = r_metal + r_anion (Å)."
)

# =====================================================
# 2D PARAMETER SCAN
# =====================================================
st.divider()
st.header("2D Parameter Scan (Heatmap)")

col_2d = st.columns(2)
with col_2d[0]:
    do_2d_scan = st.checkbox("Enable 2D scan", value=False, key="enable_2d_scan")

if do_2d_scan:
    st.subheader("2D Scan Configuration")
    
    # Select two independent parameters
    col_2d_params = st.columns(2)
    all_params = struct_params + [f"alpha(Sub{i+1})" for i in range(int(num_subs))]
    
    with col_2d_params[0]:
        param_x = st.selectbox(
            "X-axis parameter",
            all_params,
            key="param_2d_x"
        )
    with col_2d_params[1]:
        param_y = st.selectbox(
            "Y-axis parameter",
            all_params,
            key="param_2d_y",
            index=min(1, len(all_params)-1)  # Default to different param
        )
    
    # Validate params are different
    if param_x == param_y:
        st.error("X and Y parameters must be different!")
        do_2d_scan = False
    
    if do_2d_scan:
        # Get parameter labels
        def get_param_label(p):
            if p in struct_params:
                return p
            else:
                try:
                    idx = int(p.split("Sub")[1].split(")")[0].strip()) - 1
                    return f"α (Sub {idx+1})"
                except:
                    return p
        
        label_x = get_param_label(param_x)
        label_y = get_param_label(param_y)
        
        # Parameter ranges
        col_2d_x = st.columns(3)
        with col_2d_x[0]:
            x_min = st.number_input("X min", -10.0, 10.0, 0.5, 0.1, key="x_min_2d")
        with col_2d_x[1]:
            x_max = st.number_input("X max", 0.0, 20.0, 2.0, 0.1, key="x_max_2d")
        with col_2d_x[2]:
            x_steps = st.number_input("X points", 3, 50, 15, 1, key="x_steps_2d")
        
        col_2d_y = st.columns(3)
        with col_2d_y[0]:
            y_min = st.number_input("Y min", -10.0, 10.0, 0.5, 0.1, key="y_min_2d")
        with col_2d_y[1]:
            y_max = st.number_input("Y max", 0.0, 20.0, 2.0, 0.1, key="y_max_2d")
        with col_2d_y[2]:
            y_steps = st.number_input("Y points", 3, 50, 15, 1, key="y_steps_2d")
        
        # Single multiplicity target
        col_2d_k = st.columns(3)
        with col_2d_k[0]:
            k_target_2d = st.number_input("Target multiplicity N", 2, 24, 4, 1, key="k_target_2d")
        with col_2d_k[1]:
            max_workers_2d = st.number_input("Parallel workers (2D)", 1, 16, 4, 1, key="max_workers_2d")
        with col_2d_k[2]:
            st.write("")
        
        # 2D scan button
        if st.button("⚡ Run 2D Scan", use_container_width=True, key="run_2d"):
            st.session_state.run_count += 1
            
            x_vals = np.linspace(x_min, x_max, int(x_steps))
            y_vals = np.linspace(y_min, y_max, int(y_steps))
            
            # Result matrix
            heatmap = np.full((len(y_vals), len(x_vals)), np.nan)
            
            progress_bar = st.progress(0, text="Initializing 2D scan…")
            
            # 2D parallel execution
            def compute_2d_point(ix: int, iy: int, x_val: float, y_val: float) -> Tuple[int, int, Optional[float]]:
                """Compute s* for one (x,y) point at target multiplicity"""
                try:
                    # Apply x parameter
                    if param_x in struct_params:
                        p_xy = clone_params(p, **{param_x: float(x_val)})
                        subs_xy = subs
                    else:
                        p_xy = p
                        subs_xy = apply_alpha_scan(subs, param_x, float(x_val), linked_pairs)
                    
                    # Apply y parameter
                    if param_y in struct_params:
                        p_xy = clone_params(p_xy, **{param_y: float(y_val)})
                    else:
                        subs_xy = apply_alpha_scan(subs_xy, param_y, float(y_val), linked_pairs)
                    
                    s_star, _ = find_threshold_s_for_N(
                        int(k_target_2d), subs_xy, p_xy, repeat,
                        s_min=srange_min, s_max=srange_max,
                        k_samples_coarse=4, k_samples_fine=8,
                        tol_inside=tol_inside,
                        cluster_eps=cluster_eps * p_xy.a,
                        max_iter=20
                    )
                    return ix, iy, s_star
                except Exception as e:
                    st.warning(f"Error at ({ix},{iy}): {e}")
                    return ix, iy, None
            
            with ThreadPoolExecutor(max_workers=int(max_workers_2d)) as executor:
                futures = {}
                total_tasks = len(x_vals) * len(y_vals)
                
                for iy, y_val in enumerate(y_vals):
                    for ix, x_val in enumerate(x_vals):
                        fut = executor.submit(compute_2d_point, ix, iy, x_val, y_val)
                        futures[fut] = (ix, iy)
                
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    try:
                        ix, iy, s_star = future.result()
                        heatmap[iy, ix] = s_star if s_star is not None else np.nan
                    except Exception as e:
                        st.warning(f"Task failed: {e}")
                    
                    pct = int((completed / total_tasks) * 100)
                    progress_bar.progress(pct, text=f"Scanning… {completed}/{total_tasks} ({pct}%)")
            
            progress_bar.empty()
            
            # Store 2D scan data
            st.session_state.last_2d_scan_data = {
                "heatmap": heatmap,
                "x_vals": x_vals,
                "y_vals": y_vals,
                "label_x": label_x,
                "label_y": label_y,
                "param_x": param_x,
                "param_y": param_y,
                "k_target": k_target_2d,
            }
            
            # Plot heatmap using Plotly
            fig = go.Figure(data=go.Heatmap(
                z=heatmap,
                x=x_vals,
                y=y_vals,
                colorscale="Viridis",
                colorbar=dict(title=f"s* for N={k_target_2d}"),
                hoverongaps=False,
                hovertemplate=f"{label_x}: %{{x:.3f}}<br>{label_y}: %{{y:.3f}}<br>s* = %{{z:.4f}}<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"2D Scan: s* for N={k_target_2d} multiplicity",
                xaxis_title=label_x,
                yaxis_title=label_y,
                width=900,
                height=700,
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"✓ 2D scan complete! ({x_steps} × {y_steps} = {int(x_steps*y_steps)} points)")
        
        # =====================================================
        # 2D EXPORT
        # =====================================================
        if "last_2d_scan_data" in st.session_state and st.session_state.last_2d_scan_data:
            st.divider()
            st.subheader("Export 2D Scan Data")
            
            col_2d_exp = st.columns(3)
            with col_2d_exp[0]:
                exp_2d_format = st.selectbox("Format", ["CSV", "NumPy NPZ", "JSON"], key="export_2d_format")
            with col_2d_exp[1]:
                st.write("")
            with col_2d_exp[2]:
                exp_2d_btn = st.button("Download 2D", key="download_2d")
            
            if exp_2d_btn:
                data_2d = st.session_state.last_2d_scan_data
                hm = data_2d["heatmap"]
                xs_2d = data_2d["x_vals"]
                ys_2d = data_2d["y_vals"]
                
                if exp_2d_format == "CSV":
                    # Flatten heatmap: header row + header col
                    output = io.StringIO()
                    writer = csv.writer(output)
                    
                    # Header row with x values
                    header = [f"{data_2d['label_y']}/{data_2d['label_x']}"] + [f"{x:.4f}" for x in xs_2d]
                    writer.writerow(header)
                    
                    # Data rows
                    for iy, y_val in enumerate(ys_2d):
                        row = [f"{y_val:.4f}"] + [str(hm[iy, ix]) if not np.isnan(hm[iy, ix]) else "" 
                                                   for ix in range(len(xs_2d))]
                        writer.writerow(row)
                    
                    filename = f"scan_2d_{data_2d['param_x']}_{data_2d['param_y']}.csv"
                    st.download_button(
                        label="📥 Download CSV",
                        data=output.getvalue(),
                        file_name=filename,
                        mime="text/plain",
                        key="download_2d_csv"
                    )
                
                elif exp_2d_format == "NumPy NPZ":
                    # Binary NumPy format (efficient)
                    buf = io.BytesIO()
                    np.savez(
                        buf,
                        heatmap=hm,
                        x_values=xs_2d,
                        y_values=ys_2d,
                        x_label=np.string_(data_2d['label_x']),
                        y_label=np.string_(data_2d['label_y']),
                        k_target=data_2d['k_target']
                    )
                    buf.seek(0)
                    
                    filename = f"scan_2d_{data_2d['param_x']}_{data_2d['param_y']}.npz"
                    st.download_button(
                        label="📥 Download NPZ",
                        data=buf.getvalue(),
                        file_name=filename,
                        mime="application/octet-stream",
                        key="download_2d_npz"
                    )
                
                elif exp_2d_format == "JSON":
                    import json
                    json_data = {
                        "x_parameter": data_2d['param_x'],
                        "y_parameter": data_2d['param_y'],
                        "x_label": data_2d['label_x'],
                        "y_label": data_2d['label_y'],
                        "k_target": data_2d['k_target'],
                        "x_values": xs_2d.tolist(),
                        "y_values": ys_2d.tolist(),
                        "heatmap": hm.tolist(),  # 2D array as list of lists
                        "shape": list(hm.shape)
                    }
                    
                    filename = f"scan_2d_{data_2d['param_x']}_{data_2d['param_y']}.json"
                    st.download_button(
                        label="📥 Download JSON",
                        data=json.dumps(json_data, indent=2),
                        file_name=filename,
                        mime="application/json",
                        key="download_2d_json"
                    )

# =====================================================
# 3D UNIT CELL VIEW
# =====================================================
st.divider()
st.subheader("3D Unit Cell Visualiser")

colv = st.columns(4)
with colv[0]:
    vis_s = st.number_input("Visualise at s", 0.0, 5.0, max(0.001, 0.35), 0.001, key="vis_s")
with colv[1]:
    show_mult = st.number_input("Show intersections with multiplicity N =", 2, 24, value=4, step=1)
with colv[2]:
    marker_size = st.slider("Marker size (Å)", 0.05, 0.5, 0.15, 0.01)
with colv[3]:
    draw_btn = st.button("Render unit cell view")

def _cell_corners_and_edges(a_vec: np.ndarray, b_vec: np.ndarray, c_vec: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    fracs = np.array([
        [0,0,0],[1,0,0],[0,1,0],[0,0,1],
        [1,1,0],[1,0,1],[0,1,1],[1,1,1]
    ], float)
    M = np.vstack([a_vec, b_vec, c_vec]).T
    corners = (fracs @ M.T)
    edges = [
        (0,1),(0,2),(0,3),
        (1,4),(1,5),
        (2,4),(2,6),
        (3,5),(3,6),
        (4,7),(5,7),(6,7)
    ]
    return corners, edges

def _points_for_sublattice(sub: Sublattice, p: LatticeParams) -> np.ndarray:
    a_vec, b_vec, c_vec = lattice_vectors(p)
    basis = bravais_basis(sub.bravais)
    off = np.array(sub.offset_frac, float)
    pts = []
    for b in basis:
        frac = np.asarray(b) + off
        f_mod = np.mod(frac, 1.0)
        cart = frac_to_cart(f_mod, a_vec, b_vec, c_vec)
        pts.append(cart)
    return np.asarray(pts)

def _cart_to_frac(cart: np.ndarray, a_vec: np.ndarray, b_vec: np.ndarray, c_vec: np.ndarray) -> np.ndarray:
    M = np.vstack([a_vec, b_vec, c_vec]).T
    return np.linalg.solve(M, cart.T).T

if draw_btn:
    a_vec, b_vec, c_vec = lattice_vectors(p)
    sub_pts = []
    for i, sub in enumerate(subs):
        if not sub.visible:
            continue
        pts = _points_for_sublattice(sub, p)
        sub_pts.append((sub.name, pts))

    m_all, reps, repc = max_multiplicity_for_scale(
        subs, p, repeat, vis_s,
        k_samples=8, tol_inside=tol_inside,
        cluster_eps=cluster_eps * a, early_stop_at=None
    )
    tol_frac = 1e-6
    rep_list = []
    for pt, cnt in zip(reps, repc):
        if int(cnt) != int(show_mult):
            continue
        f = _cart_to_frac(np.asarray(pt), a_vec, b_vec, c_vec)
        if np.all(f >= -tol_frac) and np.all(f <= 1.0 + tol_frac):
            rep_list.append(pt)
    rep_arr = np.asarray(rep_list) if rep_list else np.empty((0,3))

    fig = go.Figure()
    corners, edges = _cell_corners_and_edges(a_vec, b_vec, c_vec)
    for i,j in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[i,0], corners[j,0]],
            y=[corners[i,1], corners[j,1]],
            z=[corners[i,2], corners[j,2]],
            mode="lines", line=dict(width=4), showlegend=False
        ))

    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
    for idx, (name, pts) in enumerate(sub_pts):
        if len(pts)==0: 
            continue
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode="markers",
            marker=dict(size=6, opacity=0.9),
            name=f"{name} sites",
            marker_symbol="circle",
            legendgroup=f"sub{idx}",
            marker_color=palette[idx % len(palette)],
        ))

    if rep_arr.size:
        fig.add_trace(go.Scatter3d(
            x=rep_arr[:,0], y=rep_arr[:,1], z=rep_arr[:,2],
            mode="markers",
            marker=dict(size=max(2, int( marker_size / max(1e-9, a) * 30 )), opacity=0.95),
            name=f"Intersections N={show_mult}",
            marker_symbol="diamond"
        ))
    else:
        st.info("No intersections of the selected multiplicity in the central unit cell at this s (after clustering).")

    fig.update_scenes(aspectmode="data")
    fig.update_layout(
        scene=dict(
            xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)",
            xaxis=dict(showbackground=False), yaxis=dict(showbackground=False), zaxis=dict(showbackground=False),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)

# EOF
