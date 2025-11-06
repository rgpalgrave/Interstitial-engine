# =====================================================
# streamlit_app.py (TURBO ULTRA)
# Interstitial-site finder — BLAZING FAST 1D scanning
# Parallel execution + vectorized caching + minimal recomputation
# =====================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    },
    "cubic_F": {
        "4a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "4b (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "8c (1/4,1/4,1/4)":    {"type": "fixed", "xyz": (0.25,0.25,0.25)},
        "24d (0,1/4,1/4)":     {"type": "fixed", "xyz": (0.0, 0.25,0.25)},
        "24e (x,0,0)":         {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "32f (x,x,x)":         {"type": "free",  "xyz": ("x","x","x")},
    },
    "cubic_I": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "6c (0,1/2,0)":        {"type": "fixed", "xyz": (0.0, 0.5, 0.0)},
        "6d (1/2,0,0)":        {"type": "fixed", "xyz": (0.5, 0.0, 0.0)},
        "12e (x,0,0)":         {"type": "free",  "xyz": ("x", 0.0, 0.0)},
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
    },
    "tetragonal_I": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (0,0,1/2)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.5)},
        "4d (0,1/2,1/4)":      {"type": "fixed", "xyz": (0.0, 0.5, 0.25)},
        "4e (0,0,z)":          {"type": "free",  "xyz": (0.0, 0.0,"z")},
        "8g (0,1/2,z)":        {"type": "free",  "xyz": (0.0, 0.5,"z")},
    },
    "orthorhombic_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "1b (0,0,1/2)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.5)},
        "1c (0,1/2,0)":        {"type": "fixed", "xyz": (0.0, 0.5, 0.0)},
        "1d (1/2,0,0)":        {"type": "fixed", "xyz": (0.5, 0.0, 0.0)},
        "2e (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "2f (0,y,0)":          {"type": "free",  "xyz": (0.0,"y", 0.0)},
        "2g (0,0,z)":          {"type": "free",  "xyz": (0.0, 0.0,"z")},
    },
    "orthorhombic_C": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (0,1/2,0)":        {"type": "fixed", "xyz": (0.0, 0.5, 0.0)},
        "4g (0,y,0)":          {"type": "free",  "xyz": (0.0,"y", 0.0)},
    },
    "orthorhombic_I": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "4e (0,0,z)":          {"type": "free",  "xyz": (0.0, 0.0,"z")},
    },
    "orthorhombic_F": {
        "4a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "4b (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "8f (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
    },
    "hexagonal_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "1b (0,0,1/2)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.5)},
        "2c (1/3,2/3,0)":      {"type": "fixed", "xyz": (1/3, 2/3, 0.0)},
        "2d (1/3,2/3,1/2)":    {"type": "fixed", "xyz": (1/3, 2/3, 0.5)},
        "2e (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "2f (x,x,0)":          {"type": "free",  "xyz": ("x","x", 0.0)},
        "2g (x,x,z)":          {"type": "free",  "xyz": ("x","x","z")},
    },
    "hexagonal_HCP": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (0,0,1/4)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.25)},
        "2c (1/3,2/3,1/4)":    {"type": "fixed", "xyz": (1/3, 2/3, 0.25)},
        "2d (1/3,2/3,3/4)":    {"type": "fixed", "xyz": (1/3, 2/3, 0.75)},
        "4f (1/3,2/3,z)":      {"type": "free",  "xyz": (1/3, 2/3, "z")},
    },
    "rhombohedral_R": {
        "3a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "3b (0,0,1/2)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.5)},
        "6c (0,0,z)":          {"type": "free",  "xyz": (0.0, 0.0, "z")},
    },
    "monoclinic_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2e (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
    },
    "monoclinic_C": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "4i (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
    },
    "triclinic_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "1b (x,y,z)":          {"type": "free",  "xyz": ("x","y","z")},
    },
}

st.set_page_config(page_title="Interstitial-site finder — TURBO ULTRA", layout="wide")
st.title("Interstitial-site finder — BLAZING FAST 1D Scan")

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if "run_count" not in st.session_state:
    st.session_state.run_count = 0

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
    a = st.number_input("a (Å)", 2.0, 20.0, 5.0, 0.1, key="global_a")
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

    if preset["type"] == "free":
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
    steps = st.number_input("Steps", 3, 50, 20, 1, key="steps")

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
    st.write("")  # spacing

def clone_params(base: LatticeParams, **kw) -> LatticeParams:
    d = dict(a=base.a, b_ratio=base.b_ratio, c_ratio=base.c_ratio,
             alpha=base.alpha, beta=base.beta, gamma=base.gamma)
    d.update(kw)
    return LatticeParams(**d)

def apply_alpha_scan(subs_in: List[Sublattice], mode: str, val: float) -> List[Sublattice]:
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
            subs_i = apply_alpha_scan(subs, scan_target, float(val))

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
        # Submit all tasks
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

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    for N in Ns:
        valid = ~np.isnan(curves[N])
        if np.any(valid):
            ax.plot(xs[valid], curves[N][valid], "o-", label=f"N={N}", markersize=4)
    ax.set_xlabel(label)
    ax.set_ylabel("s*_N (min radius scale)")
    ax.set_title("Threshold Radii vs Parameter (Parallel Scan)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True, clear_figure=True)

    st.success(f"✓ Scan complete! ({st.session_state.run_count} runs)")

st.caption(
    "Global lattice type/metrics apply to all sublattices. Chemistry mode sets α_i = r_metal + r_anion (Å). "
    "Parallel scan uses thread pool for speed."
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
