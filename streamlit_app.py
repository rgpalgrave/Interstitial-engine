# =====================================================
# streamlit_app.py
# Interstitial-site finder — fast pair-circle engine
# + Chemistry mode + 3D Unit Cell Visualiser (Plotly)
# =====================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict, Tuple

from interstitial_engine import (
    LatticeParams,
    Sublattice,
    max_multiplicity_for_scale,
    find_threshold_s_for_N,
    lattice_vectors,         # reuse engine geometry
    bravais_basis,
    frac_to_cart,
)

# -------------------------------
# Chemistry radii database (VI-coordination, ~Shannon-like; Å)
# -------------------------------
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

def get_metal_radii_options(element: str) -> Dict[int, float]:
    e = norm_el(element)
    return METAL_RADII.get(e, {})

# -------------------------------
# Streamlit setup
# -------------------------------
st.set_page_config(page_title="Interstitial-site finder — fast visualiser logic", layout="wide")
st.title("Interstitial-site finder — fast pair-circle engine (Streamlit)")

# -------------------------------
# Sidebar: lattice + sampling
# -------------------------------
st.sidebar.header("Global lattice")
a = st.sidebar.number_input("a (lattice unit)", 0.1, 10.0, 1.0, 0.01)
b_ratio = st.sidebar.number_input("b/a", 0.1, 5.0, 1.0, 0.01)
c_ratio = st.sidebar.number_input("c/a", 0.1, 5.0, 1.0, 0.01)
alpha = st.sidebar.number_input("α (deg)", 10.0, 170.0, 90.0, 0.1)
beta  = st.sidebar.number_input("β (deg)", 10.0, 170.0, 90.0, 0.1)
gamma = st.sidebar.number_input("γ (deg)", 10.0, 170.0, 90.0, 0.1)
repeat = st.sidebar.slider("Sampling supercell size", 1, 2, 1)

st.sidebar.header("Sampling & tolerances")
k_coarse = st.sidebar.select_slider("Samples per pair-circle (coarse)", options=[4, 6, 8, 12], value=6)
k_fine   = st.sidebar.select_slider("Samples per pair-circle (fine)",   options=[8, 12, 16, 24, 32], value=12)
tol_inside = st.sidebar.number_input("Inside tolerance", 0.0, 0.02, 0.001, 0.0005)
cluster_eps = st.sidebar.number_input("Cluster radius (×a)", 0.01, 0.5, 0.1, 0.01)

# -------------------------------
# Sublattices (up to 6) + Chemistry mode
# -------------------------------
st.subheader("Sublattices (1–6)")

chem_mode = st.checkbox("Chemistry mode (set α from metal + anion radii)", value=True)
anion = st.selectbox("Global anion", list(ANION_RADII.keys()), index=list(ANION_RADII.keys()).index("O"))
r_anion = ANION_RADII[anion]

bravais_choices = [
    "cubic_P","cubic_I","cubic_F","cubic_Diamond","cubic_Pyrochlore",
    "tetragonal_P","tetragonal_I",
    "orthorhombic_P","orthorhombic_C","orthorhombic_I","orthorhombic_F",
    "hexagonal_P","hexagonal_HCP",
    "rhombohedral_R",
    "monoclinic_P","monoclinic_C","triclinic_P",
]

def edit_sublattice(idx: int, use_chem: bool) -> Sublattice:
    with st.expander(f"Sublattice {idx} settings", expanded=(idx <= 3)):
        cols = st.columns(3)
        with cols[0]:
            bravais = st.selectbox(
                f"Bravais {idx}", bravais_choices, index=2 if idx == 1 else 0, key=f"brv{idx}"
            )
            enabled = st.checkbox(f"Enable {idx}", value=True, key=f"vis{idx}")

        with cols[1]:
            offx = st.number_input(f"offset x {idx}", -1.0, 1.0, 0.0, 0.01, key=f"offx{idx}")
            offy = st.number_input(f"offset y {idx}", -1.0, 1.0, 0.0, 0.01, key=f"offy{idx}")
            offz = st.number_input(f"offset z {idx}", -1.0, 1.0, 0.0, 0.01, key=f"offz{idx}")

        with cols[2]:
            name = st.text_input(f"Name {idx}", f"Sub{idx}", key=f"name{idx}")

        if use_chem:
            c1, c2 = st.columns(2)
            with c1:
                elements = sorted(METAL_RADII.keys())
                default_el = "Fe" if idx == 1 else elements[0]
                el = st.selectbox(f"Metal {idx}", elements, index=elements.index(default_el), key=f"el{idx}")
                ox_map = get_metal_radii_options(el)
                ox_states = sorted(ox_map.keys())
                oxid = st.selectbox(f"Oxidation {idx}", ox_states, index=0, key=f"oxid{idx}")
                r_metal_default = ox_map.get(oxid, None)
            with c2:
                r_override = st.number_input(
                    f"Manual metal radius {idx} (Å, optional)", 0.0, 3.0,
                    value=float(r_metal_default if r_metal_default is not None else 0.0),
                    step=0.01, key=f"rman{idx}"
                )
                use_override = st.checkbox(f"Use manual metal radius {idx}", value=False, key=f"useman{idx}")

            r_metal = float(r_override) if use_override else float(r_metal_default if r_metal_default else 0.0)
            alpha_ratio = r_metal + r_anion
            st.caption(f"α_{idx} = r_metal ({r_metal:.3f}) + r_{anion} ({r_anion:.3f}) = {alpha_ratio:.3f} Å")
        else:
            alpha_ratio = st.number_input(f"α ratio {idx}", 0.01, 5.0, 1.0, 0.01, key=f"alpha{idx}")

    return Sublattice(
        name=name,
        bravais=bravais,
        offset_frac=(offx, offy, offz),
        alpha_ratio=alpha_ratio,
        visible=enabled,
    )

n_sub = st.slider("Number of sublattices", 1, 6, 1)
subs: List[Sublattice] = [edit_sublattice(i + 1, chem_mode) for i in range(n_sub)]

p = LatticeParams(a=a, b_ratio=b_ratio, c_ratio=c_ratio, alpha=alpha, beta=beta, gamma=gamma)

# -------------------------------
# Evaluate or search
# -------------------------------
st.divider()
st.subheader("Evaluate multiplicity or find s*_N")

colA, colB = st.columns(2)

with colA:
    s_test = st.number_input("Evaluate at s", 0.0, 5.0, 0.35, 0.001)
    if st.button("Compute k_max(s)"):
        m, reps, repc = max_multiplicity_for_scale(
            subs, p, repeat, s_test,
            k_samples=k_fine,
            tol_inside=tol_inside,
            cluster_eps=cluster_eps * a,
            early_stop_at=None
        )
        st.success(f"k_max(s={s_test:.4f}) = {m}")
        st.caption(f"Hotspots (clustered): {len(reps)}")

with colB:
    N_target = st.slider("Target N", 2, 12, 4)
    s_min = st.number_input("s_min", 0.0, 5.0, 0.01, 0.001)
    s_max = st.number_input("s_max", 0.0, 5.0, 0.9, 0.001)
    if st.button("Find s*_N (bisection)"):
        s_star, milestones = find_threshold_s_for_N(
            N_target, subs, p, repeat,
            s_min=s_min, s_max=s_max,
            k_samples_coarse=k_coarse, k_samples_fine=k_fine,
            tol_inside=tol_inside,
            cluster_eps=cluster_eps * a
        )
        if s_star is None:
            st.error(f"Did not reach N={N_target} in [{s_min}, {s_max}].")
        else:
            st.success(f"s*_{N_target} ≈ {s_star:.6f}")
            st.caption(f"Milestones seen (first s for each m): {milestones}")

# -------------------------------
# Parameter scan
# -------------------------------
st.divider()
st.subheader("Parameter scan: min radii s*_N vs structural parameter")

param = st.selectbox("Parameter to scan", ["a", "b_ratio", "c_ratio", "alpha", "beta", "gamma"], index=2)

if param in ("b_ratio", "c_ratio"):
    vmin_default, vmax_default = 0.5, 1.5
elif param in ("alpha", "beta", "gamma"):
    vmin_default, vmax_default = 60.0, 140.0
else:
    vmin_default, vmax_default = 0.5, 1.5

vmin = st.number_input("Param min", value=float(vmin_default))
vmax = st.number_input("Param max", value=float(vmax_default))
steps = st.slider("Steps", 5, 101, 25)
Ns = [2, 3, 4, 5, 6]

srange_min = st.number_input("s_min (radius scale)", 0.0, 5.0, 0.01, 0.001, key="smin_scan")
srange_max = st.number_input("s_max (radius scale)", 0.0, 5.0, 0.9, 0.001, key="smax_scan")

if st.button("Run scan"):
    xs = np.linspace(vmin, vmax, int(steps))
    curves = {N: np.full(xs.shape, np.nan) for N in Ns}
    prog = st.progress(0, text="Scanning…")

    def clone_params(base: LatticeParams, **kw) -> LatticeParams:
        d = dict(a=base.a, b_ratio=base.b_ratio, c_ratio=base.c_ratio,
                 alpha=base.alpha, beta=base.beta, gamma=base.gamma)
        d.update(kw)
        return LatticeParams(**d)

    for i, val in enumerate(xs):
        p_i = clone_params(p, **{param: float(val)})
        for N in Ns:
            s_star, _ = find_threshold_s_for_N(
                N, subs, p_i, repeat,
                s_min=srange_min, s_max=srange_max,
                k_samples_coarse=k_coarse, k_samples_fine=k_fine,
                tol_inside=tol_inside,
                cluster_eps=cluster_eps * p_i.a,
                max_iter=24
            )
            curves[N][i] = s_star if s_star is not None else np.nan
        prog.progress(int((i + 1) / len(xs) * 100), text=f"Scanning… {i+1}/{len(xs)}")

    fig, ax = plt.subplots()
    for N in Ns:
        ax.plot(xs, curves[N], label=f"N={N}")
    ax.set_xlabel(param)
    ax.set_ylabel("s*_N (min radius scale)")
    ax.set_title("Threshold radii vs structural parameter")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

st.caption(
    "Chemistry mode: α_i = r_metal + r_anion (Å). Override any metal radius if needed. "
    "Radii are used as ratios (s rescales globally). Engine uses minimal-image + KDTree."
)

# =====================================================
# 3D UNIT CELL VIEW (Plotly): on-demand rendering
# =====================================================
st.divider()
st.subheader("3D Unit Cell Visualiser")

colv = st.columns(4)
with colv[0]:
    vis_s = st.number_input("Visualise at s", 0.0, 5.0, max(0.001, s_test if 's_test' in locals() else 0.35), 0.001, key="vis_s")
with colv[1]:
    show_mult = st.number_input("Show intersections with multiplicity N =", 2, 24, value=4, step=1)
with colv[2]:
    marker_size = st.slider("Marker size (Å)", 0.05, 0.5, 0.15, 0.01)
with colv[3]:
    draw_btn = st.button("Render unit cell view")

def _cell_corners_and_edges(a_vec: np.ndarray, b_vec: np.ndarray, c_vec: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    # 8 corners in fractional coords
    fracs = np.array([
        [0,0,0],[1,0,0],[0,1,0],[0,0,1],
        [1,1,0],[1,0,1],[0,1,1],[1,1,1]
    ], float)
    M = np.vstack([a_vec, b_vec, c_vec]).T  # 3x3
    corners = (fracs @ M.T)
    # edges as pairs of corner indices
    idx = {
        (0,1),(0,2),(0,3),
        (1,4),(1,5),
        (2,4),(2,6),
        (3,5),(3,6),
        (4,7),(5,7),(6,7)
    }
    edges = list(idx)
    return corners, edges

def _points_for_sublattice(sub: Sublattice, p: LatticeParams) -> np.ndarray:
    a_vec, b_vec, c_vec = lattice_vectors(p)
    basis = bravais_basis(sub.bravais)
    off = np.array(sub.offset_frac, float)
    pts = []
    for b in basis:
        frac = np.asarray(b) + off
        # keep only those within [0,1) in fractional, with tiny tolerance
        f_mod = np.mod(frac, 1.0)
        cart = frac_to_cart(f_mod, a_vec, b_vec, c_vec)
        pts.append(cart)
    return np.asarray(pts)

def _cart_to_frac(cart: np.ndarray, a_vec: np.ndarray, b_vec: np.ndarray, c_vec: np.ndarray) -> np.ndarray:
    M = np.vstack([a_vec, b_vec, c_vec]).T  # 3x3
    return np.linalg.solve(M, cart.T).T

if draw_btn:
    a_vec, b_vec, c_vec = lattice_vectors(p)
    # Lattice points per enabled sublattice
    sub_pts = []
    for i, sub in enumerate(subs):
        if not sub.visible:
            continue
        pts = _points_for_sublattice(sub, p)
        sub_pts.append((sub.name, pts))
    # Intersections at chosen multiplicity (compute all >=2, cluster already done in engine)
    m_all, reps, repc = max_multiplicity_for_scale(
        subs, p, repeat, vis_s,
        k_samples=k_fine,
        tol_inside=tol_inside,
        cluster_eps=cluster_eps * a,
        early_stop_at=None
    )
    # Filter reps to those inside the central cell and with exact multiplicity = show_mult
    # Use fractional coords test (0..1 with small tol)
    tol_frac = 1e-6
    rep_list = []
    for pt, cnt in zip(reps, repc):
        if int(cnt) != int(show_mult):
            continue
        f = _cart_to_frac(np.asarray(pt), a_vec, b_vec, c_vec)
        if np.all(f >= -tol_frac) and np.all(f <= 1.0 + tol_frac):
            rep_list.append(pt)
    rep_arr = np.asarray(rep_list) if rep_list else np.empty((0,3))

    # ---- Build Plotly figure ----
    fig = go.Figure()

    # Unit cell edges
    corners, edges = _cell_corners_and_edges(a_vec, b_vec, c_vec)
    for i,j in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[i,0], corners[j,0]],
            y=[corners[i,1], corners[j,1]],
            z=[corners[i,2], corners[j,2]],
            mode="lines",
            line=dict(width=4),
            showlegend=False
        ))

    # Sublattice points
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

    # Intersections (order N)
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
