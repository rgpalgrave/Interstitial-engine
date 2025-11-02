# =====================================================
# streamlit_app.py
# Interstitial-site finder — fast pair-circle engine
# (UI with param scan; supports up to 6 sublattices)
# =====================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from interstitial_engine import (
    LatticeParams,
    Sublattice,
    max_multiplicity_for_scale,
    find_threshold_s_for_N,
)

# Setup
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
# Sublattices (up to 6)
# -------------------------------
st.subheader("Sublattices (1–6)")

bravais_choices = [
    # Cubic
    "cubic_P","cubic_I","cubic_F","cubic_Diamond","cubic_Pyrochlore",
    # Tetragonal
    "tetragonal_P","tetragonal_I",
    # Orthorhombic
    "orthorhombic_P","orthorhombic_C","orthorhombic_I","orthorhombic_F",
    # Hexagonal
    "hexagonal_P","hexagonal_HCP",
    # Rhombohedral
    "rhombohedral_R",
    # Lower symmetry
    "monoclinic_P","monoclinic_C","triclinic_P",
]

def edit_sublattice(idx: int) -> Sublattice:
    with st.expander(f"Sublattice {idx} settings", expanded=(idx <= 3)):
        cols = st.columns(3)
        with cols[0]:
            bravais = st.selectbox(
                f"Bravais {idx}", bravais_choices, index=2 if idx == 1 else 0, key=f"brv{idx}"
            )
            alpha_ratio = st.number_input(f"α ratio {idx}", 0.01, 3.0, 1.0, 0.01, key=f"ar{idx}")
            enabled = st.checkbox(f"Enable {idx}", value=True, key=f"vis{idx}")
        with cols[1]:
            offx = st.number_input(f"offset x {idx}", -1.0, 1.0, 0.0, 0.01, key=f"ox{idx}")
            offy = st.number_input(f"offset y {idx}", -1.0, 1.0, 0.0, 0.01, key=f"oy{idx}")
            offz = st.number_input(f"offset z {idx}", -1.0, 1.0, 0.0, 0.01, key=f"oz{idx}")
        with cols[2]:
            name = st.text_input(f"Name {idx}", f"Sub{idx}", key=f"name{idx}")
    return Sublattice(
        name=name,
        bravais=bravais,
        offset_frac=(offx, offy, offz),
        alpha_ratio=alpha_ratio,
        visible=enabled,
    )

n_sub = st.slider("Number of sublattices", 1, 6, 1)   # <-- now up to 6
subs: List[Sublattice] = [edit_sublattice(i + 1) for i in range(n_sub)]

p = LatticeParams(a=a, b_ratio=b_ratio, c_ratio=c_ratio, alpha=alpha, beta=beta, gamma=gamma)

# -------------------------------
# Evaluate or search
# -------------------------------
st.divider()
st.subheader("Evaluate multiplicity or find s*_N")

colA, colB = st.columns(2)

with colA:
    s_test = st.number_input("Evaluate at s", 0.0, 2.0, 0.35, 0.001)
    if st.button("Compute k_max(s)"):
        m, reps, repc = max_multiplicity_for_scale(
            subs, p, repeat, s_test,
            k_samples=k_fine,  # single eval: use fine
            tol_inside=tol_inside,
            cluster_eps=cluster_eps * a,
            early_stop_at=None
        )
        st.success(f"k_max(s={s_test:.4f}) = {m}")
        st.caption(f"Hotspots (clustered): {len(reps)}")

with colB:
    N_target = st.slider("Target N", 2, 12, 4)
    s_min = st.number_input("s_min", 0.0, 2.0, 0.01, 0.001)
    s_max = st.number_input("s_max", 0.0, 2.0, 0.9, 0.001)
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

srange_min = st.number_input("s_min (radius scale)", 0.0, 2.0, 0.01, 0.001, key="smin_scan")
srange_max = st.number_input("s_max (radius scale)", 0.0, 2.0, 0.9, 0.001, key="smax_scan")

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
    "Radii are r_i = α_i · s · a. Engine caches geometry & uses minimal-image + KDTree counting. "
    "Tip: hexagonal → γ≈120°; rhombohedral → α=β=γ≠90°."
)
