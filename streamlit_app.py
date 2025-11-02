# streamlit_app.py
import streamlit as st
import numpy as np

from interstitial_engine import (
    LatticeParams, Sublattice,
    max_multiplicity_for_scale, find_threshold_s_for_N
)

st.set_page_config(page_title="Interstitial-site finder (blazing visualiser logic)",
                   layout="wide")

st.title("Interstitial-site finder — fast pair-circle engine")

with st.sidebar:
    st.header("Global lattice")
    a = st.number_input("a (lattice length unit)", 0.1, 10.0, 1.0, 0.01)
    b_ratio = st.number_input("b/a", 0.1, 5.0, 1.0, 0.01)
    c_ratio = st.number_input("c/a", 0.1, 5.0, 1.0, 0.01)
    alpha = st.number_input("α (deg)", 10.0, 170.0, 90.0, 0.1)
    beta  = st.number_input("β (deg)", 10.0, 170.0, 90.0, 0.1)
    gamma = st.number_input("γ (deg)", 10.0, 170.0, 90.0, 0.1)
    repeat = st.slider("Supercell repeat (for sampling region)", 1, 3, 2)

    st.header("Sampling & tolerances")
    k_samples = st.select_slider("Samples per pair-circle", options=[4,8,12,16,24,32], value=8)
    tol_inside = st.number_input("Inside tolerance (|pc| ≤ r + tol)", 0.0, 0.1, 0.01, 0.001)
    cluster_eps = st.number_input("Cluster radius (×a)", 0.01, 0.5, 0.1, 0.01)

st.subheader("Sublattices (1–3)")
def sublattice_editor(idx: int, default_name: str):
    c1, c2, c3 = st.columns(3)
    with c1:
        bravais = st.selectbox(f"Bravais {idx}", [
            "cubic_P","cubic_I","cubic_F","hexagonal_P","hexagonal_HCP",
            "tetragonal_P","tetragonal_I","orthorhombic_P","orthorhombic_F"
        ], index=0, key=f"brv{idx}")
    with c2:
        offx = st.number_input(f"offset x {idx}", -1.0, 1.0, 0.0, 0.01, key=f"ox{idx}")
        offy = st.number_input(f"offset y {idx}", -1.0, 1.0, 0.0, 0.01, key=f"oy{idx}")
        offz = st.number_input(f"offset z {idx}", -1.0, 1.0, 0.0, 0.01, key=f"oz{idx}")
    with c3:
        alpha_ratio = st.number_input(f"α ratio (radius/a·s) {idx}", 0.01, 3.0, 1.0, 0.01, key=f"ar{idx}")
        visible = st.checkbox(f"Enable {idx}", value=True, key=f"vis{idx}")

    name = st.text_input(f"Name {idx}", default_name, key=f"name{idx}")
    return Sublattice(
        name=name, bravais=bravais, offset_frac=(offx,offy,offz),
        visible=visible, alpha_ratio=alpha_ratio
    )

n_sub = st.slider("Number of sublattices", 1, 3, 1)
subs = [sublattice_editor(i+1, chr(ord('A')+i)) for i in range(n_sub)]

p = LatticeParams(a=a, b_ratio=b_ratio, c_ratio=c_ratio, alpha=alpha, beta=beta, gamma=gamma)

st.divider()
st.subheader("Search for first N-way overlap")
colA, colB, colC = st.columns([1,1,2])

with colA:
    N_target = st.slider("Target multiplicity N", 2, 12, 4)
    s_min = st.number_input("s_min", 0.0, 2.0, 0.01, 0.001)
    s_max = st.number_input("s_max", 0.0, 2.0, 0.9, 0.001)
    go = st.button("Find s*_N")

with colB:
    s_test = st.number_input("Test a single s", 0.0, 2.0, 0.35, 0.001)
    test = st.button("Evaluate max multiplicity at s")

with colC:
    st.write("**Notes**")
    st.markdown("""
- Radii are `r_i = α_i · s · a`.  
- The engine samples pair intersection circles with `k` points and counts how many spheres cover each point within a tolerance.  
- Returns **max multiplicity** and clusters nearby sample points to de-duplicate.
    """)

if test:
    m, clusters = max_multiplicity_for_scale(
        sublattices=subs, p=p, repeat=repeat, scale_s=s_test,
        k_samples=k_samples, tol_inside=tol_inside, cluster_eps=cluster_eps*a
    )
    st.success(f"At s = {s_test:.4f}, max multiplicity = {m}")
    st.caption(f"Clustered hot-spots: {len(clusters)}")

if go:
    s_star, milestones = find_threshold_s_for_N(
        N_target=N_target, sublattices=subs, p=p, repeat=repeat,
        s_min=s_min, s_max=s_max, k_samples=k_samples, tol_inside=tol_inside,
        cluster_eps=cluster_eps*a
    )
    if s_star is None:
        st.error(f"Did not reach N={N_target} in [{s_min}, {s_max}].")
    else:
        st.success(f"s*_{N_target} ≈ {s_star:.4f}")
    if milestones:
        st.write("First-seen multiplicities:")
        for m in sorted(milestones):
            st.write(f"m={m}: s≈{milestones[m]:.4f}")
