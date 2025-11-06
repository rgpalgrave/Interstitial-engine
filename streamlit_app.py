# =====================================================
# streamlit_app_v2.py
# Interstitial-site finder — K_max tool (OPTIMIZED)
# Uses all_pairlist for fast parameter scans
# =====================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io, json
from typing import List, Dict, Tuple, Optional

from interstitial_engine_v2 import (
    LatticeParams, Sublattice,
    max_multiplicity_for_scale, find_threshold_s_for_N,
    lattice_vectors, bravais_basis, frac_to_cart, _cart_to_frac,
    compute_cn_representative, build_pairlist, build_all_pairlist,
    PairList, AllPairList
)

# --------- Chemistry radii ---------
ANION_RADII = {"O":1.38,"S":1.84,"Se":1.98,"F":1.33,"Cl":1.81,"Br":1.96,"I":2.20}
METAL_RADII = {
    "Li":{1:0.76},"Na":{1:1.02},"K":{1:1.38},"Rb":{1:1.52},"Cs":{1:1.67},
    "Be":{2:0.59},"Mg":{2:0.72},"Ca":{2:1.00},"Sr":{2:1.18},"Ba":{2:1.35},
    "Al":{3:0.535},"Ga":{3:0.62},"In":{3:0.80},"Tl":{1:1.59,3:0.885},
    "Si":{4:0.40},"Ge":{4:0.53},"Sn":{2:1.18,4:0.69},"Pb":{2:1.19,4:0.775},
    "Ti":{3:0.67,4:0.605},"Zr":{4:0.72},"Hf":{4:0.71},
    "V":{2:0.79},"Nb":{5:0.64},"Ta":{5:0.64},
    "Mo":{4:0.65,5:0.61,6:0.59},"W":{4:0.66,5:0.62,6:0.60},
    "Sc":{3:0.745},"Y":{3:0.90},
    "Cr":{2:0.80,3:0.615,6:0.52},
    "Mn":{2:0.83,3:0.645,4:0.67},
    "Fe":{2:0.78,3:0.645},
    "Co":{2:0.745,3:0.61},
    "Ni":{2:0.69,3:0.56},
    "Cu":{1:0.91,2:0.73},"Zn":{2:0.74},
    "Cd":{2:0.95},"Hg":{2:1.16},
    "Ru":{3:0.68,4:0.62},"Rh":{3:0.665,4:0.60},"Pd":{2:0.86,4:0.615},
    "Ag":{1:1.15},"Au":{1:1.37,3:0.85},
    "Pt":{2:0.80,4:0.625},
    "La":{3:1.032},"Ce":{3:1.01,4:0.87},"Pr":{3:0.99},"Nd":{3:0.983},
    "Sm":{3:0.958},"Eu":{2:1.25,3:0.947},"Gd":{3:0.938},"Tb":{3:0.923},
    "Dy":{3:0.912},"Ho":{3:0.901},"Er":{3:0.89},"Tm":{3:0.88},
    "Yb":{2:1.16,3:0.868},"Lu":{3:0.861},
    "B":{3:0.27},"P":{5:0.52},"As":{5:0.60},"Sb":{5:0.74},"Bi":{3:1.03,5:0.76},
}

# --------- Wyckoff presets ---------
Wyck = {
    "cubic_P":{"1a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
               "1b (1/2,1/2,1/2)":{"type":"fixed","xyz":(0.5,0.5,0.5)},
               "3c (0,1/2,1/2)":{"type":"fixed","xyz":(0,0.5,0.5)},
               "3d (1/2,0,0)":{"type":"fixed","xyz":(0.5,0,0)},
               "6e (x,0,0)":{"type":"free","xyz":("x",0,0)}},
    "cubic_F":{"4a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
               "4b (1/2,1/2,1/2)":{"type":"fixed","xyz":(0.5,0.5,0.5)},
               "8c (1/4,1/4,1/4)":{"type":"fixed","xyz":(0.25,0.25,0.25)},
               "24d (0,1/4,1/4)":{"type":"fixed","xyz":(0,0.25,0.25)},
               "24e (x,0,0)":{"type":"free","xyz":("x",0,0)},
               "32f (x,x,x)":{"type":"free","xyz":("x","x","x")}},
    "cubic_I":{"2a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
               "2b (1/2,1/2,1/2)":{"type":"fixed","xyz":(0.5,0.5,0.5)},
               "6c (0,1/2,0)":{"type":"fixed","xyz":(0,0.5,0)},
               "6d (1/2,0,0)":{"type":"fixed","xyz":(0.5,0,0)},
               "12e (x,0,0)":{"type":"free","xyz":("x",0,0)}},
    "tetragonal_P":{"1a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
                    "1b (0,0,1/2)":{"type":"fixed","xyz":(0,0,0.5)},
                    "1c (1/2,1/2,0)":{"type":"fixed","xyz":(0.5,0.5,0)},
                    "1d (1/2,1/2,1/2)":{"type":"fixed","xyz":(0.5,0.5,0.5)},
                    "2i (x,0,0)":{"type":"free","xyz":("x",0,0)},
                    "2j (x,x,0)":{"type":"free","xyz":("x","x",0)},
                    "2k (x,x,z)":{"type":"free","xyz":("x","x","z")}},
    "tetragonal_I":{"2a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
                    "2b (0,0,1/2)":{"type":"fixed","xyz":(0,0,0.5)},
                    "4d (0,1/2,1/4)":{"type":"fixed","xyz":(0,0.5,0.25)},
                    "4e (0,0,z)":{"type":"free","xyz":(0,0,"z")},
                    "8g (0,1/2,z)":{"type":"free","xyz":(0,0.5,"z")}},
    "orthorhombic_P":{"1a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
                      "1b (0,0,1/2)":{"type":"fixed","xyz":(0,0,0.5)},
                      "1c (0,1/2,0)":{"type":"fixed","xyz":(0,0.5,0)},
                      "1d (1/2,0,0)":{"type":"fixed","xyz":(0.5,0,0)},
                      "2e (x,0,0)":{"type":"free","xyz":("x",0,0)},
                      "2f (0,y,0)":{"type":"free","xyz":(0,"y",0)},
                      "2g (0,0,z)":{"type":"free","xyz":(0,0,"z")}},
    "orthorhombic_C":{"2a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
                      "2b (0,1/2,0)":{"type":"fixed","xyz":(0,0.5,0)},
                      "4g (0,y,0)":{"type":"free","xyz":(0,"y",0)}},
    "orthorhombic_I":{"2a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
                      "2b (1/2,1/2,1/2)":{"type":"fixed","xyz":(0.5,0.5,0.5)},
                      "4e (0,0,z)":{"type":"free","xyz":(0,0,"z")}},
    "orthorhombic_F":{"4a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
                      "4b (1/2,1/2,1/2)":{"type":"fixed","xyz":(0.5,0.5,0.5)},
                      "8f (x,0,0)":{"type":"free","xyz":("x",0,0)}},
    "hexagonal_P":{"1a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
                   "1b (0,0,1/2)":{"type":"fixed","xyz":(0,0,0.5)},
                   "2c (1/3,2/3,0)":{"type":"fixed","xyz":(1/3,2/3,0)},
                   "2d (1/3,2/3,1/2)":{"type":"fixed","xyz":(1/3,2/3,0.5)},
                   "2e (x,0,0)":{"type":"free","xyz":("x",0,0)},
                   "2f (x,x,0)":{"type":"free","xyz":("x","x",0)},
                   "2g (x,x,z)":{"type":"free","xyz":("x","x","z")}},
    "hexagonal_HCP":{"2a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
                     "2b (0,0,1/4)":{"type":"fixed","xyz":(0,0,0.25)},
                     "2c (1/3,2/3,1/4)":{"type":"fixed","xyz":(1/3,2/3,0.25)},
                     "2d (1/3,2/3,3/4)":{"type":"fixed","xyz":(1/3,2/3,0.75)},
                     "4f (1/3,2/3,z)":{"type":"free","xyz":(1/3,2/3,"z")}},
    "rhombohedral_R":{"3a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
                      "3b (0,0,1/2)":{"type":"fixed","xyz":(0,0,0.5)},
                      "6c (0,0,z)":{"type":"free","xyz":(0,0,"z")}},
    "monoclinic_P":{"1a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
                    "2e (x,0,0)":{"type":"free","xyz":("x",0,0)}},
    "monoclinic_C":{"2a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
                    "4i (x,0,0)":{"type":"free","xyz":("x",0,0)}},
    "triclinic_P":{"1a (0,0,0)":{"type":"fixed","xyz":(0,0,0)},
                   "1b (x,y,z)":{"type":"free","xyz":("x","y","z")}},
}

st.set_page_config(page_title="Interstitial-site finder — K_max scans (OPTIMIZED)", layout="wide")
st.title("Interstitial-site finder — K_max scans (OPTIMIZED v2)")

# ---------- Global lattice ----------
st.sidebar.header("Global lattice")
global_bravais_choices = list(Wyck.keys())
global_bravais = st.sidebar.selectbox("Bravais / Space-group family", global_bravais_choices,
                                      index=global_bravais_choices.index("cubic_F"))
a = st.sidebar.number_input("a (Å)", 0.1, 10.0, 1.0, 0.01)
b_ratio = st.sidebar.number_input("b/a", 0.1, 5.0, 1.0, 0.01)
c_ratio = st.sidebar.number_input("c/a", 0.1, 5.0, 1.0, 0.01)
alpha = st.sidebar.number_input("α (deg)", 10.0, 170.0, 90.0, 0.1)
beta  = st.sidebar.number_input("β (deg)", 10.0, 170.0, 90.0, 0.1)
gamma = st.sidebar.number_input("γ (deg)", 10.0, 170.0, 90.0, 0.1)

st.sidebar.header("Sampling & tolerances")
k_coarse = st.sidebar.select_slider("Samples per pair-circle (coarse)", options=[4,6,8,12], value=6)
k_fine   = st.sidebar.select_slider("Samples per pair-circle (fine)",   options=[8,12,16,24,32], value=12)
tol_inside = st.sidebar.number_input("Inside tolerance (Å)", 0.0, 0.02, 0.001, 0.0005)
cluster_eps = st.sidebar.number_input("Cluster radius (×a)", 0.01, 0.5, 0.1, 0.01)

# ---------- Sublattices + chemistry ----------
st.subheader("Sublattices (1–6)")
chem_mode = st.checkbox("Chemistry mode (set α from metal + anion radii)", value=True)
anion = st.selectbox("Global anion", list(ANION_RADII.keys()), index=list(ANION_RADII.keys()).index("O"))
r_anion = ANION_RADII[anion]

def edit_sublattice(idx: int, use_chem: bool) -> Sublattice:
    with st.expander(f"Sublattice {idx} settings", expanded=(idx<=3)):
        wyck_labels = ["Free position"] + list(Wyck[global_bravais].keys())
        choice = st.selectbox(f"Site {idx}", wyck_labels, index=0, key=f"site{idx}")

        offx=offy=offz=0.0
        if choice == "Free position":
            ccols = st.columns(3)
            offx = ccols[0].number_input(f"offset x {idx}", -1.0, 1.0, 0.0, 0.01, key=f"offx{idx}")
            offy = ccols[1].number_input(f"offset y {idx}", -1.0, 1.0, 0.0, 0.01, key=f"offy{idx}")
            offz = ccols[2].number_input(f"offset z {idx}", -1.0, 1.0, 0.0, 0.01, key=f"offz{idx}")
        else:
            spec = Wyck[global_bravais][choice]
            xyz = spec["xyz"]
            def ent(val, tag):
                if isinstance(val, str):
                    return st.number_input(f"{choice}: {val} {idx}", 0.0, 1.0, 0.25, 0.005, key=f"{val}{idx}_{choice}")
                return float(val)
            offx = ent(xyz[0],"x"); offy = ent(xyz[1],"y"); offz = ent(xyz[2],"z")
            st.caption(f"Preset {choice}: ({offx:.4f}, {offy:.4f}, {offz:.4f})")

        name = st.text_input(f"Name {idx}", f"Sub{idx}", key=f"name{idx}")
        enabled = st.checkbox(f"Enable {idx}", value=True, key=f"vis{idx}")

        if use_chem:
            c1,c2 = st.columns(2)
            elements = sorted(METAL_RADII.keys())
            default_el = "Fe" if idx==1 else elements[0]
            el = c1.selectbox(f"Metal {idx}", elements, index=elements.index(default_el), key=f"el{idx}")
            ox_states = sorted(METAL_RADII[el].keys())
            oxid = c1.selectbox(f"Oxidation {idx}", ox_states, index=0, key=f"oxid{idx}")
            r_default = METAL_RADII[el][oxid]
            r_override = c2.number_input(f"Manual metal radius {idx} (Å, optional)", 0.0, 3.0, float(r_default), 0.01, key=f"rman{idx}")
            use_override = c2.checkbox(f"Use manual metal radius {idx}", value=False, key=f"useman{idx}")
            r_metal = float(r_override) if use_override else float(r_default)
            alpha_ratio = r_metal + r_anion
            st.caption(f"α_{idx} = r_metal ({r_metal:.3f}) + r_{anion} ({r_anion:.3f}) = {alpha_ratio:.3f} Å")
        else:
            alpha_ratio = st.number_input(f"α ratio {idx}", 0.01, 5.0, 1.0, 0.01, key=f"alpha{idx}")

    return Sublattice(name=name, bravais=global_bravais, offset_frac=(offx,offy,offz), alpha_ratio=alpha_ratio, visible=enabled)

n_sub = st.slider("Number of sublattices", 1, 6, 1)
subs: List[Sublattice] = [edit_sublattice(i+1, chem_mode) for i in range(n_sub)]
p = LatticeParams(a=a,b_ratio=b_ratio,c_ratio=c_ratio,alpha=alpha,beta=beta,gamma=gamma)

# ---------- Evaluate / s*_N ----------
st.divider()
st.subheader("Evaluate multiplicity or find s*_N")

colA,colB = st.columns(2)
with colA:
    s_test = st.number_input("Evaluate at s", 0.0, 5.0, 0.35, 0.001)
    if st.button("Compute k_max(s)"):
        m, reps, repc = max_multiplicity_for_scale(
            subs, p, repeat_ignored=1, scale_s=s_test,
            k_samples=k_fine, tol_inside=tol_inside,
            cluster_eps=cluster_eps * a, early_stop_at=None
        )
        st.success(f"k_max(s={s_test:.4f}) = {m}")
        st.caption(f"Hotspots (clustered): {len(reps)}")

with colB:
    N_target = st.slider("Target N", 2, 24, 6)
    s_min = st.number_input("s_min", 0.0, 5.0, 0.01, 0.001)
    s_max = st.number_input("s_max", 0.0, 5.0, 0.9, 0.001)
    use_all_pairlist = st.checkbox("Accelerate with all-pairs (MUCH faster)", value=True)
    if st.button("Find s*_N (bisection)"):
        all_pl: Optional[AllPairList] = build_all_pairlist(subs, p) if use_all_pairlist else None
        s_star, milestones = find_threshold_s_for_N(
            N_target, subs, p, repeat_ignored=1,
            s_min=s_min, s_max=s_max,
            k_samples_coarse=k_coarse, k_samples_fine=k_fine,
            tol_inside=tol_inside, cluster_eps=cluster_eps * a,
            all_pairlist=all_pl
        )
        if s_star is None:
            st.error(f"Did not reach N={N_target} in [{s_min}, {s_max}].")
        else:
            st.success(f"s*_{N_target} ≈ {s_star:.6f}")
            st.caption(f"Milestones seen (first s for each m): {milestones}")

# ---------- Parameter scan (OPTIMIZED) ----------
st.divider()
st.subheader("Parameter scan: s*_N vs parameter  •  Export CSV / JSON")

struct_params = ["a","b_ratio","c_ratio","alpha","beta","gamma"]
alpha_targets = ["alpha_scalar (all sublattices)"] + [f"alpha(Sub {i+1})" for i in range(n_sub)]
scan_target = st.selectbox("Parameter to scan", struct_params + alpha_targets, index=2, key="scan_target")

if scan_target in ("b_ratio","c_ratio"):
    vmin_default, vmax_default, xlabel = 0.5, 1.5, scan_target
elif scan_target in ("alpha","beta","gamma"):
    vmin_default, vmax_default, xlabel = 60.0, 140.0, f"{scan_target} (deg)"
elif scan_target.startswith("alpha"):
    vmin_default, vmax_default, xlabel = 0.5, 2.0, scan_target
else:
    vmin_default, vmax_default, xlabel = 0.5, 1.5, "a (Å)"

vmin = st.number_input("Scan min", value=float(vmin_default), key="scan_vmin")
vmax = st.number_input("Scan max", value=float(vmax_default), key="scan_vmax")
steps = st.slider("Steps", 5, 401, 51, key="scan_steps")

Ns = [2,3,4,5,6]
srange_min = st.number_input("s_min (radius scale)", 0.0, 5.0, 0.01, 0.001, key="scan_smin")
srange_max = st.number_input("s_max (radius scale)", 0.0, 5.0, 0.9, 0.001, key="scan_smax")

def clone_params(base: LatticeParams, **kw) -> LatticeParams:
    d = dict(a=base.a, b_ratio=base.b_ratio, c_ratio=base.c_ratio,
             alpha=base.alpha, beta=base.beta, gamma=base.gamma)
    d.update(kw)
    return LatticeParams(**d)

def apply_alpha_scan(subs_in: List[Sublattice], mode: str, val: float) -> List[Sublattice]:
    out = []
    for i, s in enumerate(subs_in):
        if mode == "alpha_scalar (all sublattices)":
            new_alpha = s.alpha_ratio * val
        elif mode.startswith("alpha(Sub"):
            try: idx = int(mode.split("Sub")[1].split(")")[0].strip()) - 1
            except Exception: idx = -1
            new_alpha = s.alpha_ratio * val if i == idx else s.alpha_ratio
        else:
            new_alpha = s.alpha_ratio
        out.append(Sublattice(s.name, s.bravais, s.offset_frac, new_alpha, s.visible))
    return out

run_scan = st.button("Run scan", key="run_scan_btn")

if run_scan:
    xs = np.linspace(vmin, vmax, int(steps))
    curves = {N: np.full(xs.shape, np.nan) for N in Ns}
    progress = st.progress(0, text="Scanning…")

    static_centers = scan_target.startswith("alpha")
    
    # NEW: Pre-compute all-pairs once for alpha scans
    global_all_pairlist: Optional[AllPairList] = None
    if static_centers:
        # For alpha scans, geometry is fixed, only alphas change
        # So we can pre-compute all pairs once
        global_all_pairlist = build_all_pairlist(subs, p)

    for i, val in enumerate(xs):
        if scan_target in struct_params:
            # Lattice param changed → rebuild geometry + all-pairs
            p_i = clone_params(p, **{scan_target: float(val)})
            subs_i = subs
            all_pl_i = build_all_pairlist(subs_i, p_i)
        else:
            # Alpha scan → reuse geometry, just change alphas
            p_i = p
            subs_i = apply_alpha_scan(subs, scan_target, float(val))
            all_pl_i = global_all_pairlist

        for N in Ns:
            s_star, _ = find_threshold_s_for_N(
                N_target=N, sublattices=subs_i, p=p_i, repeat_ignored=1,
                s_min=srange_min, s_max=srange_max,
                k_samples_coarse=k_coarse, k_samples_fine=k_fine,
                tol_inside=tol_inside, cluster_eps=cluster_eps * p_i.a,
                max_iter=20, all_pairlist=all_pl_i
            )
            curves[N][i] = s_star if s_star is not None else np.nan

        progress.progress(int((i+1)/len(xs)*100), text=f"Scanning… {i+1}/{len(xs)}")

    # Plot
    fig, ax = plt.subplots()
    for N in Ns:
        ax.plot(xs, curves[N], label=f"N={N}")
    ax.set_xlabel(xlabel); ax.set_ylabel("s*_N (min radius scale)")
    ax.set_title("Threshold radii vs parameter"); ax.legend()
    st.pyplot(fig, clear_figure=True); plt.close(fig)

    # Save results
    st.session_state["last_scan"] = {
        "scan_target": scan_target,
        "x_label": xlabel,
        "x_values": xs.tolist(),
        "s_min": float(srange_min),
        "s_max": float(srange_max),
        "Ns": Ns,
        "curves": {int(N): curves[N].tolist() for N in Ns},
        "global_lattice": {
            "bravais": global_bravais,
            "a": a, "b_ratio": b_ratio, "c_ratio": c_ratio,
            "alpha": alpha, "beta": beta, "gamma": gamma
        },
        "sublattices": [
            {
                "name": s.name,
                "bravais": s.bravais,
                "offset_frac": [float(s.offset_frac[0]), float(s.offset_frac[1]), float(s.offset_frac[2])],
                "alpha_ratio": float(s.alpha_ratio),
                "visible": bool(s.visible)
            }
            for s in subs
        ],
        "chemistry": {"anion": anion, "r_anion": r_anion}
    }

# ---------- Export ----------
st.divider()
st.subheader("Export last scan")

if "last_scan" in st.session_state:
    payload = st.session_state["last_scan"]
    xs = payload["x_values"]
    Ns = payload["Ns"]
    rows = []
    for i, x in enumerate(xs):
        row = {"x": x}
        for N in Ns:
            row[f"s*_N={N}"] = payload["curves"][int(N)][i]
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")
    json_bytes = json.dumps(payload, indent=2).encode("utf-8")

    st.dataframe(df.head(min(10, len(df))))
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="⬇️ Download CSV (x, s*₂..s*₆)",
            data=csv_bytes, file_name="kmax_scan.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            label="⬇️ Download JSON (full metadata)",
            data=json_bytes, file_name="kmax_scan.json",
            mime="application/json"
        )
    st.caption("CSV is compact for analysis; JSON preserves full lattice/sublattice setup for reproducibility.")
else:
    st.info("Run a scan to enable export.")
