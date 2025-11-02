# ============================
"a", "b_ratio", "c_ratio", "alpha", "beta", "gamma"
], index=2)


vmin = st.number_input("Param min", value=0.5 if param in ("b_ratio","c_ratio") else (60.0 if param in ("alpha","beta","gamma") else 0.5))
vmax = st.number_input("Param max", value=1.5 if param in ("b_ratio","c_ratio") else (140.0 if param in ("alpha","beta","gamma") else 1.5))
steps = st.slider("Steps", 5, 101, 25)
Ns = [2,3,4,5,6]


srange_min = st.number_input("s_min (radius scale)", 0.0, 2.0, 0.01, 0.001, key="smin_scan")
srange_max = st.number_input("s_max (radius scale)", 0.0, 2.0, 0.9, 0.001, key="smax_scan")


if st.button("Run scan"):
xs = np.linspace(vmin, vmax, steps)
curves = {N: np.full(steps, np.nan) for N in Ns}
prog = st.progress(0, text="Scanning…")


def clone_params(base: LatticeParams, **kw) -> LatticeParams:
d = dict(a=base.a, b_ratio=base.b_ratio, c_ratio=base.c_ratio,
alpha=base.alpha, beta=base.beta, gamma=base.gamma)
d.update(kw)
return LatticeParams(**d)


for i, val in enumerate(xs):
# set parameter identically for all lattices
if param in ("a","b_ratio","c_ratio","alpha","beta","gamma"):
p_i = clone_params(p, **{param: float(val)})
else:
p_i = p
for N in Ns:
s_star, _ = find_threshold_s_for_N(
N, subs, p_i, repeat,
s_min=srange_min, s_max=srange_max,
k_samples=k_samples, tol_inside=tol_inside,
cluster_eps=cluster_eps*p_i.a, max_iter=24,
)
curves[N][i] = s_star if s_star is not None else np.nan
prog.progress(int((i+1)/steps*100), text=f"Scanning… {i+1}/{steps}")


# Plot (single figure with multiple lines)
fig = plt.figure()
for N in Ns:
plt.plot(xs, curves[N], label=f"N={N}")
plt.xlabel(param)
plt.ylabel("s*_N (min radius scale)")
plt.title("Threshold radii vs structural parameter")
plt.legend()
st.pyplot(fig, clear_figure=True)


st.caption("""
Radii are r_i = α_i · s · a. Engine uses pair-circle sampling, a fast cell-list neighbour search, clustering, and bisection to find s*_N.
Tip: for hexagonal cells set γ≈120°; for rhombohedral set α=β=γ≠90° (and often a=b=c).
""")


with colB:
N_target = st.slider("Target N", 2, 12, 4)
s_min = st.number_input("s_min", 0.0, 2.0, 0.01, 0.001)
s_max = st.number_input("s_max", 0.0, 2.0, 0.9, 0.001)
if st.button("Find s*_N (bisection)"):
s_star, milestones = find_threshold_s_for_N(
N_target, subs, p, repeat,
s_min=s_min, s_max=s_max,
k_samples=k_samples, tol_inside=tol_inside,
cluster_eps=cluster_eps*a
)
if s_star is None:
st.error(f"Did not reach N={N_target} in [{s_min}, {s_max}].")
else:
st.success(f"s*_{N_target} ≈ {s_star:.6f}")
st.caption(f"Milestones seen (first s for each m): {milestones}")


st.caption("""
Radii are r_i = α_i · s · a. Engine uses pair-circle sampling, a fast cell-list neighbour search, clustering, and bisection to find s*_N.
Tip: for hexagonal cells set γ≈120°; for rhombohedral set α=β=γ≠90° (and often a=b=c).
""")
