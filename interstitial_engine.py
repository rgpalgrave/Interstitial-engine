# ============================
) -> Tuple[Optional[float], Dict[int,float]]:
milestones: Dict[int,float] = {}
# Coarse sweep to bracket
grid = np.linspace(s_min, s_max, 16)
hit = False
for s in grid:
m,_,_ = max_multiplicity_for_scale(
sublattices, p, repeat, s, k_samples, tol_inside, cluster_eps, early_stop_at=N_target
)
milestones.setdefault(int(m), float(s))
if m >= N_target:
hit = True
break
if not hit:
return None, milestones
# refine bracket
s_hi = s
s_lo = s_min
for s2 in np.linspace(s_min, s_hi, 16):
m,_,_ = max_multiplicity_for_scale(
sublattices, p, repeat, s2, k_samples, tol_inside, cluster_eps, early_stop_at=N_target
)
if m < N_target:
s_lo = s2
else:
s_hi = s2
break
# bisection
for _ in range(max_iter):
mid = 0.5*(s_lo+s_hi)
m,_,_ = max_multiplicity_for_scale(
sublattices, p, repeat, mid, k_samples, tol_inside, cluster_eps, early_stop_at=N_target
)
if m >= N_target:
s_hi = mid
else:
s_lo = mid
if abs(s_hi - s_lo) < 1e-4:
break
milestones[N_target] = s_hi
return s_hi, milestones

