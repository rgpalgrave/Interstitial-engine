# =====================================================
# interstitial_engine.py
# Geometry + fast pair-circle engine for k_max and s*_N
# =====================================================

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from math import sqrt
# No matplotlib here (UI will import it)

# -----------------
# Data structures
# -----------------
@dataclass
class LatticeParams:
    a: float = 1.0
    b_ratio: float = 1.0
    c_ratio: float = 1.0
    alpha: float = 90.0  # degrees
    beta: float = 90.0
    gamma: float = 90.0

@dataclass
class Sublattice:
    name: str
    bravais: str                 # e.g. "cubic_F"
    offset_frac: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha_ratio: float = 1.0     # per-lattice radius ratio α_i
    visible: bool = True

# -----------------
# Lattices & vectors
# -----------------
def bravais_basis(bravais: str) -> np.ndarray:
    """Fractional basis points, mirroring the visualiser."""
    B = {
        # Cubic family
        "cubic_P":        [(0,0,0)],
        "cubic_I":        [(0,0,0),(0.5,0.5,0.5)],
        "cubic_F":        [(0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
        # Compound motifs on cubic
        "cubic_Diamond":  [
            (0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5),
            (0.25,0.25,0.25),(0.75,0.75,0.25),(0.75,0.25,0.75),(0.25,0.75,0.75)
        ],
        "cubic_Pyrochlore": [
            (0.5,0.5,0.5),(0,0,0.5),(0,0.5,0),(0.5,0,0),
            (0.25,0.75,0),(0.75,0.25,0),(0.75,0.75,0.5),(0.25,0.25,0.5),
            (0.75,0,0.25),(0.25,0.5,0.25),(0.25,0,0.75),(0.75,0.5,0.75),
            (0,0.25,0.75),(0.5,0.75,0.75),(0.5,0.25,0.25),(0,0.75,0.25)
        ],
        # Tetragonal
        "tetragonal_P":   [(0,0,0)],
        "tetragonal_I":   [(0,0,0),(0.5,0.5,0.5)],
        # Orthorhombic
        "orthorhombic_P": [(0,0,0)],
        "orthorhombic_C": [(0,0,0),(0.5,0.5,0)],
        "orthorhombic_I": [(0,0,0),(0.5,0.5,0.5)],
        "orthorhombic_F": [(0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
        # Hexagonal / HCP
        "hexagonal_P":    [(0,0,0)],
        "hexagonal_HCP":  [(0,0,0),(1/3,2/3,0.5)],
        # Rhombohedral
        "rhombohedral_R": [(0,0,0)],
        # Monoclinic / Triclinic
        "monoclinic_P":   [(0,0,0)],
        "monoclinic_C":   [(0,0,0),(0.5,0.5,0)],
        "triclinic_P":    [(0,0,0)],
    }
    if bravais not in B:
        raise ValueError(f"Unknown bravais type: {bravais}")
    return np.asarray(B[bravais], dtype=float)

def lattice_vectors(p: LatticeParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a, b, c vectors. Orthogonal fast path; general (α,β,γ) fallback."""
    a = p.a
    b = p.a * p.b_ratio
    c = p.a * p.c_ratio
    if abs(p.alpha-90) < 1e-9 and abs(p.beta-90) < 1e-9 and abs(p.gamma-90) < 1e-9:
        return np.array([a,0,0.0]), np.array([0.0,b,0.0]), np.array([0.0,0.0,c])

    # General cell
    alp = np.deg2rad(p.alpha); bet = np.deg2rad(p.beta); gam = np.deg2rad(p.gamma)
    avec = np.array([a,0,0], float)
    bvec = np.array([b*np.cos(gam), b*np.sin(gam), 0.0], float)
    cx  = c*np.cos(bet)
    cy  = c*(np.cos(alp) - np.cos(bet)*np.cos(gam))/max(1e-12, np.sin(gam))
    cz2 = max(0.0, c*c - cx*cx - cy*cy)
    cvec= np.array([cx, cy, sqrt(cz2)], float)
    return avec, bvec, cvec

def frac_to_cart(frac: np.ndarray, avec: np.ndarray, bvec: np.ndarray, cvec: np.ndarray) -> np.ndarray:
    return frac[0]*avec + frac[1]*bvec + frac[2]*cvec

def generate_points(s: Sublattice, p: LatticeParams, repeat: int = 1, include_ghost: bool = True) -> np.ndarray:
    a, b, c = lattice_vectors(p)
    basis = bravais_basis(s.bravais)
    start = -1 if include_ghost else 0
    end = repeat + 1 if include_ghost else repeat
    off = np.array(s.offset_frac, float)
    pts: List[np.ndarray] = []
    for i in range(start, end):
        for j in range(start, end):
            for k in range(start, end):
                cell = np.array([i, j, k], float)
                for b0 in basis:
                    f = cell + np.array(b0, float) + off
                    pts.append(frac_to_cart(f, a, b, c))
    return np.asarray(pts)

# -----------------
# Pair-circle engine
# -----------------
def pair_circle_samples(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float, k: int = 8) -> np.ndarray:
    v = c2 - c1
    d = np.linalg.norm(v)
    if d <= 1e-12:
        return np.empty((0, 3))
    if not (abs(r1 - r2) < d < (r1 + r2)):
        return np.empty((0, 3))
    n = v / d
    t = (r1*r1 - r2*r2 + d*d) / (2*d*d)
    center = c1 + t*v
    # build in-plane basis
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) <= 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(tmp, n); nu = np.linalg.norm(u)
    if nu < 1e-12:
        tmp = np.array([0.0, 0.0, 1.0])
        u = np.cross(tmp, n); nu = np.linalg.norm(u)
        if nu < 1e-12:
            return np.empty((0, 3))
    u /= nu
    v2 = np.cross(n, u)
    h2 = r1*r1 - np.dot(center - c1, center - c1)
    if h2 <= 0:
        return np.empty((0, 3))
    h = sqrt(max(0.0, h2))
    angles = np.linspace(0.0, 2.0*np.pi, k, endpoint=False)
    return center + (h*np.cos(angles))[:, None]*u + (h*np.sin(angles))[:, None]*v2

def _cluster(points: np.ndarray, counts: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy radius clustering; small N so this is fine."""
    if len(points) == 0:
        return points, counts
    used = np.zeros(len(points), dtype=bool)
    reps = []
    repc = []
    for i in range(len(points)):
        if used[i]:
            continue
        acc = [i]
        for j in range(i+1, len(points)):
            if used[j]:
                continue
            if np.linalg.norm(points[i] - points[j]) < eps:
                acc.append(j)
        used[acc] = True
        reps.append(np.mean(points[acc], axis=0))
        repc.append(int(np.max(counts[acc])))
    return np.asarray(reps), np.asarray(repc, dtype=int)

def _build_centers_and_radii(sublattices: List[Sublattice], p: LatticeParams, repeat: int, s: float) -> Tuple[np.ndarray, np.ndarray]:
    centers = []
    radii = []
    for sub in sublattices:
        if not sub.visible:
            continue
        pts = generate_points(sub, p, repeat=repeat, include_ghost=True)
        r = sub.alpha_ratio * s * p.a
        centers.append(pts)
        radii.append(np.full(len(pts), r))
    if not centers:
        return np.empty((0, 3)), np.empty((0,))
    return np.vstack(centers), np.concatenate(radii)

def _neighbor_pairs(centers: np.ndarray, rmax: float) -> np.ndarray:
    """Cell-list neighbor search: return i<j with ||ci-cj|| < 2*rmax."""
    if len(centers) == 0:
        return np.empty((0, 2), dtype=int)
    L = rmax * 2.2
    mins = centers.min(axis=0) - 1e-9
    idx = np.floor((centers - mins) / L).astype(int)
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, key in enumerate(map(tuple, idx)):
        buckets[key].append(i)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    pairs: List[Tuple[int, int]] = []
    for key, ids in buckets.items():
        nbr_ids = []
        for o in offs:
            k2 = (key[0]+o[0], key[1]+o[1], key[2]+o[2])
            if k2 in buckets:
                nbr_ids.extend(buckets[k2])
        if not nbr_ids:
            continue
        nbr_ids = np.array(sorted(set(nbr_ids)), dtype=int)
        ids_arr = np.array(ids, dtype=int)
        for i in ids_arr:
            js = nbr_ids[nbr_ids > i]
            if len(js) == 0:
                continue
            d2 = np.sum((centers[js] - centers[i])**2, axis=1)
            within = js[d2 < (2*rmax)*(2*rmax)]
            if len(within):
                pairs.extend([(i, int(j)) for j in within])
    if not pairs:
        return np.empty((0, 2), dtype=int)
    return np.asarray(pairs, dtype=int)

def max_multiplicity_for_scale(
    sublattices: List[Sublattice],
    p: LatticeParams,
    repeat: int,
    scale_s: float,
    k_samples: int = 8,
    tol_inside: float = 1e-3,
    cluster_eps: Optional[float] = None,
    early_stop_at: Optional[int] = None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    centers, radii = _build_centers_and_radii(sublattices, p, repeat, scale_s)
    if len(centers) == 0:
        return 0, np.empty((0, 3)), np.empty((0,))
    rmax = float(np.max(radii))
    pairs = _neighbor_pairs(centers, rmax)
    if len(pairs) == 0:
        return 0, np.empty((0, 3)), np.empty((0,))

    samples: List[np.ndarray] = []
    counts: List[int] = []
    for i, j in pairs:
        pts = pair_circle_samples(centers[i], radii[i], centers[j], radii[j], k=k_samples)
        if pts.size == 0:
            continue
        d = np.linalg.norm(centers[None, :, :] - pts[:, None, :], axis=2)  # [k, n]
        c = np.sum(d <= (radii[None, :] + tol_inside), axis=1)             # [k]
        if early_stop_at is not None and np.any(c >= early_stop_at):
            m = int(np.max(c))
            # keep a couple of samples to avoid empty returns
            keep = np.where(c >= 2)[0][:2]
            samples.extend(pts[keep])
            counts.extend([int(x) for x in c[keep]])
            if cluster_eps is None:
                cluster_eps = 0.1 * p.a
            reps, repc = _cluster(np.asarray(samples), np.asarray(counts, dtype=int), eps=cluster_eps)
            return m, reps, repc
        for m_i, pt in zip(c, pts):
            if m_i >= 2:
                samples.append(pt)
                counts.append(int(m_i))

    if not samples:
        return 0, np.empty((0, 3)), np.empty((0,))

    samples_arr = np.asarray(samples)
    counts_arr = np.asarray(counts, dtype=int)
    if cluster_eps is None:
        cluster_eps = 0.1 * p.a
    reps, repc = _cluster(samples_arr, counts_arr, eps=cluster_eps)
    mmax = int(repc.max()) if len(repc) else 0
    return mmax, reps, repc

def find_threshold_s_for_N(
    N_target: int,
    sublattices: List[Sublattice],
    p: LatticeParams,
    repeat: int = 1,
    s_min: float = 0.01,
    s_max: float = 0.9,
    k_samples: int = 8,
    tol_inside: float = 1e-3,
    cluster_eps: Optional[float] = None,
    max_iter: int = 28,
) -> Tuple[Optional[float], Dict[int, float]]:
    milestones: Dict[int, float] = {}
    # Coarse sweep to bracket
    grid = np.linspace(s_min, s_max, 16)
    hit = False
    for s in grid:
        m, _, _ = max_multiplicity_for_scale(
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
        m, _, _ = max_multiplicity_for_scale(
            sublattices, p, repeat, s2, k_samples, tol_inside, cluster_eps, early_stop_at=N_target
        )
        if m < N_target:
            s_lo = s2
        else:
            s_hi = s2
            break

    # bisection
    for _ in range(max_iter):
        mid = 0.5 * (s_lo + s_hi)
        m, _, _ = max_multiplicity_for_scale(
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
