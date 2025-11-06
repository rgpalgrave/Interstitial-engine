# =====================================================
# interstitial_engine.py (TURBO)
# Minimal-image + KDTree + pruned pairs + early-stop
# =====================================================

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple, Optional, Dict, List
import numpy as np
from math import sqrt
try:
    # KDTree greatly reduces counting cost
    from scipy.spatial import cKDTree as KDTree
    _HAVE_SCIPY = True
except Exception:
    KDTree = None
    _HAVE_SCIPY = False


# -----------------
# Data structures
# -----------------
@dataclass
class LatticeParams:
    a: float = 1.0
    b_ratio: float = 1.0
    c_ratio: float = 1.0
    alpha: float = 90.0  # deg
    beta: float = 90.0
    gamma: float = 90.0

@dataclass
class Sublattice:
    name: str
    bravais: str
    offset_frac: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha_ratio: float = 1.0
    visible: bool = True


# -----------------
# Lattices & vectors
# -----------------
def bravais_basis(bravais: str) -> np.ndarray:
    B = {
        # Cubic
        "cubic_P":        [(0,0,0)],
        "cubic_I":        [(0,0,0),(0.5,0.5,0.5)],
        "cubic_F":        [(0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
        # Compounds
        "cubic_Diamond":  [
            (0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5),
            (0.25,0.25,0.25),(0.75,0.75,0.25),(0.75,0.25,0.75),(0.25,0.75,0.75)
        ],
        "cubic_Pyrochlore":[
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
        # Hexagonal
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
    a = p.a
    b = p.a * p.b_ratio
    c = p.a * p.c_ratio
    if abs(p.alpha-90)<1e-9 and abs(p.beta-90)<1e-9 and abs(p.gamma-90)<1e-9:
        return np.array([a,0,0.0]), np.array([0.0,b,0.0]), np.array([0.0,0.0,c])

    alp = np.deg2rad(p.alpha); bet = np.deg2rad(p.beta); gam = np.deg2rad(p.gamma)
    avec = np.array([a,0,0], float)
    bvec = np.array([b*np.cos(gam), b*np.sin(gam), 0.0], float)
    cx  = c*np.cos(bet)
    cy  = c*(np.cos(alp) - np.cos(bet)*np.cos(gam))/max(1e-12, np.sin(gam))
    cz2 = max(0.0, c*c - cx*cx - cy*cy)
    cvec= np.array([cx, cy, sqrt(cz2)], float)
    return avec, bvec, cvec

def frac_to_cart(frac: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return frac[0]*a + frac[1]*b + frac[2]*c


# -----------------
# Minimal-image centers (no supercell)
# -----------------
def generate_centers_minimal(sub: Sublattice, p: LatticeParams) -> np.ndarray:
    """Return centers of a sublattice for the single reference cell only."""
    a,b,c = lattice_vectors(p)
    basis = bravais_basis(sub.bravais)
    off = np.array(sub.offset_frac, float)
    pts = [frac_to_cart(np.asarray(b0)+off, a,b,c) for b0 in basis]
    return np.asarray(pts, float)

@lru_cache(maxsize=128)
def centers_alphas_and_shifts(subs_key, p_key):
    """Cache central-cell centers, per-center alpha, and 27 neighbor lattice shifts."""
    p = LatticeParams(*p_key)
    a,b,c = lattice_vectors(p)
    # 27 neighbor shifts (fractional -1,0,1 → cartesian)
    shifts = []
    for i in (-1,0,1):
        for j in (-1,0,1):
            for k in (-1,0,1):
                shifts.append(frac_to_cart(np.array([i,j,k], float), a,b,c))
    shifts = np.asarray(shifts, float)  # (27,3)

    centers = []
    alphas = []
    for (bravais, ox,oy,oz, alpha_ratio, visible) in subs_key:
        if not visible: 
            continue
        sub = Sublattice("x", bravais, (ox,oy,oz), alpha_ratio, True)
        pts = generate_centers_minimal(sub, p)
        centers.append(pts)
        alphas.append(np.full(len(pts), alpha_ratio, float))
    if centers:
        centers = np.vstack(centers)
        alphas  = np.concatenate(alphas)
    else:
        centers = np.empty((0,3)); alphas = np.empty((0,))
    return centers, alphas, shifts


# -----------------
# Pair list under PBC (reference→neighbors)
# -----------------
def periodic_candidate_pairs(centers: np.ndarray, shifts: np.ndarray, cutoff: float) -> List[Tuple[int,int,int]]:
    """
    Build (i, j, s_idx) with i from reference cell, j from reference cell, and s_idx selecting a neighbor shift.
    Keep pairs where distance between centers[i] and centers[j]+shift < cutoff.
    Only emit i < (j,shift) in a canonical ordering to avoid duplicates.
    """
    if len(centers) == 0:
        return []
    pairs: List[Tuple[int,int,int]] = []
    cutoff2 = cutoff*cutoff
    n = len(centers)
    # Precompute shifted copies per shift to avoid recomputing sums per pair
    # But keep memory sane: do it per shift
    for s_idx, S in enumerate(shifts):
        shifted = centers + S  # (n,3)
        # distance matrix to same n centers → but prune by row-wise k-d selection:
        # simple block approach to cap memory:
        B = 512
        for start in range(0, n, B):
            end = min(n, start+B)
            block = centers[start:end]  # (b,3)
            d2 = np.sum((block[:,None,:] - shifted[None,:,:])**2, axis=2)  # (b,n)
            # For canonical ordering, avoid emitting (i,j,s) where s is (0,0,0) and j<=i
            if s_idx == 13:  # shift (0,0,0) in our 27 ordering? (middle of list)
                # safer: compute index by known layout: i in (-1,0,1)^3 → index = (i+1)*9 + (j+1)*3 + (k+1)
                pass
            # We'll simply keep all and deduplicate later by a strict rule:
            bi, bj = np.where(d2 < cutoff2)
            for ii, jj in zip(bi, bj):
                gi = start + ii
                # canonical rule: (shift index, j, i) must be lexicographically greater than (13, i, i)
                # To keep it simple and robust, only keep:
                #   - s_idx > 13, or
                #   - s_idx == 13 and jj > gi
                if (s_idx > 13) or (s_idx == 13 and jj > gi):
                    pairs.append((gi, jj, s_idx))
    return pairs


# -----------------
# Pair-circle sampling
# -----------------
def pair_circle_samples(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float, k: int = 8) -> np.ndarray:
    v = c2 - c1
    d2 = float(np.dot(v,v))
    if d2 <= 1e-24:
        return np.empty((0,3))
    d = sqrt(d2)
    if not (abs(r1 - r2) < d < (r1 + r2)):
        return np.empty((0,3))
    n = v / d
    t = (r1*r1 - r2*r2 + d*d) / (2*d*d)
    center = c1 + t*v
    tmp = np.array([1.0,0.0,0.0]) if abs(n[0]) <= 0.9 else np.array([0.0,1.0,0.0])
    u = np.cross(tmp, n); nu = float(np.linalg.norm(u))
    if nu < 1e-12:
        tmp = np.array([0.0,0.0,1.0])
        u = np.cross(tmp, n); nu = float(np.linalg.norm(u))
        if nu < 1e-12:
            return np.empty((0,3))
    u /= nu
    w = np.cross(n,u)
    off = center - c1
    h2 = r1*r1 - float(np.dot(off,off))
    if h2 <= 0.0:
        return np.empty((0,3))
    h = sqrt(h2)
    ang = np.linspace(0.0, 2.0*np.pi, k, endpoint=False)
    return center + (h*np.cos(ang))[:,None]*u + (h*np.sin(ang))[:,None]*w


def _cluster(points: np.ndarray, counts: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, counts
    used = np.zeros(len(points), bool)
    reps = []
    repc = []
    eps2 = eps*eps
    for i in range(len(points)):
        if used[i]: 
            continue
        acc = [i]
        for j in range(i+1, len(points)):
            if used[j]: 
                continue
            d2 = float(np.dot(points[i]-points[j], points[i]-points[j]))
            if d2 < eps2:
                acc.append(j)
        used[acc] = True
        reps.append(np.mean(points[acc], axis=0))
        repc.append(int(np.max(counts[acc])))
    return np.asarray(reps), np.asarray(repc, int)


# -----------------
# Cached geometry bundle (minimal image)
# -----------------
@dataclass(frozen=True)
class GeoKey:
    p: Tuple[float,float,float,float,float,float]
    subs: Tuple[Tuple[str,float,float,float,float,bool], ...]  # (bravais, offx,offy,offz, alpha_ratio, visible)

def _make_key(sublattices: List[Sublattice], p: LatticeParams) -> GeoKey:
    subs_key = tuple((s.bravais, float(s.offset_frac[0]), float(s.offset_frac[1]), float(s.offset_frac[2]),
                      float(s.alpha_ratio), bool(s.visible)) for s in sublattices)
    p_key = (float(p.a), float(p.b_ratio), float(p.c_ratio), float(p.alpha), float(p.beta), float(p.gamma))
    return GeoKey(p=p_key, subs=subs_key)

@lru_cache(maxsize=256)
def build_geo(key: GeoKey):
    centers, alphas, shifts = centers_alphas_and_shifts(key.subs, key.p)
    # Precompute a canonical “same-cell” index of the zero shift (13) for dedup rule
    # We stored shifts in nested loops (-1..1). The middle element index is 13.
    zero_shift_idx = 13
    return centers, alphas, shifts, zero_shift_idx


# -----------------
# Public API
# -----------------
def _build_pairs_for_cutoff(centers: np.ndarray, shifts: np.ndarray, cutoff: float) -> List[Tuple[int,int,int]]:
    # Pairs under PBC up to cutoff
    return periodic_candidate_pairs(centers, shifts, cutoff)

def _count_multiplicity_kdtree(sample_pts: np.ndarray, tree: KDTree, radii: np.ndarray, tol_inside: float) -> np.ndarray:
    # We want count of spheres whose surface passes within tol of each sample point.
    # Approx: treat as center distance <= r + tol_inside
    # For speed, query neighbor centers within rmax+tol, then check per-sphere radius threshold.
    rmax = float(np.max(radii))
    idxs = tree.query_ball_point(sample_pts, r=rmax + tol_inside)  # list of lists
    out = np.zeros(len(sample_pts), dtype=int)
    for i, neigh in enumerate(idxs):
        if not neigh:
            continue
        d = np.linalg.norm(tree.data[neigh] - sample_pts[i], axis=1)
        out[i] = int(np.sum(d <= (radii[neigh] + tol_inside)))
    return out

def max_multiplicity_for_scale(
    sublattices: List[Sublattice],
    p: LatticeParams,
    repeat_ignored: int,  # kept for UI compatibility; minimal-image ignores it
    scale_s: float,
    k_samples: int = 8,
    tol_inside: float = 1e-3,
    cluster_eps: Optional[float] = None,
    early_stop_at: Optional[int] = None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    key = _make_key(sublattices, p)
    centers, alphas, shifts, zero_idx = build_geo(key)
    if centers.size == 0:
        return 0, np.empty((0,3)), np.empty((0,))
    radii = (alphas * scale_s * p.a).astype(float)
    rmax = float(np.max(radii))

    # Build pairs once per s using cutoff = 2*rmax (pruned by distance)
    pairs = _build_pairs_for_cutoff(centers, shifts, cutoff=2.0*rmax)
    if not pairs:
        return 0, np.empty((0,3)), np.empty((0,))

    # KDTree on all 27 images? We only need counts versus sphere centers in *all* images.
    # Build a merged set of centers over 27 shifts (but keep memory ok):
    # Instead, build one KDTree on the central cell, but query across all shifts by transforming sample points back.
    # Trick: For each shift S, counting (centers + S) vs point P equals counting centers vs (P - S).
    if _HAVE_SCIPY:
        tree = KDTree(centers.copy())
    else:
        tree = None  # will fallback to slower vector method

    samples: List[np.ndarray] = []
    counts: List[int] = []

    for (i, j, s_idx) in pairs:
        # construct the two sphere centers under this shift
        c1 = centers[i]
        c2 = centers[j] + shifts[s_idx]
        r1 = radii[i]; r2 = radii[j]
        pts = pair_circle_samples(c1, r1, c2, r2, k=k_samples)
        if pts.size == 0:
            continue

        # Count multiplicity across all periodic images using KDTree trick:
        if _HAVE_SCIPY:
            total = np.zeros(len(pts), dtype=int)
            for S in shifts:
                # centers+S vs P  <=> centers vs (P - S)
                P = pts - S
                c = _count_multiplicity_kdtree(P, tree, radii, tol_inside)
                total += c
            c = total
        else:
            # Fallback (slower): explicitly build 27 images once per call
            all_centers = np.vstack([centers + S for S in shifts])
            all_radii   = np.tile(radii, len(shifts))
            d = np.linalg.norm(all_centers[None,:,:] - pts[:,None,:], axis=2)
            c = np.sum(d <= (all_radii[None,:] + tol_inside), axis=1)

        if early_stop_at is not None and np.any(c >= early_stop_at):
            keep = np.where(c >= 2)[0][:3]
            samples.extend(pts[keep])
            counts.extend([int(x) for x in c[keep]])
            m = int(np.max(c))
            if cluster_eps is None:
                cluster_eps = 0.1 * p.a
            reps, repc = _cluster(np.asarray(samples), np.asarray(counts, int), eps=cluster_eps)
            return m, reps, repc

        good = np.where(c >= 2)[0]
        if good.size:
            samples.extend(pts[good])
            counts.extend([int(x) for x in c[good]])

    if not samples:
        return 0, np.empty((0,3)), np.empty((0,))

    if cluster_eps is None:
        cluster_eps = 0.1 * p.a
    reps, repc = _cluster(np.asarray(samples), np.asarray(counts, int), eps=cluster_eps)
    mmax = int(repc.max()) if len(repc) else 0
    return mmax, reps, repc


def find_threshold_s_for_N(
    N_target: int,
    sublattices: List[Sublattice],
    p: LatticeParams,
    repeat: int = 1,          # ignored (minimal-image)
    s_min: float = 0.01,
    s_max: float = 0.9,
    k_samples_coarse: int = 6,
    k_samples_fine: int = 12,
    tol_inside: float = 1e-3,
    cluster_eps: Optional[float] = None,
    max_iter: int = 28,
) -> Tuple[Optional[float], Dict[int, float]]:
    milestones: Dict[int, float] = {}
    # coarse sweep
    for s in np.linspace(s_min, s_max, 16):
        m,_,_ = max_multiplicity_for_scale(
            sublattices, p, 1, s,
            k_samples=k_samples_coarse, tol_inside=tol_inside,
            cluster_eps=cluster_eps, early_stop_at=N_target
        )
        milestones.setdefault(int(m), float(s))
        if m >= N_target:
            s_hi = s
            break
    else:
        return None, milestones

    # refine low bound
    s_lo = s_min
    for s2 in np.linspace(s_min, s_hi, 16):
        m,_,_ = max_multiplicity_for_scale(
            sublattices, p, 1, s2,
            k_samples=k_samples_coarse, tol_inside=tol_inside,
            cluster_eps=cluster_eps, early_stop_at=N_target
        )
        if m < N_target: s_lo = s2
        else:
            s_hi = s2; break

    # bisection
    for _ in range(max_iter):
        mid = 0.5*(s_lo+s_hi)
        ks = k_samples_coarse if (s_hi - s_lo) > 0.02 else k_samples_fine
        m,_,_ = max_multiplicity_for_scale(
            sublattices, p, 1, mid,
            k_samples=ks, tol_inside=tol_inside,
            cluster_eps=cluster_eps, early_stop_at=N_target
        )
        if m >= N_target: s_hi = mid
        else: s_lo = mid
        if abs(s_hi - s_lo) < 1e-4: break

    milestones[N_target] = s_hi
    return s_hi, milestones
