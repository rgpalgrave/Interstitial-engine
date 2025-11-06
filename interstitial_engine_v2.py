# =====================================================
# interstitial_engine_v2.py
# OPTIMIZED for speed: vectorized shifts, pre-computed pairs
# =====================================================

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
import math
import numpy as np

try:
    from scipy.spatial import KDTree
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# -------------------------------
# Data classes
# -------------------------------

@dataclass(frozen=True)
class LatticeParams:
    a: float = 1.0
    b_ratio: float = 1.0
    c_ratio: float = 1.0
    alpha: float = 90.0  # deg
    beta:  float = 90.0  # deg
    gamma: float = 90.0  # deg

@dataclass(frozen=True)
class Sublattice:
    name: str
    bravais: str
    offset_frac: Tuple[float, float, float]
    alpha_ratio: float  # r_i = alpha_ratio * s * a
    visible: bool = True


# -------------------------------
# Cell geometry utilities
# -------------------------------

def lattice_vectors(p: LatticeParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = float(p.a)
    b = float(p.a * p.b_ratio)
    c = float(p.a * p.c_ratio)
    alpha = math.radians(p.alpha)
    beta  = math.radians(p.beta)
    gamma = math.radians(p.gamma)

    ax = a; ay = 0.0; az = 0.0
    bx = b * math.cos(gamma); by = b * math.sin(gamma); bz = 0.0

    cx = c * math.cos(beta)
    s_gamma = math.sin(gamma) if abs(math.sin(gamma)) > 1e-12 else 1e-12
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / s_gamma
    cz_sq = c*c - cx*cx - cy*cy
    cz = math.sqrt(max(cz_sq, 0.0))

    a_vec = np.array([ax, ay, az], float)
    b_vec = np.array([bx, by, bz], float)
    c_vec = np.array([cx, cy, cz], float)
    return a_vec, b_vec, c_vec

def frac_to_cart(frac: np.ndarray, a_vec: np.ndarray, b_vec: np.ndarray, c_vec: np.ndarray) -> np.ndarray:
    M = np.vstack([a_vec, b_vec, c_vec]).T
    f = np.asarray(frac, float)
    return (f @ M.T)

def _cart_to_frac(cart: np.ndarray, a_vec: np.ndarray, b_vec: np.ndarray, c_vec: np.ndarray) -> np.ndarray:
    M = np.vstack([a_vec, b_vec, c_vec]).T
    return np.linalg.solve(M, np.asarray(cart).T).T


# -------------------------------
# Bravais bases (primitive motifs)
# -------------------------------

def bravais_basis(bravais: str) -> List[Tuple[float, float, float]]:
    b = bravais
    if b == "cubic_P": return [(0,0,0)]
    if b == "cubic_I": return [(0,0,0), (0.5,0.5,0.5)]
    if b == "cubic_F": return [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]
    if b == "cubic_Diamond": return [(0,0,0), (0.25,0.25,0.25)]
    if b == "cubic_Pyrochlore": return [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]
    if b == "tetragonal_P": return [(0,0,0)]
    if b == "tetragonal_I": return [(0,0,0), (0.5,0.5,0.5)]
    if b == "orthorhombic_P": return [(0,0,0)]
    if b == "orthorhombic_C": return [(0,0,0), (0,0.5,0.5)]
    if b == "orthorhombic_I": return [(0,0,0), (0.5,0.5,0.5)]
    if b == "orthorhombic_F": return [(0,0,0), (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5)]
    if b == "hexagonal_P": return [(0,0,0)]
    if b == "hexagonal_HCP": return [(0,0,0), (2/3, 1/3, 0.5)]
    if b == "rhombohedral_R": return [(0,0,0)]
    if b == "monoclinic_P": return [(0,0,0)]
    if b == "monoclinic_C": return [(0,0,0), (0,0.5,0)]
    if b == "triclinic_P": return [(0,0,0)]
    return [(0,0,0)]


def generate_centers_minimal(sub: Sublattice, p: LatticeParams) -> np.ndarray:
    a_vec, b_vec, c_vec = lattice_vectors(p)
    pts = []
    off = np.array(sub.offset_frac, float)
    for b in bravais_basis(sub.bravais):
        frac = np.array(b, float) + off
        f = np.mod(frac, 1.0)
        pts.append(frac_to_cart(f, a_vec, b_vec, c_vec))
    return np.asarray(pts)


# -------------------------------
# Cached geometry
# -------------------------------

def _make_key(sublattices: List[Sublattice], p: LatticeParams):
    subs_key = tuple((s.bravais, float(s.offset_frac[0]), float(s.offset_frac[1]), float(s.offset_frac[2]),
                      float(s.alpha_ratio), bool(s.visible)) for s in sublattices)
    p_key = (float(p.a), float(p.b_ratio), float(p.c_ratio), float(p.alpha), float(p.beta), float(p.gamma))
    return type("K", (), {"subs": subs_key, "p": p_key})

@lru_cache(maxsize=128)
def centers_alphas_and_shifts(subs_key, p_key):
    p = LatticeParams(*p_key)
    a_vec, b_vec, c_vec = lattice_vectors(p)
    shifts = []
    for i in (-1,0,1):
        for j in (-1,0,1):
            for k in (-1,0,1):
                shifts.append(frac_to_cart(np.array([i,j,k], float), a_vec, b_vec, c_vec))
    shifts = np.asarray(shifts, float)

    centers = []
    alphas = []
    sub_ids = []
    sub_idx = 0
    for (bravais, ox,oy,oz, alpha_ratio, visible) in subs_key:
        if not visible:
            sub_idx += 1
            continue
        sub = Sublattice("x", bravais, (ox,oy,oz), alpha_ratio, True)
        pts = generate_centers_minimal(sub, p)
        n = len(pts)
        centers.append(pts)
        alphas.append(np.full(n, alpha_ratio, float))
        sub_ids.append(np.full(n, sub_idx, int))
        sub_idx += 1

    if centers:
        centers = np.vstack(centers)
        alphas  = np.concatenate(alphas)
        sub_ids = np.concatenate(sub_ids)
    else:
        centers = np.empty((0,3)); alphas = np.empty((0,)); sub_ids = np.empty((0,), dtype=int)
    return centers, alphas, sub_ids, shifts


# ===== OPTIMIZATION 1: Vectorized shift loop =====
def _count_multiplicity_at_points_vectorized(points: np.ndarray,
                                             centers: np.ndarray,
                                             radii: np.ndarray,
                                             shifts: np.ndarray,
                                             tol_inside: float,
                                             early_stop_at: Optional[int]=None) -> np.ndarray:
    """
    OPTIMIZED: Stack all shifts at once, single distance pass
    Instead of: for each shift, compute distances to all centers
    Now: compute distances to (27*N_centers, 3) at once
    """
    if points.size == 0 or centers.size == 0:
        return np.zeros((0,), dtype=int) if points.size > 0 else np.zeros((0,), dtype=int)
    
    P = np.asarray(points, float)  # (N_samples, 3)
    counts = np.zeros(P.shape[0], dtype=int)
    
    # Stack all shifted centers once
    all_centers = []
    all_radii = []
    for S in shifts:
        all_centers.append(centers + S)  # (N_centers, 3)
        all_radii.append(radii)  # (N_centers,)
    
    all_centers = np.vstack(all_centers)  # (27*N_centers, 3)
    all_radii = np.concatenate(all_radii)  # (27*N_centers,)
    
    # Single vectorized distance pass
    N = len(all_centers)
    block = 4096
    for start in range(0, N, block):
        end = min(start + block, N)
        Cb = all_centers[start:end]
        Rb = all_radii[start:end]
        
        # Distances: (N_samples, block_size)
        d = np.linalg.norm(P[:, None, :] - Cb[None, :, :], axis=2)
        hits = np.abs(d - Rb[None, :]) <= tol_inside
        counts += hits.sum(axis=1)
        
        if early_stop_at is not None and np.any(counts >= early_stop_at):
            return counts
    
    return counts


# ===== OPTIMIZATION 2: Pre-computed ALL-PAIRS (no s_cut) =====
@dataclass(frozen=True)
class AllPairList:
    """Pre-computed all sphere pairs (no distance limit)"""
    pairs: List[Tuple[int, int, int, Tuple[float, float, float]]]
    centers: np.ndarray
    alphas: np.ndarray
    shifts: np.ndarray

def build_all_pairlist(sublattices: List[Sublattice], p: LatticeParams) -> AllPairList:
    """
    Pre-compute ALL pairs without s_cut limit.
    Later: filter by distance dynamically when scale_s is known.
    """
    key = _make_key(sublattices, p)
    centers, alphas, sub_ids, shifts = centers_alphas_and_shifts(key.subs, key.p)
    
    if centers.size == 0:
        return AllPairList([], centers, alphas, shifts)
    
    pairs: List[Tuple[int,int,int,Tuple[float,float,float]]] = []
    
    if _HAVE_SCIPY:
        # Build all shifted centers for KDTree
        img_centers = []
        img_index = []
        for si, S in enumerate(shifts):
            img_centers.append(centers + S)
            img_index.append(np.full(len(centers), si, int))
        img_centers = np.vstack(img_centers)
        img_index = np.concatenate(img_index)
        tree = KDTree(img_centers)
        
        # Find ALL pairs (very large radius to get everything)
        R_cut_max = 100.0  # large enough to get all potential pairs
        
        for i, c0 in enumerate(centers):
            idxs = tree.query_ball_point(c0, r=R_cut_max)
            for gidx in idxs:
                sidx = img_index[gidx]
                j = gidx % len(centers)
                if sidx == 13 and j <= i:  # avoid double-counting center cell
                    continue
                c1 = centers[j] + shifts[sidx]
                d_vec = c1 - c0
                pairs.append((i, j, sidx, (float(d_vec[0]), float(d_vec[1]), float(d_vec[2]))))
    else:
        # Fallback without KDTree
        for i, c0 in enumerate(centers):
            for sidx, S in enumerate(shifts):
                C = centers + S
                for j, c1 in enumerate(C):
                    if sidx == 13 and j <= i:
                        continue
                    d_vec = c1 - c0
                    pairs.append((i, j, sidx, (float(d_vec[0]), float(d_vec[1]), float(d_vec[2]))))
    
    return AllPairList(pairs, centers, alphas, shifts)


# Old pairlist for backward compat
@dataclass(frozen=True)
class PairList:
    pairs: List[Tuple[int,int,int,Tuple[float,float,float]]]
    centers: np.ndarray
    alphas: np.ndarray
    shifts: np.ndarray

def build_pairlist(sublattices: List[Sublattice], p: LatticeParams, s_cut: float) -> PairList:
    """Legacy function for backward compatibility"""
    key = _make_key(sublattices, p)
    centers, alphas, sub_ids, shifts = centers_alphas_and_shifts(key.subs, key.p)
    if centers.size == 0:
        return PairList([], centers, alphas, shifts)

    a_len = float(p.a)
    alpha_sum_max = float((alphas[:,None] + alphas[None,:]).max())
    R_cut = s_cut * a_len * alpha_sum_max + 1e-9

    pairs: List[Tuple[int,int,int,Tuple[float,float,float]]] = []
    if _HAVE_SCIPY:
        img_centers = []
        img_index   = []
        for si, S in enumerate(shifts):
            img_centers.append(centers + S)
            img_index.append(np.full(len(centers), si, int))
        img_centers = np.vstack(img_centers)
        img_index   = np.concatenate(img_index)
        tree = KDTree(img_centers)

        for i, c0 in enumerate(centers):
            idxs = tree.query_ball_point(c0, r=R_cut)
            for gidx in idxs:
                sidx = img_index[gidx]
                j    = gidx % len(centers)
                if sidx == 13 and j <= i:
                    continue
                c1 = centers[j] + shifts[sidx]
                d_vec = c1 - c0
                pairs.append((i, j, sidx, (float(d_vec[0]), float(d_vec[1]), float(d_vec[2]))))
    else:
        for i, c0 in enumerate(centers):
            for sidx, S in enumerate(shifts):
                C = centers + S
                for j, c1 in enumerate(C):
                    if sidx == 13 and j <= i: continue
                    d = np.linalg.norm(c1 - c0)
                    if d <= R_cut:
                        d_vec = c1 - c0
                        pairs.append((i, j, sidx, (float(d_vec[0]), float(d_vec[1]), float(d_vec[2]))))

    return PairList(pairs, centers, alphas, shifts)


# ===== OPTIMIZATION 3: Pair-circle sampling helpers =====

def _orthonormal_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, float)
    if abs(n[0]) < 0.9:
        t = np.array([1.0, 0.0, 0.0])
    else:
        t = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, t); u /= (np.linalg.norm(u) + 1e-18)
    v = np.cross(n, u); v /= (np.linalg.norm(v) + 1e-18)
    return u, v

def _sample_pair_circle(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float, k_samples: int) -> Optional[np.ndarray]:
    d_vec = c2 - c1
    d = np.linalg.norm(d_vec)
    if d < 1e-12: return None
    if d > (r1 + r2) or d < abs(r1 - r2): return None
    n = d_vec / d
    a = (r1*r1 - r2*r2 + d*d) / (2*d)
    rho2 = r1*r1 - a*a
    rho = math.sqrt(max(rho2, 0.0))
    center = c1 + a*n
    u, v = _orthonormal_basis(n)
    angles = np.linspace(0.0, 2.0*math.pi, int(k_samples), endpoint=False)
    pts = center[None,:] + rho*np.cos(angles)[:,None]*u[None,:] + rho*np.sin(angles)[:,None]*v[None,:]
    return pts

def _cluster_points(points: np.ndarray, values: np.ndarray, eps: float) -> Tuple[List[np.ndarray], List[int]]:
    if points.size == 0:
        return [], []
    pts = np.asarray(points, float)
    vals = np.asarray(values, int)
    used = np.zeros(len(pts), dtype=bool)
    clusters, cvals = [], []
    for i in range(len(pts)):
        if used[i]: continue
        center = pts[i].copy()
        best_val = vals[i]
        count = 1
        for j in range(i+1, len(pts)):
            if used[j]: continue
            if np.linalg.norm(pts[j] - center) <= eps:
                used[j] = True
                count += 1
                center += (pts[j] - center) / count
                if vals[j] > best_val:
                    best_val = vals[j]
        used[i] = True
        clusters.append(center)
        cvals.append(int(best_val))
    return clusters, cvals


# ===== MAIN EVALUATION FUNCTION (uses vectorized counting) =====

def max_multiplicity_for_scale(sublattices: List[Sublattice],
                               p: LatticeParams,
                               repeat_ignored: int,
                               scale_s: float,
                               k_samples: int = 12,
                               tol_inside: float = 1e-3,
                               cluster_eps: float = 0.1,
                               early_stop_at: Optional[int] = None,
                               pairlist: Optional[PairList] = None,
                               all_pairlist: Optional[AllPairList] = None
                               ) -> Tuple[int, List[np.ndarray], List[int]]:
    """
    Evaluate k_max at fixed s using pair-circle sampling + clustering.
    NEW: Supports all_pairlist (pre-computed all pairs, dynamically filtered)
    """
    if all_pairlist is not None:
        centers = all_pairlist.centers
        alphas = all_pairlist.alphas
        shifts = all_pairlist.shifts
        use_all_pairs = all_pairlist.pairs
    elif pairlist is not None:
        centers = pairlist.centers
        alphas = pairlist.alphas
        shifts = pairlist.shifts
        use_all_pairs = None
    else:
        key = _make_key(sublattices, p)
        centers, alphas, sub_ids, shifts = centers_alphas_and_shifts(key.subs, key.p)
        use_all_pairs = None

    if centers.size == 0:
        return 0, [], []

    radii = (alphas * scale_s * p.a).astype(float)

    sample_pts = []
    
    # If we have all-pairs, filter and sample
    if use_all_pairs is not None:
        for (i, j, sidx, dxyz) in use_all_pairs:
            c0 = centers[i]
            c1 = centers[j] + shifts[sidx]
            r1 = radii[i]
            r2 = radii[j]
            d = math.sqrt(dxyz[0]*dxyz[0] + dxyz[1]*dxyz[1] + dxyz[2]*dxyz[2])
            # Dynamic filter based on current scale_s
            if d > (r1 + r2) or d < abs(r1 - r2) or d < 1e-12:
                continue
            pts = _sample_pair_circle(c0, r1, c1, r2, k_samples)
            if pts is not None:
                sample_pts.append(pts)
    else:
        # Original logic (for backward compat with pairlist only)
        if _HAVE_SCIPY:
            img_centers = []
            img_index = []
            for si, S in enumerate(shifts):
                img_centers.append(centers + S)
                img_index.append(np.full(len(centers), si, int))
            img_centers = np.vstack(img_centers)
            img_index = np.concatenate(img_index)
            tree = KDTree(img_centers)
            rmax = float(radii.max(initial=0.0))
            for i, c0 in enumerate(centers):
                r1 = radii[i]
                idxs = tree.query_ball_point(c0, r=r1 + rmax + 1.0)
                for idx in idxs:
                    sidx = img_index[idx]
                    j = idx % len(centers)
                    if sidx == 13 and j <= i:
                        continue
                    c1 = centers[j] + shifts[sidx]
                    r2 = radii[j]
                    pts = _sample_pair_circle(c0, r1, c1, r2, k_samples)
                    if pts is not None:
                        sample_pts.append(pts)
        else:
            for i, c0 in enumerate(centers):
                r1 = radii[i]
                for sidx, S in enumerate(shifts):
                    C = centers + S
                    for j, c1 in enumerate(C):
                        if sidx == 13 and j <= i:
                            continue
                        r2 = radii[j]
                        d = np.linalg.norm(c1 - c0)
                        if d > (r1 + r2) or d < abs(r1 - r2) or d < 1e-12:
                            continue
                        pts = _sample_pair_circle(c0, r1, c1, r2, k_samples)
                        if pts is not None:
                            sample_pts.append(pts)

    if not sample_pts:
        return 0, [], []

    samples = np.vstack(sample_pts)
    # USE VECTORIZED COUNTING HERE
    counts = _count_multiplicity_at_points_vectorized(samples, centers, radii, shifts, tol_inside, early_stop_at)

    # Skip clustering if threshold hit
    if (early_stop_at is not None) and np.any(counts >= early_stop_at):
        return int(early_stop_at), [], []

    kmax = int(counts.max(initial=0))
    clusters, cvals = _cluster_points(samples, counts, cluster_eps)
    return kmax, clusters, cvals


def find_threshold_s_for_N(N_target: int,
                           sublattices: List[Sublattice],
                           p: LatticeParams,
                           repeat_ignored: int,
                           s_min: float,
                           s_max: float,
                           k_samples_coarse: int = 8,
                           k_samples_fine: int = 16,
                           tol_inside: float = 1e-3,
                           cluster_eps: float = 0.1,
                           max_iter: int = 30,
                           pairlist: Optional[PairList] = None,
                           all_pairlist: Optional[AllPairList] = None
                           ) -> Tuple[Optional[float], Dict[int, float]]:
    """Binary search with support for all_pairlist"""
    milestones: Dict[int, float] = {}
    lo, hi = float(s_min), float(s_max)

    km_lo, _, _ = max_multiplicity_for_scale(sublattices, p, 1, lo,
                                             k_samples=k_samples_coarse,
                                             tol_inside=tol_inside, cluster_eps=cluster_eps,
                                             early_stop_at=N_target, pairlist=pairlist,
                                             all_pairlist=all_pairlist)
    milestones[km_lo] = milestones.get(km_lo, lo)
    km_hi, _, _ = max_multiplicity_for_scale(sublattices, p, 1, hi,
                                             k_samples=k_samples_coarse,
                                             tol_inside=tol_inside, cluster_eps=cluster_eps,
                                             early_stop_at=N_target, pairlist=pairlist,
                                             all_pairlist=all_pairlist)
    milestones[km_hi] = milestones.get(km_hi, hi)
    if km_hi < N_target:
        return None, milestones

    s_star = None
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        km, _, _ = max_multiplicity_for_scale(sublattices, p, 1, mid,
                                              k_samples=k_samples_fine,
                                              tol_inside=tol_inside, cluster_eps=cluster_eps,
                                              early_stop_at=N_target, pairlist=pairlist,
                                              all_pairlist=all_pairlist)
        milestones[km] = milestones.get(km, mid)
        if km >= N_target:
            s_star = mid
            hi = mid
        else:
            lo = mid
        if abs(hi-lo) <= 1e-6:
            break
    return s_star, milestones


# CN (representative site)
def _geo_key_and_arrays(sublattices: List[Sublattice], p: LatticeParams):
    key = _make_key(sublattices, p)
    centers, alphas, sub_ids, shifts = centers_alphas_and_shifts(key.subs, key.p)
    return key, centers, alphas, sub_ids, shifts


def compute_cn_representative(
    sublattices: List[Sublattice],
    p: LatticeParams,
    sub_index: int,
    scale_s: float,
    k_filter: int = 6,
    mode: str = "exact",
    k_samples: int = 12,
    tol_surface: float = 1e-3,
    cluster_eps: Optional[float] = None,
) -> int:
    key, centers, alphas, sub_ids, shifts = _geo_key_and_arrays(sublattices, p)
    mask = (sub_ids == int(sub_index))
    if centers.size == 0 or not np.any(mask) or not sublattices[sub_index].visible:
        return 0

    rep_idx = int(np.flatnonzero(mask)[0])
    rep_radius = float(alphas[rep_idx] * scale_s * p.a)

    _, reps, repc = max_multiplicity_for_scale(
        sublattices, p, 1, scale_s, k_samples=k_samples,
        tol_inside=tol_surface,
        cluster_eps=cluster_eps if cluster_eps is not None else 0.1 * p.a,
        early_stop_at=None
    )
    if len(reps) == 0:
        return 0
    reps = np.asarray(reps); repc = np.asarray(repc, int)

    if mode == "exact":
        pick = np.where(repc == int(k_filter))[0]
    else:
        pick = np.where(repc >= int(k_filter))[0]
    if pick.size == 0:
        return 0
    pts = reps[pick]

    if _HAVE_SCIPY:
        tree = KDTree(centers.copy())
        cn = 0
        for pnt in pts:
            best_d = float("inf"); best_idx = -1
            for S in shifts:
                d, idx = tree.query(pnt - S, k=1)
                if d < best_d:
                    best_d = d; best_idx = idx
            if (best_idx == rep_idx) and (abs(best_d - rep_radius) <= tol_surface):
                cn += 1
        return cn
    else:
        all_centers = np.vstack([centers + S for S in shifts])
        d = np.linalg.norm(pts[:, None, :] - all_centers[None, :, :], axis=2)
        nn = np.argmin(d, axis=1)
        dmin = d[np.arange(len(pts)), nn]
        N = len(centers)
        idx = (nn % N)
        ok = (idx == rep_idx) & (np.abs(dmin - rep_radius) <= tol_surface)
        return int(np.count_nonzero(ok))
