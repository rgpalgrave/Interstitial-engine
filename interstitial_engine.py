# =====================================================
# interstitial_engine.py
# Geometry + fast pair-circle engine for k_max and s*_N
# (with caching + neighbor buckets + early-stop)
# =====================================================

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Tuple, Optional, Dict, List
import numpy as np
from math import sqrt


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
        "cubic_P":        [(0, 0, 0)],
        "cubic_I":        [(0, 0, 0), (0.5, 0.5, 0.5)],
        "cubic_F":        [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
        # Compound motifs on cubic
        "cubic_Diamond": [
            (0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),
            (0.25, 0.25, 0.25), (0.75, 0.75, 0.25), (0.75, 0.25, 0.75), (0.25, 0.75, 0.75)
        ],
        "cubic_Pyrochlore": [
            (0.5, 0.5, 0.5), (0, 0, 0.5), (0, 0.5, 0), (0.5, 0, 0),
            (0.25, 0.75, 0), (0.75, 0.25, 0), (0.75, 0.75, 0.5), (0.25, 0.25, 0.5),
            (0.75, 0, 0.25), (0.25, 0.5, 0.25), (0.25, 0, 0.75), (0.75, 0.5, 0.75),
            (0, 0.25, 0.75), (0.5, 0.75, 0.75), (0.5, 0.25, 0.25), (0, 0.75, 0.25)
        ],
        # Tetragonal
        "tetragonal_P":   [(0, 0, 0)],
        "tetragonal_I":   [(0, 0, 0), (0.5, 0.5, 0.5)],
        # Orthorhombic
        "orthorhombic_P": [(0, 0, 0)],
        "orthorhombic_C": [(0, 0, 0), (0.5, 0.5, 0)],
        "orthorhombic_I": [(0, 0, 0), (0.5, 0.5, 0.5)],
        "orthorhombic_F": [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
        # Hexagonal / HCP
        "hexagonal_P":    [(0, 0, 0)],
        "hexagonal_HCP":  [(0, 0, 0), (1/3, 2/3, 0.5)],
        # Rhombohedral
        "rhombohedral_R": [(0, 0, 0)],
        # Monoclinic / Triclinic
        "monoclinic_P":   [(0, 0, 0)],
        "monoclinic_C":   [(0, 0, 0), (0.5, 0.5, 0)],
        "triclinic_P":    [(0, 0, 0)],
    }
    if bravais not in B:
        raise ValueError(f"Unknown bravais type: {bravais}")
    return np.asarray(B[bravais], dtype=float)


def lattice_vectors(p: LatticeParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a, b, c vectors. Orthogonal fast path; general (α,β,γ) fallback."""
    a = p.a
    b = p.a * p.b_ratio
    c = p.a * p.c_ratio
    if abs(p.alpha - 90) < 1e-9 and abs(p.beta - 90) < 1e-9 and abs(p.gamma - 90) < 1e-9:
        return np.array([a, 0, 0.0]), np.array([0.0, b, 0.0]), np.array([0.0, 0.0, c])

    # General cell
    alp = np.deg2rad(p.alpha)
    bet = np.deg2rad(p.beta)
    gam = np.deg2rad(p.gamma)
    avec = np.array([a, 0, 0], float)
    bvec = np.array([b * np.cos(gam), b * np.sin(gam), 0.0], float)
    cx = c * np.cos(bet)
    cy = c * (np.cos(alp) - np.cos(bet) * np.cos(gam)) / max(1e-12, np.sin(gam))
    cz2 = max(0.0, c * c - cx * cx - cy * cy)
    cvec = np.array([cx, cy, sqrt(cz2)], float)
    return avec, bvec, cvec


def frac_to_cart(frac: np.ndarray, avec: np.ndarray, bvec: np.ndarray, cvec: np.ndarray) -> np.ndarray:
    return frac[0] * avec + frac[1] * bvec + frac[2] * cvec


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
    d2 = float(np.dot(v, v))
    if d2 <= 1e-24:
        return np.empty((0, 3))
    d = sqrt(d2)
    # overlap condition (not contained, not disjoint)
    if not (abs(r1 - r2) < d < (r1 + r2)):
        return np.empty((0, 3))
    n = v / d
    t = (r1 * r1 - r2 * r2 + d * d) / (2 * d * d)
    center = c1 + t * v

    # in-plane basis
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) <= 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(tmp, n)
    nu = float(np.linalg.norm(u))
    if nu < 1e-12:
        tmp = np.array([0.0, 0.0, 1.0])
        u = np.cross(tmp, n)
        nu = float(np.linalg.norm(u))
        if nu < 1e-12:
            return np.empty((0, 3))
    u /= nu
    w = np.cross(n, u)

    # circle radius (h) from r1 and offset
    off = center - c1
    h2 = r1 * r1 - float(np.dot(off, off))
    if h2 <= 0:
        return np.empty((0, 3))
    h = sqrt(h2)

    # sample points
    angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    return center + (h * np.cos(angles))[:, None] * u + (h * np.sin(angles))[:, None] * w


def _cluster(points: np.ndarray, counts: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy radius clustering; small N so this is fine."""
    if len(points) == 0:
        return points, counts
    used = np.zeros(len(points), dtype=bool)
    reps = []
    repc = []
    eps2 = eps * eps
    for i in range(len(points)):
        if used[i]:
            continue
        acc = [i]
        for j in range(i + 1, len(points)):
            if used[j]:
                continue
            d2 = float(np.dot(points[i] - points[j], points[i] - points[j]))
            if d2 < eps2:
                acc.append(j)
        used[acc] = True
        reps.append(np.mean(points[acc], axis=0))
        repc.append(int(np.max(counts[acc])))
    return np.asarray(reps), np.asarray(repc, dtype=int)


# -----------------
# Geometry cache
# -----------------
@dataclass(frozen=True)
class GeoKey:
    p: Tuple[float, float, float, float, float, float]
    repeat: int
    subs: Tuple[Tuple[str, float, float, float, float, bool], ...]  # (bravais, offx, offy, offz, alpha_ratio, visible)


def _make_key(sublattices: List[Sublattice], p: LatticeParams, repeat: int) -> GeoKey:
    subs_key = tuple((s.bravais, float(s.offset_frac[0]), float(s.offset_frac[1]), float(s.offset_frac[2]),
                      float(s.alpha_ratio), bool(s.visible)) for s in sublattices)
    p_key = (float(p.a), float(p.b_ratio), float(p.c_ratio), float(p.alpha), float(p.beta), float(p.gamma))
    return GeoKey(p=p_key, repeat=int(repeat), subs=subs_key)


@lru_cache(maxsize=128)
def build_geometry_cache(key: GeoKey):
    """Build centers, per-sphere α multipliers, and a static spatial hash buckets."""
    p = LatticeParams(*key.p)
    centers_list = []
    alpha_list = []
    for bravais, offx, offy, offz, alpha_ratio, visible in key.subs:
        if not visible:
            continue
        sub = Sublattice(name="x", bravais=bravais, offset_frac=(offx, offy, offz), alpha_ratio=alpha_ratio, visible=True)
        pts = generate_points(sub, p, repeat=key.repeat, include_ghost=True)
        centers_list.append(pts)
        alpha_list.append(np.full(len(pts), alpha_ratio, dtype=float))

    if not centers_list:
        return (np.empty((0, 3)), np.empty((0,)), None)

    centers = np.vstack(centers_list).astype(float)
    alphas = np.concatenate(alpha_list).astype(float)

    # Build static spatial hash with a modest cell size (linked to a)
    L = p.a * 0.9
    mins = centers.min(axis=0) - 1e-9
    idx = np.floor((centers - mins) / L).astype(int)
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, keyc in enumerate(map(tuple, idx)):
        buckets[keyc].append(i)

    # Pre-store neighbor offset list
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    return centers, alphas, (mins, L, buckets, offs)


def _pairs_within_cutoff(centers: np.ndarray, grid, cutoff: float) -> np.ndarray:
    """Return i<j pairs within given cutoff using cached buckets."""
    if centers.size == 0:
        return np.empty((0, 2), dtype=int)
    mins, L, buckets, offs = grid
    cutoff2 = cutoff * cutoff
    pairs: List[Tuple[int, int]] = []
    # Iterate buckets; compare within own+neighbor buckets
    for keyc, ids in buckets.items():
        nbr = []
        for o in offs:
            k2 = (keyc[0] + o[0], keyc[1] + o[1], keyc[2] + o[2])
            if k2 in buckets:
                nbr += buckets[k2]
        if not nbr:
            continue
        nbr_arr = np.array(sorted(set(nbr)), dtype=int)
        ids_arr = np.array(ids, dtype=int)
        for i in ids_arr:
            js = nbr_arr[nbr_arr > i]
            if js.size == 0:
                continue
            d2 = np.sum((centers[js] - centers[i]) ** 2, axis=1)
            within = js[d2 < cutoff2]
            if within.size:
                pairs.extend([(i, int(j)) for j in within])
    if not pairs:
        return np.empty((0, 2), dtype=int)
    return np.asarray(pairs, dtype=int)


# -----------------
# Public API
# -----------------
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
    """Compute k_max at this s, returning (k_max, hotspots_xyz, hotspots_counts)."""
    key = _make_key(sublattices, p, repeat)
    centers, alphas, grid = build_geometry_cache(key)
    if len(centers) == 0:
        return 0, np.empty((0, 3)), np.empty((0,))

    radii = (alphas * scale_s * p.a).astype(float)
    rmax = float(np.max(radii))
    # Candidate pairs only within 2*rmax
    if grid is None:
        return 0, np.empty((0, 3)), np.empty((0,))
    pairs = _pairs_within_cutoff(centers, grid, cutoff=2.0 * rmax)
    if len(pairs) == 0:
        return 0, np.empty((0, 3)), np.empty((0,))

    samples: List[np.ndarray] = []
    counts: List[int] = []
    tol2 = (tol_inside) ** 2  # used in exact-equality mode; we use (d <= r+tol) below

    # Loop candidate pairs → sample circle → count multiplicity (vectorized)
    for i, j in pairs:
        pts = pair_circle_samples(centers[i], radii[i], centers[j], radii[j], k=k_samples)
        if pts.size == 0:
            continue
        # distances: [k, n]
        d = np.linalg.norm(centers[None, :, :] - pts[:, None, :], axis=2)
        c = np.sum(d <= (radii[None, :] + tol_inside), axis=1)
        if early_stop_at is not None and np.any(c >= early_stop_at):
            # collect a couple of hits to avoid empty return
            keep = np.where(c >= 2)[0][:3]
            samples.extend(pts[keep])
            counts.extend([int(x) for x in c[keep]])
            if cluster_eps is None:
                cluster_eps = 0.1 * p.a
            reps, repc = _cluster(np.asarray(samples), np.asarray(counts, dtype=int), eps=cluster_eps)
            return int(np.max(c)), reps, repc

        good = np.where(c >= 2)[0]
        if good.size:
            samples.extend(pts[good])
            counts.extend([int(x) for x in c[good]])

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
    k_samples_coarse: int = 6,
    k_samples_fine: int = 12,
    tol_inside: float = 1e-3,
    cluster_eps: Optional[float] = None,
    max_iter: int = 28,
) -> Tuple[Optional[float], Dict[int, float]]:
    """Return (s*_N, milestones). Uses coarse→fine sampling and aggressive early-stop."""
    milestones: Dict[int, float] = {}
    # Coarse sweep to bracket
    grid = np.linspace(s_min, s_max, 16)
    hit = False
    for s in grid:
        m, _, _ = max_multiplicity_for_scale(
            sublattices, p, repeat, s,
            k_samples=k_samples_coarse,
            tol_inside=tol_inside,
            cluster_eps=cluster_eps,
            early_stop_at=N_target
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
            sublattices, p, repeat, s2,
            k_samples=k_samples_coarse,
            tol_inside=tol_inside,
            cluster_eps=cluster_eps,
            early_stop_at=N_target
        )
        if m < N_target:
            s_lo = s2
        else:
            s_hi = s2
            break

    # bisection (adaptive samples: coarse while wide, fine near convergence)
    for _ in range(max_iter):
        mid = 0.5 * (s_lo + s_hi)
        ks = k_samples_coarse if (s_hi - s_lo) > 0.02 else k_samples_fine
        m, _, _ = max_multiplicity_for_scale(
            sublattices, p, repeat, mid,
            k_samples=ks,
            tol_inside=tol_inside,
            cluster_eps=cluster_eps,
            early_stop_at=N_target
        )
        if m >= N_target:
            s_hi = mid
        else:
            s_lo = mid
        if abs(s_hi - s_lo) < 1e-4:
            break
    milestones[N_target] = s_hi
    return s_hi, milestones
