"""Pure functions computing per-candidate ego-vehicle interaction descriptors.

Everything here is numpy + shapely only (no torch, no nuplan), so it can be
unit-tested without the NAVSIM stack and reused both at labeling time and at
deployment time.

Conventions
-----------
- All polygons are shapely polygons in the *global* frame (the frame used by
  ``PDMScorer`` / ``PDMObservation``), one per timestep, ``dt`` seconds apart.
- ``ego_polys`` has length ``T+1`` (t=0 is the current state).
- ``j_polys`` has length ``T+1``; entries may be ``None`` where vehicle ``j``
  is absent from the observation at that timestep.
- Headings are in radians, ``j_heading`` is measured at t=0 (deployment
  available). The descriptor therefore approximates "relative heading at first
  conflict step" with the t=0 relative heading (the conflict step's heading is
  not stored in the metric cache observation).
- Fields that are undefined are returned as ``NaN``; the boolean mask flags
  below make the reason explicit.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

# interaction type codes (stored as float in the descriptor array)
ITYPE_NONE = 0
ITYPE_SAME_DIR = 1
ITYPE_CROSSING = 2
ITYPE_ONCOMING = 3

ITYPE_NAMES = {
    ITYPE_NONE: "NONE",
    ITYPE_SAME_DIR: "SAME_DIR",
    ITYPE_CROSSING: "CROSSING",
    ITYPE_ONCOMING: "ONCOMING",
}

SAME_DIR_THRESH_RAD = np.deg2rad(30.0)
ONCOMING_THRESH_RAD = np.deg2rad(150.0)

# Field order of the descriptor array written by label_candidates.py.
# Every field is float32; boolean flags are stored as 0.0/1.0.
DESCRIPTOR_FIELDS: List[str] = [
    # GT attribution flags (TRAINING-SIDE labels; NOT deployment-available —
    # they come from the PDM scorer's collided-track lists).
    "att_collision",      # j newly in proposal_fault_collided_track_ids[k]
    "att_ttc",            # j newly in ttc_collided_track_ids[k]
    # swept-volume occupancy
    "conflict",           # any(ego_occ) and any(j_occ)
    "t_k_in",             # first t [s] with ego_occ   (NaN if none)
    "t_k_out",            # last  t [s] with ego_occ   (NaN if none)
    "t_j_in",             # first t [s] with j_occ     (NaN if none)
    "t_j_out",            # last  t [s] with j_occ     (NaN if none)
    "dt_enter",           # t_k_in - t_j_in [s]; negative = ego enters first (NaN if no conflict)
    "overlap",            # occupancy intervals intersect
    "pet",                # max(t_k_in,t_j_in) - min(t_k_out,t_j_out) [s] if no overlap else 0
    "k_in_censored",      # ego_occ already true at t=0
    "k_out_censored",     # ego_occ still true at last timestep
    "j_in_censored",      # j_occ already true at t=0
    "j_out_censored",     # j_occ still true at last timestep
    "multi_entry",        # ego_occ or j_occ has >1 contiguous segment
    "min_dist",           # min_t ego_poly[t].distance(j_poly[t]) [m] (NaN if never co-present)
    "itype",              # ITYPE_* code (NONE if no conflict)
    # vehicle j state at t=0 in ego frame (deployment-available)
    "rel_x",              # longitudinal offset of j center wrt ego [m]
    "rel_y",              # lateral offset of j center wrt ego [m]
    "rel_heading",        # wrap_to_pi(j_heading - ego_heading) [rad]
    "speed",              # |v_j| at t=0 [m/s]
]

DESCRIPTOR_FIELD_INDEX: Dict[str, int] = {f: i for i, f in enumerate(DESCRIPTOR_FIELDS)}
NUM_DESCRIPTOR_FIELDS = len(DESCRIPTOR_FIELDS)

# Fields used for the kNN / parametric descriptor predictors in
# oracle_knn_check.py (itype is appended separately as one-hot).
# att_collision/att_ttc are deliberately NOT features: they are GT labels
# (NC<1 is essentially "some vehicle newly at-fault collided"), so feeding
# them in would leak the target into the oracle check.
KNN_NUMERIC_FIELDS: List[str] = [
    "conflict",
    "dt_enter",
    "pet",
    "min_dist",
    "rel_x",
    "rel_y",
    "rel_heading",
    "speed",
    "overlap",
    "k_in_censored",
    "k_out_censored",
    "j_in_censored",
    "j_out_censored",
    "multi_entry",
]

# Leakage-safe default feature set (--feature_set timing): additionally drops
# min_dist and overlap, which nearly encode the collision label themselves
# (a collision implies min_dist~0 and overlapping occupancy intervals).
TIMING_FIELDS: List[str] = [
    "conflict",
    "dt_enter",
    "pet",
    "k_in_censored",
    "k_out_censored",
    "j_in_censored",
    "j_out_censored",
    "multi_entry",
    "rel_x",
    "rel_y",
    "rel_heading",
    "speed",
]


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _contiguous_runs(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    """Return list of (start, end) inclusive index ranges of True-runs."""
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def compute_interaction_descriptor(
    ego_polys: Sequence[Polygon],
    j_polys: Sequence[Optional[Polygon]],
    ego_heading: float,
    j_heading: float,
    j_speed: float = np.nan,
    dt: float = 0.1,
    att_collision: bool = False,
    att_ttc: bool = False,
) -> Dict[str, float]:
    """Compute the ego-vehicle interaction descriptor for one (candidate, vehicle) pair.

    :param ego_polys: T+1 shapely polygons of the ego footprint (global frame).
    :param j_polys: T+1 shapely polygons or None for vehicle j (global frame).
    :param ego_heading: ego heading at t=0 [rad].
    :param j_heading: vehicle j heading at t=0 [rad].
    :param j_speed: vehicle j speed at t=0 [m/s] (NaN if unknown).
    :param dt: timestep [s].
    :param att_collision: j was newly attributed an at-fault collision for this candidate.
    :param att_ttc: j was newly attributed a TTC infraction for this candidate.
    :return: dict mapping every field of DESCRIPTOR_FIELDS to a float.
    """
    num_steps = len(ego_polys)
    assert len(j_polys) == num_steps

    desc: Dict[str, float] = {f: np.nan for f in DESCRIPTOR_FIELDS}
    desc["att_collision"] = float(att_collision)
    desc["att_ttc"] = float(att_ttc)

    # ---------------- swept volumes ----------------
    present_j = [p for p in j_polys if p is not None]
    swept_ego = unary_union(list(ego_polys))
    swept_j = unary_union(present_j) if present_j else None

    ego_occ = np.zeros(num_steps, dtype=bool)
    j_occ = np.zeros(num_steps, dtype=bool)
    if swept_j is not None:
        ego_occ = np.array([p.intersects(swept_j) for p in ego_polys], dtype=bool)
        j_occ = np.array(
            [p is not None and p.intersects(swept_ego) for p in j_polys], dtype=bool
        )

    ego_runs = _contiguous_runs(ego_occ)
    j_runs = _contiguous_runs(j_occ)

    conflict = bool(ego_occ.any() and j_occ.any())
    desc["conflict"] = float(conflict)

    # ---------------- occupancy intervals ----------------
    t = np.arange(num_steps, dtype=np.float64) * dt
    t_k_in = t_j_in = t_k_out = t_j_out = np.nan
    if ego_occ.any():
        t_k_in = float(t[ego_occ][0])
        t_k_out = float(t[ego_occ][-1])
        desc["t_k_in"] = t_k_in
        desc["t_k_out"] = t_k_out
    if j_occ.any():
        t_j_in = float(t[j_occ][0])
        t_j_out = float(t[j_occ][-1])
        desc["t_j_in"] = t_j_in
        desc["t_j_out"] = t_j_out

    if conflict:
        dt_enter = t_k_in - t_j_in
        overlap = max(t_k_in, t_j_in) <= min(t_k_out, t_j_out)
        desc["dt_enter"] = float(dt_enter)
        desc["overlap"] = float(overlap)
        desc["pet"] = 0.0 if overlap else float(max(t_k_in, t_j_in) - min(t_k_out, t_j_out))

    desc["k_in_censored"] = float(bool(ego_occ[0]))
    desc["k_out_censored"] = float(bool(ego_occ[-1]))
    desc["j_in_censored"] = float(bool(j_occ[0]))
    desc["j_out_censored"] = float(bool(j_occ[-1]))
    desc["multi_entry"] = float(len(ego_runs) > 1 or len(j_runs) > 1)

    # ---------------- minimum distance ----------------
    dists = [
        e.distance(j) for e, j in zip(ego_polys, j_polys) if (e is not None and j is not None)
    ]
    desc["min_dist"] = float(np.min(dists)) if dists else np.nan

    # ---------------- interaction type ----------------
    dpsi = wrap_angle(j_heading - ego_heading)
    if not conflict:
        desc["itype"] = float(ITYPE_NONE)
    elif abs(dpsi) < SAME_DIR_THRESH_RAD:
        desc["itype"] = float(ITYPE_SAME_DIR)
    elif abs(dpsi) > ONCOMING_THRESH_RAD:
        desc["itype"] = float(ITYPE_ONCOMING)
    else:
        desc["itype"] = float(ITYPE_CROSSING)

    # ---------------- j state at t=0 in ego frame (deployment-available) ----------------
    j0 = j_polys[0]
    e0 = ego_polys[0]
    if j0 is not None:
        dx = j0.centroid.x - e0.centroid.x
        dy = j0.centroid.y - e0.centroid.y
        c, s = np.cos(ego_heading), np.sin(ego_heading)
        desc["rel_x"] = float(c * dx + s * dy)
        desc["rel_y"] = float(-s * dx + c * dy)
        desc["rel_heading"] = dpsi
        desc["speed"] = float(j_speed)

    return desc


def descriptor_sort_key(desc: Dict[str, float], use_attribution: bool = True) -> Tuple:
    """Ordering key for ranking vehicles of one candidate.

    With attribution (TRAINING-SIDE labels only): attributed collision >
    attributed TTC > conflict with smallest |dt_enter| > smallest min_dist >
    |rel_x| (deterministic tiebreak).
    Without attribution (deployment-feasible): conflict with smallest
    |dt_enter| > smallest min_dist > |rel_x|.
    """
    dt_enter = desc["dt_enter"]
    min_dist = desc["min_dist"]
    key = (
        -desc["conflict"],
        abs(dt_enter) if np.isfinite(dt_enter) else np.inf,
        min_dist if np.isfinite(min_dist) else np.inf,
        abs(desc["rel_x"]) if np.isfinite(desc["rel_x"]) else np.inf,
    )
    if use_attribution:
        return (-desc["att_collision"], -desc["att_ttc"]) + key
    return key


def select_top_m_vehicles(
    descs: List[Dict[str, float]], m: int, use_attribution: bool = True
) -> List[int]:
    """Return indices of the top-m vehicles for one candidate, sorted by priority.

    Index 0 of the returned list is the "main vehicle".

    :param use_attribution: True ranks by GT attribution flags first
        (att_collision / att_ttc) — a TRAINING-SIDE label; deployment must not
        use GT attribution to choose vehicles. False uses only
        deployment-available quantities (conflict / timing / distance).
    """
    order = sorted(
        range(len(descs)), key=lambda i: descriptor_sort_key(descs[i], use_attribution)
    )
    return order[:m]


def descriptor_feature_vector(
    desc: Dict[str, float],
    ego_speed: float = np.nan,
    fields: Optional[List[str]] = None,
) -> np.ndarray:
    """Flatten a descriptor (+ ego speed) to a fixed-size float32 feature vector.

    Layout: `fields` (default KNN_NUMERIC_FIELDS) + ego_speed + itype one-hot
    (same/cross/oncoming). Pass TIMING_FIELDS for the leakage-safe default.
    NaNs are preserved; callers standardize then fill NaN with 0 (mean impute).
    """
    fields = KNN_NUMERIC_FIELDS if fields is None else fields
    vec = [desc[f] for f in fields]
    vec.append(ego_speed)
    itype = int(desc["itype"]) if np.isfinite(desc["itype"]) else ITYPE_NONE
    vec.extend([
        float(itype == ITYPE_SAME_DIR),
        float(itype == ITYPE_CROSSING),
        float(itype == ITYPE_ONCOMING),
    ])
    return np.asarray(vec, dtype=np.float32)


def descriptor_feature_names(fields: Optional[List[str]] = None) -> List[str]:
    """Names matching descriptor_feature_vector layout (for schema/debug)."""
    fields = KNN_NUMERIC_FIELDS if fields is None else fields
    return list(fields) + ["ego_speed", "itype_same_dir", "itype_crossing", "itype_oncoming"]
