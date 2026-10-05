#!/usr/bin/env python
"""A2 - Label exported candidates with train-PDMScorer sub-scores and
vehicle-interaction descriptors (CPU, multiprocess).

For every exported scene record (<token>.npz from export_candidates.py) this
script reproduces the training-side scoring of
``score_module/compute_navsim_score.py::get_sub_score`` *without* requiring the
``anchors_scores_index`` file, and additionally stores:

    subscores        (32, 6)   float32  [NC, DAC, EP, TTC, Comfort, final]
    descriptors      (32, M, F) float32  per-candidate top-M vehicle descriptors
                     (field order: descriptors.DESCRIPTOR_FIELDS, see schema.json)
                     slots ordered by the GT-attribution ranking
    main_desc_noatt  (32, F)   float32  descriptor of the top-1 vehicle under
                     the attribution-free ranking (conflict > |dt_enter| >
                     min_dist > |rel_x|) — the leakage-safe main vehicle
    main_vehicle_token_noatt (32,) str  token of main_desc_noatt ('' if none)
    vehicle_mask     (32, M)   bool     slot is a real vehicle
    main_vehicle_token (32,)   str      token of descriptors[k,0] ('' if none)
    att_fault_tokens (32,)     str      ';'-joined newly at-fault collided tokens
    att_ttc_tokens   (32,)     str      ';'-joined newly TTC-attributed tokens
    fault_vehicle_flag (32,)   bool     any newly fault-collided object is VEHICLE
    ade_to_human     (32,)     float32  mean L2 to human GT xy (NaN if no GT)
    ego_speed        ()        float32

Usage (navtrain, matches model training labels):
    python scripts/experience/label_candidates.py \
        --export_dir $NAVSIM_EXP_ROOT/experience/navtrain_export \
        --metric_cache_dir $NAVSIM_EXP_ROOT/Drive-JEPA-cache/train_metric_cache \
        --out_dir $NAVSIM_EXP_ROOT/experience/navtrain_labels --workers 16

navtest official cache ($NAVSIM_EXP_ROOT/Drive-JEPA-cache/metric_cache) is also
supported: it stores no pdm_progress, so the reference progress is recomputed
by simulating the cached PDM trajectory (see README).
"""

import argparse
import json
import lzma
import os
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from shapely.ops import unary_union

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))

from navsim.agents.drive_jepa_perception_based.experience.records import (  # noqa: E402
    ade_to_human,
    load_npz,
    save_npz,
)
from navsim.agents.drive_jepa_perception_based.experience.descriptors import (  # noqa: E402
    ACLASS_BICYCLE,
    ACLASS_NAMES,
    ACLASS_PEDESTRIAN,
    ACLASS_VEHICLE,
    DESCRIPTOR_FIELDS,
    DESCRIPTOR_FIELDS_EXT,
    NUM_DESCRIPTOR_FIELDS,
    ITYPE_NAMES,
    compute_interaction_descriptor,
    select_top_m_vehicles,
    descriptor_feature_names,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export_dir", required=True)
    p.add_argument("--metric_cache_dir", required=True,
                   help="dir containing metadata/*.csv with metric cache paths")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max_scenes", type=int, default=None)
    p.add_argument("--token_list", type=str, default=None)
    p.add_argument("--cache_type", choices=["auto", "train", "official"], default="auto",
                   help="metric cache flavour: 'train' = navtrain train_metric_cache "
                        "(stores pdm_progress), 'official' = navtest metric_cache "
                        "(recomputes reference progress); 'auto' infers it from "
                        "the presence of pdm_progress")
    p.add_argument("--top_m", type=int, default=4)
    p.add_argument("--prefilter_dist", type=float, default=50.0,
                   help="only describe vehicles within this distance [m] of the "
                        "candidate's swept ego bounding box")
    p.add_argument("--include_vru", action="store_true",
                   help="additionally write multi_* arrays covering VEHICLE + "
                        "PEDESTRIAN + BICYCLE tracks with agent_class and the "
                        "`emerging` flag (t0-absent, later-conflicting). "
                        "Vehicle-only outputs stay identical.")
    p.add_argument("--top_m_vru", type=int, default=6,
                   help="top-m slots of the multi_* descriptor array")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Per-scene worker (module level for multiprocessing pickling)
# --------------------------------------------------------------------------- #

_METRIC_CACHE_FILE = "metric_cache.pkl"


def _metric_cache_map(metric_cache_dir: Path) -> Dict[str, str]:
    """token -> metric_cache.pkl path by scanning the local cache tree.

    Layout is ``<cache_dir>/<log_token>/<agent>/<token>/metric_cache.pkl``;
    the token is the scene directory name. MetricCacheLoader's metadata csv
    stores absolute paths baked at generation time, which are stale on any
    other machine, so we glob the real files instead.
    """
    mapping = {}
    for p in Path(metric_cache_dir).glob(f"*/*/*/{_METRIC_CACHE_FILE}"):
        mapping[p.parent.name] = str(p)
    if not mapping:
        raise FileNotFoundError(
            f"no */*/*/{_METRIC_CACHE_FILE} under {metric_cache_dir}"
        )
    return mapping


def _load_metric_cache(path: str):
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def _pdm_reference_progress(metric_cache, cns) -> np.ndarray:
    """Recompute the PDM reference progress for the *official* metric cache.

    Mirrors what the train cache stores as ``metric_cache.pdm_progress``
    (raw progress of the PDM-closed trajectory times its multiplicative score)
    and what the official evaluator normalizes against per proposal.
    """
    from navsim.evaluate.pdm_score import get_trajectory_as_array

    initial_ego_state = metric_cache.ego_state
    pdm_states = get_trajectory_as_array(
        metric_cache.trajectory, cns.proposal_sampling, initial_ego_state.time_point
    )
    sim = cns.simulator.simulate_proposals(pdm_states[None], initial_ego_state)
    cns.scorer.score_proposals(
        sim,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        np.zeros(1, dtype=np.float64),
    )
    return cns.scorer._progress_raw * cns.scorer._multi_metrics.prod(axis=0)


def label_scene(
    record_path: str,
    metric_cache_path: str,
    out_path: str,
    top_m: int,
    prefilter_dist: float,
    cache_type: str = "auto",
    include_vru: bool = False,
    top_m_vru: int = 6,
) -> str:
    """Label one exported scene; returns out_path."""
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

    from navsim.agents.drive_jepa_perception_based.score_module import (
        compute_navsim_score as cns,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        MultiMetricIndex,
        WeightedMetricIndex,
    )

    rec = load_npz(Path(record_path))
    proposals = np.asarray(rec["proposals"], dtype=np.float64)  # (K, 8, 3)
    num_cand = proposals.shape[0]
    metric_cache = _load_metric_cache(metric_cache_path)

    # ---- simulate + score exactly like get_sub_score -------------------------
    simulated_states = cns.before_score(metric_cache, proposals)  # (K, 41, 11)
    if cache_type == "train":
        pdm_progress = metric_cache.pdm_progress
    elif cache_type == "official":
        pdm_progress = _pdm_reference_progress(metric_cache, cns)
    else:
        pdm_progress = getattr(metric_cache, "pdm_progress", None)
        if pdm_progress is None:
            # official navtest metric cache: recompute the reference progress
            pdm_progress = _pdm_reference_progress(metric_cache, cns)

    final_scores = cns.scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        pdm_progress,
    )
    scorer = cns.scorer

    subscores = np.stack(
        [
            scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, :],
            scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, :],
            scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, :],
            scorer._weighted_metrics[WeightedMetricIndex.TTC, :],
            scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, :],
            final_scores,
        ],
        axis=-1,
    ).astype(np.float32)

    # ---- newly attributed collision / ttc tokens -----------------------------
    pre_collided = set(metric_cache.observation.collided_track_ids)
    att_fault: List[List[str]] = [
        [t for t in scorer.proposal_fault_collided_track_ids[k] if t not in pre_collided]
        for k in range(num_cand)
    ]
    att_ttc: List[List[str]] = [
        [t for t in scorer.ttc_collided_track_ids[k] if t not in pre_collided]
        for k in range(num_cand)
    ]

    observation = metric_cache.observation
    unique_objects = observation.unique_objects
    num_steps = simulated_states.shape[1]  # 41
    dt = cns.proposal_sampling.interval_length
    ego_heading = float(metric_cache.ego_state.rear_axle.heading)

    vehicle_tokens = [
        tok
        for tok, obj in unique_objects.items()
        if obj.tracked_object_type == TrackedObjectType.VEHICLE
    ]

    multi_tokens: List[str] = []
    if include_vru:
        aclass_of = {
            TrackedObjectType.VEHICLE: ACLASS_VEHICLE,
            TrackedObjectType.PEDESTRIAN: ACLASS_PEDESTRIAN,
            TrackedObjectType.BICYCLE: ACLASS_BICYCLE,
        }
        multi_tokens = [
            tok for tok, obj in unique_objects.items()
            if obj.tracked_object_type in aclass_of
        ]

    descriptors = np.full((num_cand, top_m, NUM_DESCRIPTOR_FIELDS), np.nan, dtype=np.float32)
    vehicle_mask = np.zeros((num_cand, top_m), dtype=bool)
    main_vehicle_token = np.array([""] * num_cand, dtype=object)
    main_desc_noatt = np.full((num_cand, NUM_DESCRIPTOR_FIELDS), np.nan, dtype=np.float32)
    main_vehicle_token_noatt = np.array([""] * num_cand, dtype=object)
    fault_vehicle_flag = np.zeros(num_cand, dtype=bool)

    n_ext = len(DESCRIPTOR_FIELDS_EXT)
    multi_descriptors = np.full((num_cand, top_m_vru, n_ext), np.nan, dtype=np.float32)
    multi_mask = np.zeros((num_cand, top_m_vru), dtype=bool)
    multi_toks = np.array([[""] * top_m_vru for _ in range(num_cand)], dtype=object)

    for k in range(num_cand):
        ego_polys = list(scorer._ego_polygons[k])  # (T+1,) shapely, global frame
        swept = unary_union(ego_polys)
        minx, miny, maxx, maxy = swept.bounds
        prefilter_box = (minx - prefilter_dist, miny - prefilter_dist,
                         maxx + prefilter_dist, maxy + prefilter_dist)

        att_col_set = set(att_fault[k])
        att_ttc_set = set(att_ttc[k])
        fault_vehicle_flag[k] = any(
            unique_objects[t].tracked_object_type == TrackedObjectType.VEHICLE
            for t in att_col_set
            if t in unique_objects
        )

        descs: List[Dict[str, float]] = []
        desc_tokens: List[str] = []
        for tok in vehicle_tokens:
            obj = unique_objects[tok]
            # spatial prefilter on the t=0 polygon
            occ0 = observation[0]
            if tok not in occ0.token_to_idx:
                continue
            j0 = occ0[tok]
            cx, cy = j0.centroid.x, j0.centroid.y
            if not (prefilter_box[0] <= cx <= prefilter_box[2]
                    and prefilter_box[1] <= cy <= prefilter_box[3]):
                continue

            j_polys: List[Optional[object]] = []
            for t in range(num_steps):
                occ = observation[t]
                j_polys.append(occ[tok] if tok in occ.token_to_idx else None)

            velocity = getattr(obj, "velocity", None)
            j_speed = (
                float(np.hypot(velocity.x, velocity.y)) if velocity is not None else np.nan
            )
            d = compute_interaction_descriptor(
                ego_polys,
                j_polys,
                ego_heading=ego_heading,
                j_heading=float(obj.box.center.heading),
                j_speed=j_speed,
                dt=dt,
                att_collision=tok in att_col_set,
                att_ttc=tok in att_ttc_set,
            )
            descs.append(d)
            desc_tokens.append(tok)

        order = select_top_m_vehicles(descs, top_m)
        for rank, di in enumerate(order):
            descriptors[k, rank] = [descs[di][f] for f in DESCRIPTOR_FIELDS]
            vehicle_mask[k, rank] = True
            if rank == 0:
                main_vehicle_token[k] = desc_tokens[di]
        # attribution-free top-1: the deployment-feasible "main vehicle"
        order_noatt = select_top_m_vehicles(descs, 1, use_attribution=False)
        if order_noatt:
            di = order_noatt[0]
            main_desc_noatt[k] = [descs[di][f] for f in DESCRIPTOR_FIELDS]
            main_vehicle_token_noatt[k] = desc_tokens[di]

        # ---- multi-class descriptors (vehicle + pedestrian + bicycle) --------
        if include_vru:
            m_descs: List[Dict[str, float]] = []
            m_tokens: List[str] = []
            for tok in multi_tokens:
                obj = unique_objects[tok]
                occ0 = observation[0]
                # spatial prefilter: on the t=0 polygon when present, else on
                # the first frame the track appears (emerging agents)
                prefilter_xy = None
                if tok in occ0.token_to_idx:
                    j0 = occ0[tok]
                    prefilter_xy = (j0.centroid.x, j0.centroid.y)
                else:
                    for t in range(1, num_steps):
                        occ = observation[t]
                        if tok in occ.token_to_idx:
                            jt = occ[tok]
                            prefilter_xy = (jt.centroid.x, jt.centroid.y)
                            break
                if prefilter_xy is None:
                    continue
                cx, cy = prefilter_xy
                if not (prefilter_box[0] <= cx <= prefilter_box[2]
                        and prefilter_box[1] <= cy <= prefilter_box[3]):
                    continue

                j_polys: List[Optional[object]] = []
                for t in range(num_steps):
                    occ = observation[t]
                    j_polys.append(occ[tok] if tok in occ.token_to_idx else None)

                velocity = getattr(obj, "velocity", None)
                j_speed = (
                    float(np.hypot(velocity.x, velocity.y))
                    if velocity is not None else np.nan
                )
                d = compute_interaction_descriptor(
                    ego_polys,
                    j_polys,
                    ego_heading=ego_heading,
                    j_heading=float(obj.box.center.heading),
                    j_speed=j_speed,
                    dt=dt,
                    att_collision=tok in att_col_set,
                    att_ttc=tok in att_ttc_set,
                    agent_class=float(aclass_of[obj.tracked_object_type]),
                )
                m_descs.append(d)
                m_tokens.append(tok)

            m_order = select_top_m_vehicles(m_descs, top_m_vru)
            for rank, di in enumerate(m_order):
                multi_descriptors[k, rank] = [
                    m_descs[di][f] for f in DESCRIPTOR_FIELDS_EXT
                ]
                multi_mask[k, rank] = True
                multi_toks[k, rank] = m_tokens[di]

    # ---- misc labels ----------------------------------------------------------
    es = np.asarray(rec["ego_status"], dtype=np.float64) if "ego_status" in rec else np.full(11, np.nan)
    ego_speed = float(np.hypot(es[3], es[4])) if np.isfinite(es[3:5]).all() else np.nan
    ade = (
        ade_to_human(proposals, np.asarray(rec["trajectory"]))
        if "trajectory" in rec
        else np.full(num_cand, np.nan)
    )

    save_npz(
        Path(out_path),
        token=rec["token"],
        log_name=rec["log_name"],
        selected_idx=rec["selected_idx"],
        subscores=subscores,
        descriptors=descriptors,
        vehicle_mask=vehicle_mask,
        main_vehicle_token=np.asarray(main_vehicle_token, dtype=object),
        main_desc_noatt=main_desc_noatt,
        main_vehicle_token_noatt=np.asarray(main_vehicle_token_noatt, dtype=object),
        att_fault_tokens=np.asarray([";".join(t) for t in att_fault], dtype=object),
        att_ttc_tokens=np.asarray([";".join(t) for t in att_ttc], dtype=object),
        fault_vehicle_flag=fault_vehicle_flag,
        ade_to_human=ade.astype(np.float32),
        ego_speed=np.float32(ego_speed),
        pdm_score=rec["pdm_score"].astype(np.float32),
        **(
            {
                "multi_descriptors": multi_descriptors,
                "multi_mask": multi_mask,
                "multi_tokens": np.asarray(multi_toks, dtype=object),
            }
            if include_vru
            else {}
        ),
    )
    return out_path


def main() -> None:
    args = parse_args()
    export_dir = Path(args.export_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_map = _metric_cache_map(Path(args.metric_cache_dir).resolve())

    token_filter: Optional[set] = None
    if args.token_list:
        token_filter = {l.strip() for l in open(args.token_list) if l.strip()}

    jobs = []
    for rec_path in sorted(export_dir.glob("*.npz")):
        token = rec_path.stem
        if token_filter is not None and token not in token_filter:
            continue
        out_path = out_dir / f"{token}.npz"
        if out_path.exists():
            continue
        if token not in cache_map:
            print(f"[label] WARNING: no metric cache for {token}, skipping")
            continue
        jobs.append((str(rec_path), cache_map[token], str(out_path)))
        if args.max_scenes and len(jobs) >= args.max_scenes:
            break

    print(f"[label] {len(jobs)} scenes to label with {args.workers} workers")
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(label_scene, rp, mcp, op, args.top_m, args.prefilter_dist,
                        args.cache_type, args.include_vru, args.top_m_vru): rp
            for rp, mcp, op in jobs
        }
        for i, fut in enumerate(as_completed(futures)):
            try:
                fut.result()
            except Exception:
                errors += 1
                print(f"[label] FAILED {futures[fut]}:")
                traceback.print_exc()
            if (i + 1) % 50 == 0:
                print(f"[label] {i + 1}/{len(jobs)} done ({errors} errors)")

    # ---- schema file ----------------------------------------------------------
    schema = {
        "subscores": {
            "shape": ["num_candidates", 6],
            "columns": ["NC", "DAC", "EP", "TTC", "Comfort", "final"],
            "note": "train_pdm_scorer PDMScorer sub-scores; final may differ "
                    "from the official navtest evaluator",
        },
        "descriptors": {
            "shape": ["num_candidates", "top_m", "num_fields"],
            "fields": DESCRIPTOR_FIELDS,
            "itype_codes": {str(k): v for k, v in ITYPE_NAMES.items()},
            "note": "att_* flags are TRAINING-SIDE labels from the PDM scorer; "
                    "deployment must not use GT attribution to choose vehicles. "
                    "rel_x/rel_y are in the ego-box-centroid frame at t=0.",
        },
        "main_desc_noatt": "float32 (num_candidates, num_fields) - descriptor of "
                           "the top-1 vehicle under the attribution-free ranking "
                           "(conflict > |dt_enter| > min_dist > |rel_x|); NaN row "
                           "when no vehicle was described. Leakage-safe default "
                           "for oracle_knn_check.",
        "main_vehicle_token_noatt": "track token of main_desc_noatt ('' if none)",
        "vehicle_mask": "bool (num_candidates, top_m) - real vehicle in slot",
        "main_vehicle_token": "track token of descriptors[:,0] ('' if none)",
        "descriptor_feature_vector": descriptor_feature_names(),
        "multi_descriptors": {
            "shape": ["num_candidates", "top_m_vru", "num_fields_ext"],
            "fields": DESCRIPTOR_FIELDS_EXT,
            "agent_class_codes": {str(k): v for k, v in ACLASS_NAMES.items()},
            "note": "only when --include_vru: same fields + agent_class + "
                    "emerging (t0-absent, later-conflicting), covering "
                    "VEHICLE/PEDESTRIAN/BICYCLE tracks ranked together",
        },
        "multi_mask": "bool (num_candidates, top_m_vru) - real agent in slot",
        "multi_tokens": "track tokens of multi_descriptors slots ('' if none)",
    }
    with open(out_dir / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    print(f"[label] done: {len(jobs) - errors}/{len(jobs)} labelled -> {out_dir}")


if __name__ == "__main__":
    main()
