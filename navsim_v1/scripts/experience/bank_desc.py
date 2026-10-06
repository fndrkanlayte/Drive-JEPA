#!/usr/bin/env python
"""Compute agent interaction descriptors for the CLOVER bank.

bank_ours.npz rows are (count_i, 8, 3) ego-relative trajectories per scene
token. For every scene we reload its train metric_cache, run the PDM scorer
on the bank trajectories, and reuse the same per-candidate descriptor code
as label_candidates (_vehicle_descriptor_row, use_attribution=True slot
order). Output bank_ours_desc.npz:

    tokens       (S,) str        scene token, same order as bank_ours.npz
    counts       (S,) int64
    descriptors  (N, M, F) f32   row-aligned with bank_ours.npz trajs
    vehicle_mask (N, M) bool

--check N: consistency check — recompute descriptors for the B0 proposals of
N label scenes and assert they equal the stored label npz exactly.
"""
import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_candidates import (  # noqa: E402
    NUM_DESCRIPTOR_FIELDS,
    _load_metric_cache,
    _metric_cache_map,
    _pdm_reference_progress,
    _vehicle_descriptor_row,
)


def describe_token(task):
    """task = (token, trajs (n,8,3), cache_path, top_m, prefilter_dist).

    Returns (token, descriptors (n,top_m,F), vehicle_mask (n,top_m))."""
    from nuplan.common.actor_state.tracked_objects_types import (
        TrackedObjectType,
    )
    from navsim.agents.drive_jepa_perception_based.score_module import (
        compute_navsim_score as cns,
    )

    token, trajs, mc_path, top_m, prefilter_dist = task
    mc = _load_metric_cache(mc_path)
    trajs = np.asarray(trajs, dtype=np.float64)
    sim = cns.before_score(mc, trajs)

    pdm_progress = getattr(mc, "pdm_progress", None)
    if pdm_progress is None:
        pdm_progress = _pdm_reference_progress(mc, cns)
    cns.scorer.score_proposals(
        sim,
        mc.observation,
        mc.centerline,
        mc.route_lane_ids,
        mc.drivable_area_map,
        pdm_progress,
    )
    scorer = cns.scorer
    obs = mc.observation
    uo = obs.unique_objects
    n_cand, num_steps = len(trajs), sim.shape[1]
    dt = cns.proposal_sampling.interval_length
    ego_heading = float(mc.ego_state.rear_axle.heading)
    vtoks = [
        t for t, o in uo.items()
        if o.tracked_object_type == TrackedObjectType.VEHICLE
    ]
    pre = set(obs.collided_track_ids)
    att_fault = [
        [t for t in scorer.proposal_fault_collided_track_ids[k]
         if t not in pre]
        for k in range(n_cand)
    ]
    att_ttc = [
        [t for t in scorer.ttc_collided_track_ids[k] if t not in pre]
        for k in range(n_cand)
    ]

    desc = np.full((n_cand, top_m, NUM_DESCRIPTOR_FIELDS), np.nan,
                   dtype=np.float32)
    vmask = np.zeros((n_cand, top_m), dtype=bool)
    for k in range(n_cand):
        row = _vehicle_descriptor_row(
            list(scorer._ego_polygons[k]), obs, uo, vtoks,
            set(att_fault[k]), set(att_ttc[k]), ego_heading, dt,
            num_steps, top_m, prefilter_dist,
        )
        desc[k], vmask[k] = row[0], row[1]
    return token, desc, vmask


def describe_token_safe(task):
    try:
        return describe_token(task)
    except Exception:
        traceback.print_exc()
        return task[0], None, None


def load_bank(bank_path):
    z = np.load(bank_path, allow_pickle=True)
    return z["tokens"].astype(str), z["counts"], z["trajs"]


def check_consistency(export_dir, labels_dir, cache_map, n_scenes, top_m,
                      prefilter_dist):
    """Recompute B0-proposal descriptors for n_scenes label scenes and
    compare to the stored label arrays."""
    export_dir, labels_dir = Path(export_dir), Path(labels_dir)
    done = 0
    for lf in sorted(labels_dir.glob("*.npz")):
        if done >= n_scenes:
            break
        tok = lf.stem
        rec_f = export_dir / f"{tok}.npz"
        if not rec_f.exists():
            hits = list(export_dir.rglob(f"{tok}.npz"))
            rec_f = hits[0] if hits else rec_f
        if tok not in cache_map or not rec_f.exists():
            continue
        rec = np.load(rec_f, allow_pickle=True)
        lab = np.load(lf, allow_pickle=True)
        _, desc, vmask = describe_token(
            (tok, rec["proposals"], cache_map[tok], top_m, prefilter_dist))
        ref_d = np.asarray(lab["descriptors"], dtype=np.float32)
        ref_m = np.asarray(lab["vehicle_mask"], dtype=bool)
        eq_m = np.array_equal(vmask, ref_m)
        eq_d = (np.isnan(desc) == np.isnan(ref_d)).all() and np.allclose(
            np.nan_to_num(desc), np.nan_to_num(ref_d), atol=0)
        print(f"[check] {tok} mask_eq={eq_m} desc_eq={eq_d}")
        if not (eq_m and eq_d):
            raise SystemExit(f"consistency check FAILED on {tok}")
        done += 1
    print(f"[check] {done} scenes all identical")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bank_npz", required=True)
    p.add_argument("--metric_cache_dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--top_m", type=int, default=4)
    p.add_argument("--prefilter_dist", type=float, default=50.0,
                   help="must match label_candidates default (50.0)")
    p.add_argument("--workers", type=int, default=96)
    p.add_argument("--check", type=int, default=0,
                   help="run N-scene B0-descriptor consistency check")
    p.add_argument("--export_dir", default=None)
    p.add_argument("--labels_dir", default=None)
    p.add_argument("--scenes", type=int, default=0,
                   help="limit to first N bank scenes (smoke test)")
    args = p.parse_args()

    cache_map = _metric_cache_map(Path(args.metric_cache_dir))
    toks, counts, trajs = load_bank(args.bank_npz)

    if args.check:
        check_consistency(args.export_dir, args.labels_dir, cache_map,
                          args.check, args.top_m, args.prefilter_dist)
        sys.exit(0)

    n_scenes = args.scenes or len(toks)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    tasks = []
    for i in range(min(n_scenes, len(toks))):
        t = toks[i]
        if t not in cache_map:
            continue
        sl = slice(offsets[i], offsets[i] + counts[i])
        tasks.append((t, trajs[sl], cache_map[t], args.top_m,
                      args.prefilter_dist))
    print(f"[bank_desc] scenes={len(tasks)}/{len(toks)} "
          f"trajs={sum(len(t[1]) for t in tasks)} workers={args.workers}")

    out_d = np.empty((0, args.top_m, NUM_DESCRIPTOR_FIELDS), np.float32)
    out_m = np.empty((0, args.top_m), bool)
    out_tok, out_cnt = [], []
    n_done = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut_res in ex.map(describe_token_safe, tasks, chunksize=4):
            tok, desc, vmask = fut_res
            if desc is None:
                n_err += 1
                continue
            out_tok.append(tok)
            out_cnt.append(len(desc))
            out_d = np.concatenate([out_d, desc])
            out_m = np.concatenate([out_m, vmask])
            n_done += 1
            if n_done % 500 == 0:
                print(f"[bank_desc] {n_done}/{len(tasks)} scenes done",
                      flush=True)
    if n_err:
        print(f"[bank_desc] WARNING {n_err} scenes failed")
    if args.out:
        np.savez(args.out, tokens=np.asarray(out_tok),
                 counts=np.asarray(out_cnt, np.int64),
                 descriptors=out_d, vehicle_mask=out_m)
        print(f"[bank_desc] wrote {args.out} scenes={len(out_tok)} "
              f"trajs={len(out_d)}")
