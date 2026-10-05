#!/usr/bin/env python
"""EWM-JEPA Step 2 - Per-scene PDM subscores over the fixed anchor vocabulary.

Computes o(x) in [0,1]^{|V|x5}: for each metric_cache token, score all anchor
trajectories with the train-side PDMScorer and store the (n_anchor, 6) matrix
[NC, DAC, EP, TTC, Comfort, final] as float16. CPU-only, multiprocess,
resumable, sharded by log.

This is a v1 port of ``navsim_v2/scripts/misc/calc_anchors_scores.py`` with the
``anchors_scores_index`` dependency removed -- we compute the full subscore
matrix directly instead of relying on the precomputed top-256 index.

Usage:

    # 50-scene timing run first
    python scripts/experience/compute_anchor_subscores.py \
        --metric_cache_dir $NAVSIM_EXP_ROOT/Drive-JEPA-cache/train_metric_cache \
        --token_list navtrain_runnable.txt \
        --out_dir $NAVSIM_EXP_ROOT/experience/anchor_scores_navtrain \
        --max_scenes 50 --workers 32

    # memory-safe subset: score only a fixed random anchor subset
    python scripts/experience/compute_anchor_subscores.py ... \
        --anchor_subset 1024 --subset_seed 0
"""

import argparse
import lzma
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric_cache_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--token_list", default=None,
                   help="file with one token per line; omit = all tokens in cache")
    p.add_argument("--max_scenes", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--anchor_subset", type=int, default=None,
                   help="score only this many anchors (uniform random)")
    p.add_argument("--subset_seed", type=int, default=0)
    p.add_argument("--anchors_path", default=str(NAVSIM_V1_ROOT / "data" / "8192.npy"))
    return p.parse_args()


def find_metric_cache(root: Path, token: str) -> Optional[Path]:
    hits = list(root.rglob(f"{token}/metric_cache.pkl"))
    return hits[0] if hits else None


def score_one(args_tuple) -> Tuple[str, str, float, Optional[str]]:
    """Score one scene; returns (token, log, seconds, error)."""
    token, log, pkl_path, poses, out_path = args_tuple
    t0 = time.time()
    try:
        from navsim.agents.pad.score_module.compute_navsim_score import (
            before_score,
            scorer,
        )
        from navsim.planning.metric_caching.metric_cache import MetricCache

        with lzma.open(pkl_path, "rb") as f:
            metric_cache: MetricCache = pickle.load(f)

        simulated = before_score(metric_cache, poses)
        final_scores = scorer.score_proposals(
            simulated,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.pdm_progress,
        )
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
            MultiMetricIndex,
            WeightedMetricIndex,
        )

        nc = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, :]
        dac = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, :]
        ep = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, :]
        ttc = scorer._weighted_metrics[WeightedMetricIndex.TTC, :]
        comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, :]
        scores = np.stack([nc, dac, ep, ttc, comfort, final_scores], axis=-1)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, scores.astype(np.float16))
        return token, log, time.time() - t0, None
    except Exception as e:
        return token, log, time.time() - t0, f"{type(e).__name__}: {e}"


def main():
    args = parse_args()
    root = Path(args.metric_cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    poses = np.load(args.anchors_path)[:, 4::5]  # (8192, 8, 3)
    if args.anchor_subset:
        rng = np.random.default_rng(args.subset_seed)
        idx = np.sort(rng.choice(len(poses), args.anchor_subset, replace=False))
        np.save(out_dir / "anchor_subset_idx.npy", idx)
        poses = poses[idx]
        print(f"[anchor_scores] using subset of {len(poses)} anchors "
              f"(idx saved to anchor_subset_idx.npy)")

    # token -> (log, pkl path)
    token_filter = None
    if args.token_list:
        token_filter = {l.strip() for l in open(args.token_list) if l.strip()}

    jobs = []
    for log_dir in sorted(root.iterdir()):
        if not log_dir.is_dir():
            continue
        for pkl in sorted(log_dir.rglob("metric_cache.pkl")):
            token = pkl.parent.name
            if token_filter is not None and token not in token_filter:
                continue
            out_path = out_dir / log_dir.name / f"{token}.npy"
            if out_path.exists():
                continue
            jobs.append((token, log_dir.name, str(pkl), poses, str(out_path)))
            if args.max_scenes and len(jobs) >= args.max_scenes:
                break
        if args.max_scenes and len(jobs) >= args.max_scenes:
            break

    print(f"[anchor_scores] {len(jobs)} scenes to score "
          f"({poses.shape[0]} anchors each) with {args.workers} workers")

    times = []
    n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(score_one, j): j[0] for j in jobs}
        for i, fut in enumerate(as_completed(futs)):
            token, log, dt, err = fut.result()
            if err:
                n_err += 1
                print(f"[anchor_scores] ERROR {token}: {err}")
            else:
                times.append(dt)
            if (i + 1) % 10 == 0 or i + 1 == len(futs):
                med = np.median(times) if times else float("nan")
                print(f"[anchor_scores] {i+1}/{len(futs)} done, median {med:.1f}s/scene, errors={n_err}")

    if times:
        print(f"[anchor_scores] median {np.median(times):.1f}s/scene; "
              f"est. serial total for 5k scenes: {np.median(times)*5000/3600:.1f}h")
    print(f"[anchor_scores] done: {len(times)} ok, {n_err} errors -> {out_dir}")


if __name__ == "__main__":
    main()
