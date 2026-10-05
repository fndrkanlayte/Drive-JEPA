#!/usr/bin/env python
"""EWM-JEPA Step 2 - Build token -> future-token map for z_{t+H} targets.

For every scene token in a metric_cache dir, read ego_state.time_point.time_us
and, within the same log, find the token whose timestamp is closest to
+``--horizon`` seconds (within +-``--tol`` seconds). Saves a JSON map

    {token: {"future_token": ..., "dt_s": float}}

plus a per-log timestamp index for inspection. The latent cache shards
(export_latents.py) are joined on this map afterwards; a token without a
+4s neighbour simply gets no z_{t+H} entry (boundary of each log window).

Usage:

    python scripts/experience/build_future_map.py \
        --metric_cache_dir $NAVSIM_EXP_ROOT/Drive-JEPA-cache/train_metric_cache \
        --token_list $EXPORT/manifest_tokens.txt \
        --out $NAVSIM_EXP_ROOT/experience/navtrain_future_map.json
"""

import argparse
import json
import lzma
import pickle
import sys
from pathlib import Path

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric_cache_dir", required=True,
                   help="dir with <log>/<subdir>/<token>/metric_cache.pkl layout")
    p.add_argument("--token_list", default=None,
                   help="optional file with tokens to include (others are skipped in the map but still indexable)")
    p.add_argument("--horizon", type=float, default=4.0, help="future horizon in seconds")
    p.add_argument("--tol", type=float, default=1.0, help="acceptable |dt - horizon| in seconds")
    p.add_argument("--out", required=True)
    return p.parse_args()


def iter_metric_caches(root: Path):
    for log_dir in sorted(root.iterdir()):
        if not log_dir.is_dir():
            continue
        for pkl in log_dir.rglob("metric_cache.pkl"):
            yield log_dir.name, pkl


def main():
    args = parse_args()
    root = Path(args.metric_cache_dir)

    token_filter = None
    if args.token_list:
        token_filter = {l.strip() for l in open(args.token_list) if l.strip()}

    # pass 1: token -> (log, time_s)
    times = {}          # token -> (log, t_seconds)
    by_log = {}         # log -> list[(t_seconds, token)]
    for log, pkl in iter_metric_caches(root):
        token = pkl.parent.name
        try:
            with lzma.open(pkl, "rb") as f:
                mc = pickle.load(f)
            t = mc.ego_state.time_point.time_us / 1e6
        except Exception as e:
            print(f"[future_map] skip {token}: {e}")
            continue
        times[token] = (log, t)
        by_log.setdefault(log, []).append((t, token))
    print(f"[future_map] indexed {len(times)} tokens in {len(by_log)} logs")

    # pass 2: nearest token ~ +horizon
    fmap = {}
    n_ok = n_edge = 0
    horizon_us = args.horizon
    for token, (log, t) in times.items():
        if token_filter is not None and token not in token_filter:
            continue
        best = None
        for t2, tok2 in by_log[log]:
            dt = t2 - t
            if abs(dt - horizon_us) <= args.tol and (best is None or abs(dt - horizon_us) < abs(best[0] - horizon_us)):
                best = (dt, tok2)
        if best is None:
            n_edge += 1
            continue
        fmap[token] = {"future_token": best[1], "dt_s": round(best[0], 3), "log_name": log}
        n_ok += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"horizon_s": args.horizon, "tol_s": args.tol, "map": fmap}, f)
    total = len(fmap) + n_edge
    print(f"[future_map] {n_ok}/{total} tokens have a +{args.horizon}s neighbour "
          f"(coverage {n_ok/max(total,1)*100:.1f}%) -> {out}")


if __name__ == "__main__":
    main()
