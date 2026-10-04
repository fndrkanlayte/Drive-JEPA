#!/usr/bin/env python
"""A4 - Headroom analysis: how much score is left on the table by argmax(pdm_score)?

Per label directory (one per split), compares per scene:
  * the labelled subscore of the selected candidate (argmax pdm_score), vs
  * the oracle best subscore over all 32 candidates,
for final score, NC and TTC; also restricted to the subset of scenes whose
main vehicle conflicts with |dt_enter| < 2s. Scene-level bootstrap 95% CI.

Usage:
    python scripts/experience/headroom.py --labels_dir DIR [--name navtrain] [--num_boot 1000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))

from navsim.agents.drive_jepa_perception_based.experience.descriptors import (  # noqa: E402
    DESCRIPTOR_FIELD_INDEX as FI,
)
from navsim.agents.drive_jepa_perception_based.experience.records import (  # noqa: E402
    bootstrap_scene_ci,
    fmt_ci,
    load_npz,
)

SUB_NC, SUB_TTC, SUB_FINAL = 0, 3, 5
METRICS = {"final": SUB_FINAL, "NC": SUB_NC, "TTC": SUB_TTC}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--name", type=str, default=None, help="split name for the table title")
    p.add_argument("--num_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt_enter_thresh", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    name = args.name or labels_dir.name

    sel, oracle, subset = {m: [] for m in METRICS}, {m: [] for m in METRICS}, []
    n_files = 0
    for fp in sorted(labels_dir.glob("*.npz")):
        rec = load_npz(fp)
        sub = rec["subscores"]
        sel_idx = int(rec["selected_idx"].item())
        for m, col in METRICS.items():
            sel[m].append(sub[sel_idx, col])
            oracle[m].append(sub[:, col].max())
        desc, mask = rec["descriptors"], rec["vehicle_mask"]
        # subset: the SELECTED candidate's main vehicle conflicts with |dt_enter| < thresh
        if mask[sel_idx, 0] and desc[sel_idx, 0, FI["conflict"]] == 1.0:
            dt_enter = desc[sel_idx, 0, FI["dt_enter"]]
            subset.append(np.isfinite(dt_enter) and abs(dt_enter) < args.dt_enter_thresh)
        else:
            subset.append(False)
        n_files += 1

    subset = np.asarray(subset, dtype=bool)
    print(f"\n## Headroom — {name} ({n_files} scenes, {int(subset.sum())} in conflict subset)\n")
    print("| metric | argmax(pdm_score) | oracle best of 32 | gap | subset argmax | subset oracle | subset gap |")
    print("|---|---|---|---|---|---|---|")
    for m in METRICS:
        s = np.asarray(sel[m]); o = np.asarray(oracle[m])
        ps, lo_s, hi_s = bootstrap_scene_ci(s, args.num_boot, args.seed)
        po, lo_o, hi_o = bootstrap_scene_ci(o, args.num_boot, args.seed + 1)
        pg, lo_g, hi_g = bootstrap_scene_ci(o - s, args.num_boot, args.seed + 2)
        if subset.sum() >= 5:
            pss, lss, hss = bootstrap_scene_ci(s[subset], args.num_boot, args.seed + 3)
            pos, los, hos = bootstrap_scene_ci(o[subset], args.num_boot, args.seed + 4)
            pgs, lgs, hgs = bootstrap_scene_ci((o - s)[subset], args.num_boot, args.seed + 5)
            row = (f"| {m} | {fmt_ci(ps, lo_s, hi_s)} | {fmt_ci(po, lo_o, hi_o)} | "
                   f"{fmt_ci(pg, lo_g, hi_g)} | {fmt_ci(pss, lss, hss)} | "
                   f"{fmt_ci(pos, los, hos)} | {fmt_ci(pgs, lgs, hgs)} |")
        else:
            row = (f"| {m} | {fmt_ci(ps, lo_s, hi_s)} | {fmt_ci(po, lo_o, hi_o)} | "
                   f"{fmt_ci(pg, lo_g, hi_g)} | - | - | - |")
        print(row)


if __name__ == "__main__":
    main()
