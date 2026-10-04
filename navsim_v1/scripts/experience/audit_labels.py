#!/usr/bin/env python
"""A3 - Audit the labels produced by label_candidates.py.

Checks:
  1. Recompute check (needs --official_csv from run_pdm_score.py): for a few
     navtest tokens, compare the final subscore of the exported selected_idx
     against the official evaluator's `score`. NOTE: labels come from the
     *training* PDMScorer (score_module/train_pdm_scorer.py) and for the
     official navtest cache the reference progress is recomputed - small
     differences vs the official evaluator are expected and reported, not
     asserted.
  2. Sanity: every candidate with NC==0 caused by a VEHICLE should have its
     main vehicle (descriptors[:,0]) with att_collision=1 and min_dist~=0.
  3. Summary stats: conflict fraction, itype histogram, dt_enter histogram,
     censoring rates (all on the main vehicle).

Usage:
    python scripts/experience/audit_labels.py \
        --labels_dir $NAVSIM_EXP_ROOT/experience/navtest_labels \
        [--official_csv <run_pdm_score output .csv>] [--n_scenes 20]
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
    ITYPE_NAMES,
)
from navsim.agents.drive_jepa_perception_based.experience.records import load_npz  # noqa: E402

SUB_FINAL = 5
SUB_NC = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--official_csv", type=str, default=None,
                   help="CSV written by run_pdm_score.py (has token,score,valid columns)")
    p.add_argument("--n_scenes", type=int, default=20,
                   help="max scenes used for the recompute check")
    p.add_argument("--min_dist_tol", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    files = sorted(labels_dir.glob("*.npz"))
    print(f"[audit] {len(files)} labelled scenes in {labels_dir}")

    # ------------------------------------------------------------------ #
    # 1. recompute check vs official run_pdm_score csv
    # ------------------------------------------------------------------ #
    if args.official_csv:
        import csv as csvmod

        official = {}
        with open(args.official_csv) as f:
            for row in csvmod.DictReader(f):
                if row.get("token") and row["token"] != "average" and row.get("valid") == "True":
                    try:
                        official[row["token"]] = float(row["score"])
                    except (TypeError, ValueError):
                        pass
        diffs = []
        checked = 0
        for fp in files:
            if checked >= args.n_scenes:
                break
            rec = load_npz(fp)
            token = str(rec["token"].item())
            if token not in official:
                continue
            ours = float(rec["subscores"][int(rec["selected_idx"].item()), SUB_FINAL])
            diffs.append(ours - official[token])
            checked += 1
        diffs = np.asarray(diffs)
        print(f"[audit] recompute check on {checked} tokens (train PDMScorer vs official evaluator)")
        if checked:
            print(
                f"         mean|diff|={np.abs(diffs).mean():.5f}  max|diff|={np.abs(diffs).max():.5f}  "
                f"num>|0.01|={int((np.abs(diffs) > 0.01).sum())}"
            )
        else:
            print("         no overlapping tokens between labels and official csv")

    # ------------------------------------------------------------------ #
    # 2 + 3. sanity + summary stats
    # ------------------------------------------------------------------ #
    n_cand = n_nc0_vehicle = n_sanity_ok = 0
    main_conflict = []
    any_conflict = []
    itypes = []
    dt_enters = []
    censored = {k: [] for k in ("k_in", "k_out", "j_in", "j_out")}

    for fp in files:
        rec = load_npz(fp)
        sub = rec["subscores"]
        desc = rec["descriptors"]
        mask = rec["vehicle_mask"]
        fault_vehicle = rec["fault_vehicle_flag"]
        K = sub.shape[0]
        n_cand += K

        nc0_vehicle = (sub[:, SUB_NC] == 0.0) & fault_vehicle
        n_nc0_vehicle += int(nc0_vehicle.sum())
        # main vehicle fields (slot 0)
        main_att = desc[:, 0, FI["att_collision"]] == 1.0
        main_min_dist = desc[:, 0, FI["min_dist"]]
        ok = mask[:, 0] & main_att & (np.nan_to_num(main_min_dist, nan=np.inf) <= args.min_dist_tol)
        n_sanity_ok += int((nc0_vehicle & ok).sum())

        main_conflict.append(desc[mask[:, 0], 0, FI["conflict"]])
        any_conflict.append(np.nan_to_num(desc[..., FI["conflict"]], nan=0.0).max(axis=1))
        itypes.append(desc[:, 0, FI["itype"]])
        dt_enters.append(desc[:, 0, FI["dt_enter"]])
        for name, field in (("k_in", "k_in_censored"), ("k_out", "k_out_censored"),
                            ("j_in", "j_in_censored"), ("j_out", "j_out_censored")):
            censored[name].append(desc[:, 0, FI[field]])

    main_conflict = np.concatenate(main_conflict)
    any_conflict = np.concatenate(any_conflict)
    itypes = np.nan_to_num(np.concatenate(itypes), nan=0.0).astype(int)
    dt_enters = np.concatenate(dt_enters)

    print(f"\n[audit] sanity: {n_nc0_vehicle} candidates with NC==0 due to a VEHICLE; "
          f"{n_sanity_ok} have main vehicle att_collision & min_dist<={args.min_dist_tol}m "
          f"({(n_sanity_ok / max(n_nc0_vehicle, 1)):.1%})")

    print(f"\n[audit] summary over {n_cand} candidates in {len(files)} scenes")
    if len(main_conflict):
        print(f"         main-vehicle conflict rate: {main_conflict.mean():.3f} "
              f"(over {len(main_conflict)} candidates with a main vehicle)")
    print(f"         any-of-topM conflict rate:  {any_conflict.mean():.3f}")

    print("         itype histogram (main vehicle):")
    for code, name in ITYPE_NAMES.items():
        print(f"           {name:9s}: {(itypes == code).mean():.3f}")

    valid_dt = dt_enters[np.isfinite(dt_enters)]
    print(f"         |dt_enter| histogram (n={len(valid_dt)}):")
    if len(valid_dt):
        for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, np.inf)]:
            frac = ((np.abs(valid_dt) >= lo) & (np.abs(valid_dt) < hi)).mean()
            print(f"           [{lo},{hi})s: {frac:.3f}")
        print(f"           sign: ego-first {(valid_dt < 0).mean():.3f}, j-first {(valid_dt > 0).mean():.3f}")

    print("         censoring rates (main vehicle):")
    for name, vals in censored.items():
        v = np.nan_to_num(np.concatenate(vals), nan=0.0)
        print(f"           {name}: {v.mean():.3f}")


if __name__ == "__main__":
    main()
