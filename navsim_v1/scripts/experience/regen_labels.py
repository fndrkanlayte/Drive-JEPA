#!/usr/bin/env python
"""Regenerate label npz descriptors with the current code -> labels_v2.

For every scene in an existing labels dir, reload the export record's B0
proposals, recompute the descriptor block via bank_desc.describe_token, and
write a new npz to --out_dir that copies every field of the old label except
the descriptor-derived ones, which are replaced:

    descriptors, vehicle_mask, main_vehicle_token, main_desc_noatt,
    main_vehicle_token_noatt, fault_vehicle_flag,
    att_fault_tokens, att_ttc_tokens

subscores / selected_idx / pdm_score / ade / ego_speed are copied verbatim.

--determinism N: run N scenes twice and compare both recomputes bitwise.
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

from label_candidates import _metric_cache_map  # noqa: E402
from bank_desc import describe_token  # noqa: E402

REPLACE_KEYS = {
    "descriptors", "vehicle_mask", "main_vehicle_token", "main_desc_noatt",
    "main_vehicle_token_noatt", "fault_vehicle_flag", "att_fault_tokens",
    "att_ttc_tokens",
}


def regen_one(task):
    """(tok, record_path, label_path, cache_path, out_dir, top_m,
    prefilter_dist) -> (tok, n_written or None)."""
    tok, rec_p, lab_p, mc_p, out_dir, top_m, pf = task
    try:
        rec = np.load(rec_p, allow_pickle=True)
        _, desc, vmask, aux = describe_token(
            (tok, rec["proposals"], mc_p, top_m, pf))
        old = np.load(lab_p, allow_pickle=True)
        out = {}
        for k in old.files:
            out[k] = old[k] if k not in REPLACE_KEYS else None
        out["descriptors"] = desc
        out["vehicle_mask"] = vmask
        for k, v in aux.items():
            out[k] = v
        out = {k: v for k, v in out.items() if v is not None}
        np.savez(Path(out_dir) / f"{tok}.npz", **out)
        return tok, len(desc)
    except Exception:
        traceback.print_exc()
        return tok, None


def determinism_check(tasks, n):
    res = {}
    for t in tasks[:n]:
        a = regen_one(t[:6] + ("/tmp/_det_a",))[0] if False else None
    # run describe_token twice per scene (no file write)
    bad = []
    for t in tasks[:n]:
        tok, rec_p, _, mc_p, _, top_m, pf = t
        rec = np.load(rec_p, allow_pickle=True)
        r1 = describe_token((tok, rec["proposals"], mc_p, top_m, pf))
        r2 = describe_token((tok, rec["proposals"], mc_p, top_m, pf))
        ok = (np.array_equal(np.isnan(r1[1]), np.isnan(r2[1]))
              and np.allclose(np.nan_to_num(r1[1]),
                              np.nan_to_num(r2[1]), atol=0)
              and np.array_equal(r1[2], r2[2])
              and np.array_equal(
                  r1[3]["main_vehicle_token"], r2[3]["main_vehicle_token"])
              and np.array_equal(
                  r1[3]["att_fault_tokens"], r2[3]["att_fault_tokens"])
              and np.array_equal(
                  r1[3]["att_ttc_tokens"], r2[3]["att_ttc_tokens"]))
        print(f"[det] {tok} identical={ok}")
        if not ok:
            bad.append(tok)
    if bad:
        raise SystemExit(f"NON-DETERMINISTIC scorer on {bad}")
    print(f"[det] all {n} scenes deterministic")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--export_dir", required=True)
    p.add_argument("--metric_cache_dir", required=True)
    p.add_argument("--out_dir", required=False, default=None)
    p.add_argument("--top_m", type=int, default=4)
    p.add_argument("--prefilter_dist", type=float, default=50.0)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--determinism", type=int, default=0)
    args = p.parse_args()

    cache_map = _metric_cache_map(Path(args.metric_cache_dir))
    labels_dir = Path(args.labels_dir)
    tasks = []
    for lf in sorted(labels_dir.glob("*.npz")):
        tok = lf.stem
        if tok not in cache_map:
            continue
        rec = Path(args.export_dir) / f"{tok}.npz"
        if not rec.exists():
            hits = list(Path(args.export_dir).rglob(f"{tok}.npz"))
            rec = hits[0] if hits else None
        if rec is None:
            continue
        tasks.append((tok, str(rec), str(lf), cache_map[tok],
                      args.out_dir or "/tmp", args.top_m,
                      args.prefilter_dist))
    print(f"[regen] scenes={len(tasks)} workers={args.workers}",
          flush=True)

    if args.determinism:
        determinism_check(tasks, args.determinism)
        sys.exit(0)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    n_done = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for tok, n in ex.map(regen_one, tasks, chunksize=4):
            if n is None:
                n_err += 1
            else:
                n_done += 1
            if (n_done + n_err) % 500 == 0:
                print(f"[regen] {n_done+n_err}/{len(tasks)} err={n_err}",
                      flush=True)
    print(f"[regen] DONE scenes={n_done} errors={n_err} -> {args.out_dir}")
