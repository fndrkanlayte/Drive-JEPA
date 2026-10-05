#!/usr/bin/env python3
"""Q7 eval: held-out memory protocol on navtest.

Compares selections ``argmax(pdm_score - lam * r_hat)`` produced from risk
npz files generated under different memory banks (A-only vs A+B logs),
on navtest subsets: all, conflict, top rarity deciles (d7/d8/d9 + top30%).

Usage:
  python scripts/experience/q7_eval.py \
      --navtest_labels_dir $EXP/experience/navtest_labels_full \
      --navtest_export_dir $EXP/experience/navtest_export_full \
      --rarity_npz $EXP/experience/rarity_strat_out/rarity.npz \
      --risk_npz memA_retrieval_int=$E/q7_A/retrieval_int.npz \
                 memAB_retrieval_int=$E/q7_AB/retrieval_int.npz ... \
      --lam 0.5 --out_dir $EXP/experience/q7_eval_out
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fusion_rerank as fr  # noqa: E402
from navsim.agents.drive_jepa_perception_based.experience.records import (  # noqa: E402
    bootstrap_scene_ci,
    fmt_ci,
    load_npz,
)

SUB_NC, SUB_EP, SUB_TTC, SUB_FINAL = 0, 2, 3, 5


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--navtest_labels_dir", required=True)
    p.add_argument("--navtest_export_dir", required=True)
    p.add_argument("--rarity_npz", required=True)
    p.add_argument("--risk_npz", nargs="+", required=True,
                   help="name=path entries")
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_boot", type=int, default=1000)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--dt_enter_thresh", type=float, default=2.0)
    return p.parse_args()


def r_hat(risk):
    p = np.clip(risk.astype(np.float64), 0.0, 1.0)
    return 1.0 - (1.0 - p[..., 0]) * (1.0 - p[..., 1])


def eval_subset(sel, subscores, keep, num_boot, seed, sel0):
    rows = np.arange(subscores.shape[0])[keep]
    if len(rows) == 0:
        return None
    out = {}
    for metric, col in (("final", SUB_FINAL), ("NC", SUB_NC),
                        ("TTC", SUB_TTC), ("EP", SUB_EP)):
        out[metric] = bootstrap_scene_ci(subscores[rows, sel[rows], col],
                                         num_boot, seed)
    out["frac"] = float((sel[rows] != sel0[rows]).mean())
    out["n"] = int(len(rows))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    nt = fr.scene_pack(Path(args.navtest_labels_dir),
                       Path(args.navtest_export_dir), args.dt_enter_thresh)
    S = len(nt["tokens"])
    sel0 = np.argmax(nt["pdm_score"], axis=1)
    scene_subset = nt["subset"][np.arange(S), sel0]

    rz = np.load(args.rarity_npz, allow_pickle=True)
    rare = rz["rarity"]
    t2r = {t: i for i, t in enumerate(rz["tokens"].tolist())}
    rare = rare[[t2r[t] for t in nt["tokens"]]]
    dec = np.quantile(rare, np.linspace(0, 1, 11))
    subsets = {"all": np.ones(S, bool), "conflict": scene_subset}
    for d in (7, 8, 9):
        lo, hi = dec[d], dec[d + 1]
        subsets[f"rare_d{d}"] = (rare >= lo) & (
            rare <= hi if d == 9 else rare < hi)
    subsets["rare_top30"] = rare >= dec[7]
    subsets["conflict∩rare_top30"] = scene_subset & (rare >= dec[7])

    rows_md = []
    results = {}

    def report(name, sel):
        for sname, keep in subsets.items():
            res = eval_subset(sel, nt["subscores"], keep,
                              args.num_boot, args.split_seed, sel0)
            results[f"{name}|{sname}"] = res
            for m in ("final", "NC", "TTC", "EP"):
                rows_md.append([name, f"{sname} (n={res['n']})", m,
                                fmt_ci(*res[m]), f"{res['frac']:.3f}"])

    report("original_argmax", sel0)
    for entry in args.risk_npz:
        name, path = entry.split("=", 1)
        risk = load_npz(Path(path))
        t2i = {t: i for i, t in enumerate(risk["tokens"].tolist())}
        idx = np.array([t2i[t] for t in nt["tokens"] if t in t2i])
        keep = np.array([i for i, t in enumerate(nt["tokens"])
                         if t in t2i])
        sel = sel0.copy()
        sel[keep] = np.argmax(nt["pdm_score"][keep]
                              - args.lam * r_hat(risk["risk"][idx]), axis=1)
        report(name, sel)

    md = ["# Q7 held-out memory protocol (navtest)", ""]
    md.append(f"navtest scenes: {S} | lam={args.lam} | "
              f"rarity decile edges: {np.round(dec, 4).tolist()}")
    md.append("")
    md.append("| selector | subset | metric | value [95% CI] | frac_changed |")
    md.append("|---|---|---|---|---|")
    for r in rows_md:
        md.append("| " + " | ".join(r) + " |")
    (out_dir / "q7_results.md").write_text("\n".join(md) + "\n")
    print(f"[q7] done -> {out_dir}")


if __name__ == "__main__":
    main()
