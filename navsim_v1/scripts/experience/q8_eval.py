#!/usr/bin/env python3
"""Q8 eval: VRU / emerging-agent subsets on navtest (or navtrain) labels.

Subsets (scene-level):
  vru_conflict     — some pedestrian/bicycle track conflicts (descriptor
                     conflict==1) with at least one candidate in the scene
  emerging_conflict — some track with emerging==1 and conflict==1 exists
                      (any agent class)

Reports headroom (oracle vs argmax) and original-vs-variant selections
(argmax(pdm - lam * r_hat)) on each subset: final/NC/TTC/EP + CI +
frac_changed.

Usage:
  python scripts/experience/q8_eval.py \
      --labels_dir $E/navtest_labels_vru \
      --export_dir $E/navtest_export_full \
      --risk_npz retrieval_int=$E/train_experience_full5_m/... .npz ... \
      --lam 0.5 --out_dir $E/q8_eval_out
"""

import argparse
import sys
from pathlib import Path

import numpy as np

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fusion_rerank as fr  # noqa: E402
from navsim.agents.drive_jepa_perception_based.experience.descriptors import (  # noqa: E402
    ACLASS_PEDESTRIAN,
    ACLASS_BICYCLE,
    DESCRIPTOR_FIELDS_EXT,
)
from navsim.agents.drive_jepa_perception_based.experience.records import (  # noqa: E402
    bootstrap_scene_ci,
    fmt_ci,
    load_npz,
)

SUB_NC, SUB_EP, SUB_TTC, SUB_FINAL = 0, 2, 3, 5
FI = {f: i for i, f in enumerate(DESCRIPTOR_FIELDS_EXT)}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--export_dir", required=True)
    p.add_argument("--risk_npz", nargs="+", default=[],
                   help="name=path entries")
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_boot", type=int, default=1000)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--dt_enter_thresh", type=float, default=2.0)
    return p.parse_args()


def scene_subset_masks(labels_dir: Path, tokens):
    """Scan label npz files -> dict of scene-level subset bool masks."""
    n = len(tokens)
    tok_set = set(tokens)
    masks = {
        "vru_conflict": np.zeros(n, dtype=bool),
        "emerging_conflict": np.zeros(n, dtype=bool),
        "emerging_vru_conflict": np.zeros(n, dtype=bool),
    }
    t2i = {t: i for i, t in enumerate(tokens)}
    n_seen = 0
    for p in sorted(labels_dir.glob("*.npz")):
        z = load_npz(p)
        tok = str(z["token"].item())
        if tok not in tok_set:
            continue
        n_seen += 1
        i = t2i[tok]
        if "multi_descriptors" not in z:
            continue
        md = np.asarray(z["multi_descriptors"])  # (K, M, F_ext)
        mm = np.asarray(z["multi_mask"], dtype=bool)
        if not mm.any():
            continue
        cls = md[..., FI["agent_class"]]
        conf = md[..., FI["conflict"]] > 0.5
        emerg = md[..., FI["emerging"]] > 0.5
        is_vru = (cls == ACLASS_PEDESTRIAN) | (cls == ACLASS_BICYCLE)
        conf_real = conf & mm
        masks["vru_conflict"][i] = bool((conf_real & is_vru).any())
        masks["emerging_conflict"][i] = bool((conf_real & emerg).any())
        masks["emerging_vru_conflict"][i] = bool(
            (conf_real & emerg & is_vru).any()
        )
    print(f"[q8] subsets over {n_seen}/{n} labelled scenes: "
          + ", ".join(f"{k}={int(v.sum())}" for k, v in masks.items()))
    return masks


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

    nt = fr.scene_pack(Path(args.labels_dir), Path(args.export_dir),
                       args.dt_enter_thresh)
    S = len(nt["tokens"])
    sel0 = np.argmax(nt["pdm_score"], axis=1)
    oracle = np.argmax(nt["subscores"][..., SUB_FINAL], axis=1)

    masks = scene_subset_masks(Path(args.labels_dir), nt["tokens"].tolist())
    subsets = {"all": np.ones(S, bool), "conflict": nt["subset"][
        np.arange(S), sel0]}
    subsets.update(masks)

    rows_md = []
    results = {}

    def report(name, sel):
        for sname, keep in subsets.items():
            res = eval_subset(sel, nt["subscores"], keep,
                              args.num_boot, args.split_seed, sel0)
            if res is None:
                continue
            results[f"{name}|{sname}"] = res
            for m in ("final", "NC", "TTC", "EP"):
                rows_md.append([name, f"{sname} (n={res['n']})", m,
                                fmt_ci(*res[m]), f"{res['frac']:.3f}"])

    report("original_argmax", sel0)
    report("oracle_best", oracle)
    for entry in args.risk_npz:
        name, path = entry.split("=", 1)
        risk = load_npz(Path(path))
        t2i = {t: i for i, t in enumerate(risk["tokens"].tolist())}
        keep = np.array([i for i, t in enumerate(nt["tokens"])
                         if t in t2i])
        idx = np.array([t2i[t] for t in nt["tokens"] if t in t2i])
        sel = sel0.copy()
        sel[keep] = np.argmax(nt["pdm_score"][keep]
                              - args.lam * r_hat(risk["risk"][idx]), axis=1)
        report(name, sel)

    md = ["# Q8 VRU / emerging-agent subsets", ""]
    md.append(f"scenes: {S} | lam={args.lam}")
    md.append("")
    md.append("| selector | subset | metric | value [95% CI] | frac_changed |")
    md.append("|---|---|---|---|---|")
    for r in rows_md:
        md.append("| " + " | ".join(r) + " |")
    (out_dir / "q8_results.md").write_text("\n".join(md) + "\n")
    print(f"[q8] done -> {out_dir}")


if __name__ == "__main__":
    main()
