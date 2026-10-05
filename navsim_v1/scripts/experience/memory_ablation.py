#!/usr/bin/env python3
"""Q3: memory-size ablation for retrieval-based variants -- NO retraining.

Loads models saved by train_experience.py --save_models, then predicts the
navtest risk/residual per candidate using only a fraction of the memory LOGS
(e.g. 25% / 50% / 100% of the memory split). Writes risk npz files per
fraction into <out_dir>/frac_<f>/<variant>.npz for rerank_eval.py.

Usage:
  python scripts/experience/memory_ablation.py \
      --models_dir $EXP/experience/train_experience_full5_m/models \
      --labels_dir $EXP/experience/navtrain_labels_3k \
      --export_dir $EXP/experience/navtrain_export_3k \
      --navtest_labels_dir $EXP/experience/navtest_labels_full \
      --navtest_export_dir $EXP/experience/navtest_export_full \
      --out_dir $EXP/experience/memory_ablation \
      --variants retrieval pred_desc_retrieval_pp \
      --fractions 0.25 0.5 1.0
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

import train_experience as te  # noqa: E402
from navsim.agents.drive_jepa_perception_based.experience.model import (  # noqa: E402
    ExperienceModel,
)
from navsim.agents.drive_jepa_perception_based.experience.retrieval import (  # noqa: E402
    exp_dim_for,
)
from navsim.agents.drive_jepa_perception_based.experience.knn import (  # noqa: E402
    standardize_apply,
    standardize_fit,
)
from navsim.agents.drive_jepa_perception_based.experience.records import (  # noqa: E402
    load_npz,
    save_npz,
    split_logs_by_name,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models_dir", required=True)
    p.add_argument("--labels_dir", required=True,
                   help="navtrain labels (memory bank source)")
    p.add_argument("--export_dir", required=True)
    p.add_argument("--navtest_labels_dir", required=True)
    p.add_argument("--navtest_export_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--variants", nargs="+", default=["retrieval"])
    p.add_argument("--fractions", type=float, nargs="+",
                   default=[0.25, 0.5, 1.0])
    p.add_argument("--ratios", type=float, nargs=3, default=[0.6, 0.2, 0.2])
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--max_per_scene", type=int, default=4)
    p.add_argument("--dt_enter_thresh", type=float, default=2.0)
    p.add_argument("--device",
                   default="cuda" if __import__("torch").cuda.is_available()
                   else "cpu")
    return p.parse_args()


def subset_by_log_frac(rows: Dict[str, np.ndarray], frac: float,
                       seed: int) -> Dict[str, np.ndarray]:
    """Keep candidates from a deterministic frac of the row logs."""
    logs = np.unique(rows["log"])
    n_keep = max(1, int(round(len(logs) * frac)))
    keep_logs = set(np.random.default_rng(seed).permutation(logs)[:n_keep])
    m = np.array([l in keep_logs for l in rows["log"]])
    return {k: v[m] for k, v in rows.items()}


def main() -> None:
    args = parse_args()
    import torch
    out_dir = Path(args.out_dir).resolve()

    rows = te.load_rows(Path(args.labels_dir), Path(args.export_dir),
                        args.dt_enter_thresh)
    groups = split_logs_by_name(sorted(set(rows["log"].tolist())),
                                ratios=args.ratios, seed=args.split_seed)
    mem_mask = np.isin(rows["log"], list(groups["memory"]))
    qtr_mask = np.array(
        [l in set(groups["query_train"]) for l in rows["log"]])
    zm, zs = standardize_fit(rows["desc"][qtr_mask])
    rows["desc_valid"] = np.isfinite(rows["desc"])
    rows["desc_z"] = np.nan_to_num((rows["desc"] - zm) / zs, nan=0.0)
    rows["cont_cols"] = np.array(
        [te.descriptor_feature_names(te.TIMING_FIELDS).index(n)
         for n in te.DESC_CONT_NAMES], dtype=np.int64)

    mem_rows_full = {k: rows[k][mem_mask] for k in
                     ("x", "y", "scene_id", "log", "desc_z", "desc_valid")}

    meta_probe = torch.load(
        sorted(Path(args.models_dir).glob("*.pt"))[0], map_location="cpu")
    task = meta_probe["meta"].get("task", "risk")
    args.task = task  # predict_navtest reads args.task
    if task == "residual":
        y_res = (rows["final"] - rows["pdm"])[:, None].astype(np.float32)
        rows["y"] = y_res
        mem_rows_full["y"] = y_res[mem_mask]

    nt = te.load_rows(Path(args.navtest_labels_dir),
                      Path(args.navtest_export_dir), args.dt_enter_thresh)
    nt["desc_valid"] = np.isfinite(nt["desc"])
    nt["desc_z"] = np.nan_to_num((nt["desc"] - zm) / zs, nan=0.0)
    nt["cont_cols"] = rows["cont_cols"]
    n_nt = len(nt["tokens"])

    C = meta_probe["meta"]["n_targets"]

    for variant in args.variants:
        seed_preds = {f: [] for f in args.fractions}
        for seed in args.seeds:
            ck = torch.load(
                Path(args.models_dir) / f"{variant}_seed{seed}.pt",
                map_location="cpu")
            meta = ck["meta"]
            latent = meta["latent"]
            exp_dim = exp_dim_for(
                variant, latent,
                n_cont=meta.get("n_cont", len(nt["cont_cols"])))
            model = ExperienceModel(
                meta["feat_in"], latent + exp_dim,
                use_int=meta["use_int"], desc_dim=meta["desc_dim"],
                n_targets=meta["n_targets"]).to(args.device)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            for frac in args.fractions:
                mem_rows = subset_by_log_frac(mem_rows_full, frac, seed=13)
                pred = te.predict_navtest(
                    model, variant, nt, mem_rows, args, seed)
                seed_preds[frac].append(pred.reshape(n_nt, -1, C))
        for frac in args.fractions:
            per_seed = np.stack(seed_preds[frac])
            d = out_dir / f"frac_{frac:g}"
            d.mkdir(parents=True, exist_ok=True)
            save_npz(d / f"{variant}.npz",
                     tokens=nt["tokens"],
                     log_names=nt["scene_log"],
                     risk=per_seed.mean(0).astype(np.float32),
                     risk_per_seed=per_seed.astype(np.float32))
            print(f"[mem] {variant} frac={frac:g} -> {d}", flush=True)
    print(f"[mem] done -> {out_dir}")


if __name__ == "__main__":
    main()
