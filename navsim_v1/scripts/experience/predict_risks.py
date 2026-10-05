#!/usr/bin/env python3
"""Predict-only risk/residual export from saved models (skips metric eval).

Loads checkpoints written by train_experience.py --save_models and produces
the same <variant>.npz files it would have written into
query_val_risk/ and navtest_risk/ (risk + optional conf fields), without
re-running the expensive eval_with_ci table. Used to finish a run whose
metrics loop is redundant.

Usage:
  python scripts/experience/predict_risks.py \
      --models_dir $EXP/experience/train_experience_full5_m/models \
      --labels_dir $EXP/experience/navtrain_labels_3k \
      --export_dir $EXP/experience/navtrain_export_3k \
      --navtest_labels_dir $EXP/experience/navtest_labels_full \
      --navtest_export_dir $EXP/experience/navtest_export_full \
      --out_dir $EXP/experience/train_experience_full5_m \
      --variants retrieval
"""

import argparse
import sys
from pathlib import Path

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
    standardize_fit,
)
from navsim.agents.drive_jepa_perception_based.experience.records import (  # noqa: E402
    save_npz,
    split_logs_by_name,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models_dir", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--export_dir", required=True)
    p.add_argument("--navtest_labels_dir", required=True)
    p.add_argument("--navtest_export_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--variants", nargs="+", required=True)
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


def load_model(ckpt_path, args):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu")
    meta = ck["meta"]
    exp_dim = exp_dim_for(meta["variant"], meta["latent"],
                          n_cont=meta.get("n_cont", 7))
    model = ExperienceModel(
        meta["feat_in"], meta["latent"] + exp_dim,
        use_int=meta["use_int"], desc_dim=meta["desc_dim"],
        n_targets=meta["n_targets"]).to(args.device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, meta


def main() -> None:
    args = parse_args()
    import torch
    out_dir = Path(args.out_dir).resolve()

    rows = te.load_rows(Path(args.labels_dir), Path(args.export_dir),
                        args.dt_enter_thresh)
    groups = split_logs_by_name(sorted(set(rows["log"].tolist())),
                                ratios=args.ratios, seed=args.split_seed)
    group_of = {l: g for g, ls in groups.items() for l in ls}
    rows["group"] = np.array([group_of[l] for l in rows["log"]])
    qtr = rows["group"] == "query_train"
    zm, zs = standardize_fit(rows["desc"][qtr])
    rows["desc_valid"] = np.isfinite(rows["desc"])
    rows["desc_z"] = np.nan_to_num((rows["desc"] - zm) / zs,
                                  nan=0.0).astype(np.float32)
    names = te.descriptor_feature_names(te.TIMING_FIELDS)
    rows["cont_cols"] = np.array(
        [names.index(n) for n in te.DESC_CONT_NAMES], dtype=np.int64)
    mem_mask = rows["group"] == "memory"
    mem_rows = {k: rows[k][mem_mask] for k in
                ("x", "y", "scene_id", "log", "desc_z", "desc_valid")}
    val_mask = rows["group"] == "query_val"

    # task setup identical to train_experience.main
    meta_probe = torch.load(
        sorted(Path(args.models_dir).glob("*.pt"))[0], map_location="cpu")
    task = meta_probe["meta"].get("task", "risk")
    args.task = task
    C = meta_probe["meta"]["n_targets"]
    if task == "residual":
        rows["y"] = (rows["final"] - rows["pdm"])[:, None].astype(np.float32)
        mem_rows["y"] = rows["y"][mem_mask]

    nt = te.load_rows(Path(args.navtest_labels_dir),
                      Path(args.navtest_export_dir), args.dt_enter_thresh)
    nt["desc_valid"] = np.isfinite(nt["desc"])
    nt["desc_z"] = np.nan_to_num((nt["desc"] - zm) / zs,
                                nan=0.0).astype(np.float32)
    nt["cont_cols"] = rows["cont_cols"]

    # val rows in scene order (same as main(): scene_id-sorted via load_rows)
    val_sid = rows["scene_id"][val_mask]
    val_scene_ids = np.unique(val_sid)
    val_scene_tokens = rows["tokens"][val_scene_ids]
    x_val = torch.from_numpy(rows["x"][val_mask]).float()
    x_nt = torch.from_numpy(nt["x"]).float()
    n_nt = len(nt["tokens"])
    qva_dir = out_dir / "query_val_risk"
    risk_dir = out_dir / "navtest_risk"
    qva_dir.mkdir(parents=True, exist_ok=True)
    risk_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        real = te.BASE.get(variant, variant)
        val_preds, nt_preds, val_confs, nt_confs = [], [], [], []
        for seed in args.seeds:
            model, meta = load_model(
                Path(args.models_dir) / f"{real}_seed{seed}.pt", args)
            # query_val predictions (identical to training's val slice:
            # retrieval only reaches into mem_rows, not other query rows)
            rng = np.random.default_rng(seed + 500)
            if real.startswith("pred_desc"):
                d_hat_v = te.desc_rows(model, x_val, device=args.device)
                if real == "pred_desc_retrieval_pp":
                    mem_desc = te.desc_rows(
                        model, torch.from_numpy(mem_rows["x"]).float(),
                        device=args.device)
                    mem_valid = np.ones_like(mem_desc, dtype=bool)
                else:
                    mem_desc = mem_rows["desc_z"]
                    mem_valid = mem_rows["desc_valid"]
                exp_v = te.desc_retrieval_features(
                    d_hat_v, rows["scene_id"][val_mask],
                    rows["log"][val_mask], mem_desc, mem_rows["y"],
                    mem_rows["scene_id"], mem_rows["log"], rows["cont_cols"],
                    mem_valid, topk=args.topk,
                    max_per_scene=args.max_per_scene, device=args.device)
            else:
                mem = {"y": mem_rows["y"], "scene_id": mem_rows["scene_id"],
                       "log": mem_rows["log"],
                       "latent": te.encode_rows(
                           model, torch.from_numpy(mem_rows["x"]).float(),
                           device=args.device)}
                q_lat = te.encode_rows(model, x_val, device=args.device)
                exp_v = te.build_exp(
                    real, q_lat, rows["scene_id"][val_mask],
                    rows["log"][val_mask], mem, args, rng)
            sig = task != "residual"
            vp = te.predict(model, x_val, exp_v, args.device, sigmoid=sig)
            if variant == "shuffle":
                exp_sh = te.build_exp(
                    real, q_lat, rows["scene_id"][val_mask],
                    rows["log"][val_mask], mem, args,
                    np.random.default_rng(seed + 778), shuffle_labels=True)
                vp = te.predict(model, x_val, exp_sh, args.device,
                                sigmoid=sig)
            val_preds.append(vp)
            if variant == "shuffle":
                continue  # eval-only variant: no navtest npz
            np_ = te.predict_navtest(model, real, nt, mem_rows, args, seed)
            nt_preds.append(np_.reshape(n_nt, -1, C))
            for coll, x_ in ((val_confs, x_val), (nt_confs, x_nt)):
                c = te.conf_rows(model, x_, args.device, conf_col=0,
                                 zm_c=float(zm[0]), zs_c=float(zs[0]))
                coll.append(c)
        save_npz(
            qva_dir / f"{variant}.npz",
            tokens=val_scene_tokens,
            risk=np.stack(val_preds).mean(0).reshape(
                len(val_scene_ids), -1, C).astype(np.float32),
            **({} if all(c is None for c in val_confs)
               else {"conf": np.stack([
                       c if c is not None
                       else np.full(len(val_confs[0]), np.nan)
                       for c in val_confs]).mean(0).reshape(
                           len(val_scene_ids), -1).astype(np.float32)}))
        if variant == "shuffle":
            print(f"[pred] {variant} done (qval only)", flush=True)
            continue
        save_npz(
            risk_dir / f"{variant}.npz",
            tokens=nt["tokens"],
            log_names=nt["scene_log"],
            risk=np.stack(nt_preds).mean(0).astype(np.float32),
            risk_per_seed=np.stack(nt_preds).astype(np.float32),
            **({} if all(c is None for c in nt_confs)
               else {"conf": np.stack([
                       c if c is not None
                       else np.full(len(nt_confs[0]), np.nan)
                       for c in nt_confs]).mean(0).reshape(
                           n_nt, -1).astype(np.float32)}))
        print(f"[pred] {variant} done -> {risk_dir}/{variant}.npz", flush=True)
    print(f"[pred] all done -> {out_dir}")


if __name__ == "__main__":
    main()
