#!/usr/bin/env python
"""A5 - The decisive cheap check: does oracle "experience" add signal over the
frozen proposal latent?

Splits scenes BY log_name into memory / query-train / query-val (no log
overlap) and evaluates, per candidate, predictors of NC-unsafety (NC<1) and
TTC-badness (TTC<1):

  1. global_prior      mean label of the memory split
  2. parametric_noexp  logistic regression on proposal_feature (trained on
                       query-train) -- no experience at all
  3. oracle_knn_k{8,32} kNN over TRUE main-vehicle descriptors (+ego speed,
                       itype one-hot, masks) retrieved from memory; same-log
                       neighbours excluded; predicts mean neighbour label
  4. parametric_desc   GBDT on the same TRUE descriptors (trained on
                       query-train) -- does kNN add anything over a parametric
                       model with identical information?
  5. noexp+knn_stats   logistic regression on [p_noexp, knn_mean, knn_var]
  6. random_memory     same as 5 with random (not nearest) memory neighbours

!!! ORACLE NOTE: predictors 3-5 use descriptors computed from the query's TRUE
future (GT replay of other vehicles). They are an ORACLE UPPER BOUND on what
experience retrieval could add, not a deployable predictor.

Outputs: oracle_knn_results.{md,json} with Brier / AUPRC / top-1-risk ranking
and scene-level bootstrap 95% CIs, also on the conflict & |dt_enter|<2s subset.

Usage:
    python scripts/experience/oracle_knn_check.py \
        --labels_dir L --export_dir E --out_dir O [--ratios 0.6 0.2 0.2] [--seed 0]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))

from navsim.agents.drive_jepa_perception_based.experience.descriptors import (  # noqa: E402
    DESCRIPTOR_FIELDS,
    DESCRIPTOR_FIELD_INDEX as FI,
    descriptor_feature_vector,
)
from navsim.agents.drive_jepa_perception_based.experience.knn import (  # noqa: E402
    knn_predict,
    metric_bundle,
    predict_proba_or_prior,
    safe_logreg,
    standardize_apply,
    standardize_fit,
)
from navsim.agents.drive_jepa_perception_based.experience.records import (  # noqa: E402
    bootstrap_scene_ci,
    fmt_ci,
    load_npz,
    split_logs_by_name,
)

SUB_NC, SUB_TTC = 0, 3
TARGETS = {"nc_unsafe": SUB_NC, "ttc_bad": SUB_TTC}  # label = subscore < 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--export_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ratios", type=float, nargs=3, default=[0.6, 0.2, 0.2],
                   help="memory / query_train / query_val ratios over logs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ks", type=int, nargs="+", default=[8, 32])
    p.add_argument("--num_boot", type=int, default=500)
    p.add_argument("--dt_enter_thresh", type=float, default=2.0)
    return p.parse_args()


def load_dataset(labels_dir: Path, export_dir: Path) -> List[dict]:
    """Join label + export records into per-scene dicts."""
    scenes = []
    for lp in sorted(labels_dir.glob("*.npz")):
        ep = export_dir / lp.name
        if not ep.is_file():
            continue
        lab, exp = load_npz(lp), load_npz(ep)
        desc, vmask = lab["descriptors"], lab["vehicle_mask"]
        K = lab["subscores"].shape[0]
        main_desc = desc[:, 0, :]  # (K, F)
        z = np.stack(
            [
                descriptor_feature_vector(
                    {f: main_desc[k, i] for i, f in enumerate(DESCRIPTOR_FIELDS)},
                    ego_speed=float(lab["ego_speed"].item()),
                )
                for k in range(K)
            ]
        )
        scenes.append(
            dict(
                token=str(lab["token"].item()),
                log_name=str(lab["log_name"].item()),
                proposal_feature=np.asarray(exp["proposal_feature"], dtype=np.float32),
                z=z,
                has_main_vehicle=vmask[:, 0],
                conflict=main_desc[:, FI["conflict"]] == 1.0,
                dt_enter=np.abs(main_desc[:, FI["dt_enter"]]),
                subscores=lab["subscores"],
            )
        )
    return scenes


def flatten(scenes: List[dict], target_col: int, dt_enter_thresh: float = 2.0):
    """Flatten scenes to per-candidate arrays."""
    rows = []
    for sid, s in enumerate(scenes):
        K = s["subscores"].shape[0]
        rows.append(
            dict(
                scene_id=np.full(K, sid),
                log=np.array([s["log_name"]] * K, dtype=object),
                feat=s["proposal_feature"],
                z=s["z"],
                y=(s["subscores"][:, target_col] < 1.0).astype(np.float64),
                subset=s["has_main_vehicle"] & s["conflict"] & (s["dt_enter"] < dt_enter_thresh),
            )
        )
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def eval_with_ci(pred, y, scene_ids, num_boot, seed):
    """Metrics + scene-level bootstrap 95% CIs."""
    point = metric_bundle(pred, y, scene_ids)
    scenes = np.unique(scene_ids)
    rng = np.random.default_rng(seed)
    boots = {m: np.empty(num_boot) for m in point}
    for b in range(num_boot):
        keep = rng.choice(scenes, size=len(scenes), replace=True)
        idx = np.concatenate([np.flatnonzero(scene_ids == s) for s in keep])
        sid_map = np.repeat(np.arange(len(keep)), [int((scene_ids == s).sum()) for s in keep])
        for m, v in metric_bundle(pred[idx], y[idx], sid_map).items():
            boots[m][b] = v
    return {
        m: (point[m], float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        for m, v in boots.items()
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = load_dataset(Path(args.labels_dir), Path(args.export_dir))
    print(f"[oracle] {len(scenes)} scenes loaded")
    logs = sorted({s["log_name"] for s in scenes})
    groups = split_logs_by_name(logs, ratios=args.ratios, seed=args.seed)
    group_of = {l: g for g, ls in groups.items() for l in ls}
    print(
        f"[oracle] logs: memory={len(groups['memory'])} "
        f"query_train={len(groups['query_train'])} query_val={len(groups['query_val'])}"
    )
    for s in scenes:
        s["group"] = group_of[s["log_name"]]

    results: Dict[str, dict] = {}
    rng = np.random.default_rng(args.seed)

    for target_name, col in TARGETS.items():
        data = flatten(scenes, col, args.dt_enter_thresh)
        grp = np.array([group_of[l] for l in data["log"]])
        mem = grp == "memory"
        qtr = grp == "query_train"
        qva = grp == "query_val"
        print(f"[oracle:{target_name}] rows mem={mem.sum()} qtrain={qtr.sum()} qval={qva.sum()} "
              f"(unsafe rate mem={data['y'][mem].mean():.3f} qval={data['y'][qva].mean():.3f})")

        # ---- standardization stats on query-train ----
        zm, zs = standardize_fit(data["z"][qtr])
        fm, fs = standardize_fit(data["feat"][qtr])
        z_all = standardize_apply(data["z"], zm, zs)
        f_all = standardize_apply(data["feat"], fm, fs)

        y, sid = data["y"], data["scene_id"]
        log_block = data["log"]
        preds: Dict[str, np.ndarray] = {}
        prior = float(y[mem].mean())

        # 1. global prior
        preds["global_prior"] = np.full(len(y), prior)

        # 2. no-experience parametric
        clf = safe_logreg(f_all[qtr], y[qtr])
        preds["parametric_noexp"] = predict_proba_or_prior(clf, f_all, prior)

        # 3. oracle kNN (k in ks)
        knn_stats = {}
        for k in args.ks:
            mean_, var_ = knn_predict(
                z_all[mem], y[mem], z_all, k,
                pool_block=log_block[mem], query_block=log_block,
            )
            preds[f"oracle_knn_k{k}"] = mean_
            knn_stats[k] = (mean_, var_)

        # 4. parametric on TRUE descriptors (GBDT)
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier

            if len(np.unique(y[qtr])) > 1:
                gb = HistGradientBoostingClassifier(max_iter=200, random_state=args.seed)
                gb.fit(z_all[qtr], y[qtr])
                preds["parametric_desc"] = gb.predict_proba(z_all)[:, 1]
            else:
                preds["parametric_desc"] = np.full(len(y), prior)
        except ImportError:
            clf2 = safe_logreg(z_all[qtr], y[qtr])
            preds["parametric_desc"] = predict_proba_or_prior(clf2, z_all, prior)

        # 5. no-exp parametric + oracle-kNN features
        k_main = args.ks[-1]
        km, kv = knn_stats[k_main]
        stack = np.stack([preds["parametric_noexp"], km, kv], axis=1)
        stack = np.nan_to_num(stack, nan=prior)
        clf3 = safe_logreg(stack[qtr], y[qtr])
        preds["noexp_plus_knn"] = predict_proba_or_prior(clf3, stack, prior)

        # 6. random memory (same stack, random neighbours)
        rm, rv = knn_predict(
            z_all[mem], y[mem], z_all, k_main,
            pool_block=log_block[mem], query_block=log_block, rng=rng,
        )
        stack_r = np.nan_to_num(np.stack([preds["parametric_noexp"], rm, rv], axis=1), nan=prior)
        clf4 = safe_logreg(stack_r[qtr], y[qtr])
        preds["random_memory"] = predict_proba_or_prior(clf4, stack_r, prior)

        # ---- evaluate on query-val ----
        sub = data["subset"]
        target_res = {}
        for name, pred in preds.items():
            target_res[name] = {
                "all": eval_with_ci(pred[qva], y[qva], sid[qva], args.num_boot, args.seed),
                "conflict_subset": eval_with_ci(
                    pred[qva & sub], y[qva & sub], sid[qva & sub], args.num_boot, args.seed
                ),
            }
        results[target_name] = target_res

    # ---------------- markdown + json output -----------------------------------
    header = (
        f"# Oracle kNN check\n\n"
        f"**ORACLE UPPER BOUND**: predictors marked `oracle`/`knn` use descriptors computed "
        f"from the query's TRUE future (GT replay of other vehicles). This bounds what "
        f"experience retrieval could add; it is not deployable.\n\n"
        f"Scenes: {len(scenes)} | logs: memory={len(groups['memory'])}, "
        f"query_train={len(groups['query_train'])}, query_val={len(groups['query_val'])} "
        f"| seed={args.seed}\n"
    )
    md = [header]
    jsonable = {"note": "oracle upper bound", "scenes": len(scenes),
                "groups": {k: len(v) for k, v in groups.items()}, "targets": results}
    for target_name, target_res in results.items():
        for subset_name in ("all", "conflict_subset"):
            md.append(f"\n## {target_name} — {subset_name}\n")
            md.append("| predictor | brier | AUPRC | top-1-risk hit |")
            md.append("|---|---|---|---|")
            for name, res in target_res.items():
                cell = res[subset_name]
                md.append(
                    f"| {name} | {fmt_ci(*cell['brier'])} | {fmt_ci(*cell['auprc'])} | "
                    f"{fmt_ci(*cell['top1_risk'])} |"
                )
    text = "\n".join(md)
    (out_dir / "oracle_knn_results.md").write_text(text)
    with open(out_dir / "oracle_knn_results.json", "w") as f:
        json.dump(jsonable, f, indent=2)
    print(text)
    print(f"\n[oracle] wrote {out_dir}/oracle_knn_results.{{md,json}}")


if __name__ == "__main__":
    main()
