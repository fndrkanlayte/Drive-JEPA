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
    KNN_NUMERIC_FIELDS,
    TIMING_FIELDS,
    descriptor_feature_names,
    descriptor_feature_vector,
)
from navsim.agents.drive_jepa_perception_based.experience.knn import (  # noqa: E402
    eval_with_ci,
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
    p.add_argument("--main_vehicle", choices=["noatt", "att"], default="noatt",
                   help="which stored main vehicle to describe: 'noatt' (default) "
                        "uses main_desc_noatt, the attribution-free ranking "
                        "(deployment-feasible, leakage-safe); 'att' uses the "
                        "GT-attributed top-1 of descriptors[:,0]")
    p.add_argument("--feature_set", choices=["timing", "full"], default="timing",
                   help="'timing' (default) excludes min_dist/overlap, which "
                        "nearly encode the collision label; 'full' keeps all "
                        "KNN_NUMERIC_FIELDS")
    return p.parse_args()


def load_dataset(labels_dir: Path, export_dir: Path, main_vehicle: str, fields: List[str]) -> List[dict]:
    """Join label + export records into per-scene dicts."""
    scenes = []
    missing_noatt = 0
    for lp in sorted(labels_dir.glob("*.npz")):
        ep = export_dir / lp.name
        if not ep.is_file():
            continue
        lab, exp = load_npz(lp), load_npz(ep)
        desc, vmask = lab["descriptors"], lab["vehicle_mask"]
        K = lab["subscores"].shape[0]
        if main_vehicle == "noatt" and "main_desc_noatt" in lab:
            main_desc = lab["main_desc_noatt"]  # (K, F)
            has_main = np.isfinite(main_desc[:, FI["conflict"]])
        else:
            missing_noatt += int(main_vehicle == "noatt")
            main_desc = desc[:, 0, :]  # (K, F)
            has_main = vmask[:, 0]
        z = np.stack(
            [
                descriptor_feature_vector(
                    {f: main_desc[k, i] for i, f in enumerate(DESCRIPTOR_FIELDS)},
                    ego_speed=float(lab["ego_speed"].item()),
                    fields=fields,
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
                has_main_vehicle=has_main,
                conflict=main_desc[:, FI["conflict"]] == 1.0,
                dt_enter=np.abs(main_desc[:, FI["dt_enter"]]),
                subscores=lab["subscores"],
            )
        )
    if missing_noatt:
        raise ValueError(
            f"{missing_noatt} label file(s) lack main_desc_noatt, so "
            f"--main_vehicle noatt cannot be honored without leaking GT "
            f"attribution. Re-run label_candidates (which now writes both "
            f"orderings), or pass --main_vehicle att explicitly."
        )
    return scenes


def _desc_fit_predict(x_tr, y_tr, x_va, prior: float, seed: int) -> np.ndarray:
    """GBDT (logreg fallback) descriptor-model proba for x_va."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        if len(np.unique(y_tr)) < 2:
            return np.full(len(x_va), prior)
        gb = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        gb.fit(x_tr, y_tr)
        return gb.predict_proba(x_va)[:, 1]
    except ImportError:
        return predict_proba_or_prior(safe_logreg(x_tr, y_tr), x_va, prior)


def oof_predict(fit_predict, x, y, mask, log, seed, n_folds=5) -> np.ndarray:
    """Out-of-fold proba for rows under `mask`, split into n_folds BY LOG.

    Any stacked feature that is itself trained on query_train must enter the
    stacker via out-of-fold predictions, or the meta-model just memorizes.
    """
    idx = np.flatnonzero(mask)
    logs = np.unique(log[idx])
    fold_of = {l: i % n_folds for i, l in enumerate(
        np.random.default_rng(seed).permutation(logs))}
    fold_row = np.array([fold_of[l] for l in log[idx]])
    oof = np.full(len(idx), np.nan)
    for f in range(n_folds):
        tr = idx[fold_row != f]
        va = fold_row == f
        if not va.any():
            continue
        if len(np.unique(y[tr])) < 2:
            oof[va] = y[tr].mean() if len(tr) else y[mask].mean()
        else:
            oof[va] = fit_predict(x[tr], y[tr], x[idx[va]])
    return oof


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



def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = TIMING_FIELDS if args.feature_set == "timing" else KNN_NUMERIC_FIELDS
    scenes = load_dataset(Path(args.labels_dir), Path(args.export_dir),
                          args.main_vehicle, fields)
    print(f"[oracle] {len(scenes)} scenes loaded "
          f"(main_vehicle={args.main_vehicle}, feature_set={args.feature_set})")
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
    label_stats: Dict[str, dict] = {}
    rng = np.random.default_rng(args.seed)

    for target_name, col in TARGETS.items():
        data = flatten(scenes, col, args.dt_enter_thresh)
        grp = np.array([group_of[l] for l in data["log"]])
        mem = grp == "memory"
        qtr = grp == "query_train"
        qva = grp == "query_val"
        # scenes in query_val with at least one positive (unsafe) candidate
        qva_scenes = np.unique(data["scene_id"][qva])
        scenes_with_pos = int(
            sum(data["y"][(data["scene_id"] == s) & qva].any() for s in qva_scenes)
        )
        label_stats[target_name] = {
            "pos_rate": {
                "memory": float(data["y"][mem].mean()),
                "query_train": float(data["y"][qtr].mean()),
                "query_val": float(data["y"][qva].mean()),
            },
            "qval_scenes_with_pos": scenes_with_pos,
            "qval_scenes": int(len(qva_scenes)),
        }
        print(f"[oracle:{target_name}] rows mem={mem.sum()} qtrain={qtr.sum()} qval={qva.sum()} "
              f"| pos-rate mem={data['y'][mem].mean():.3f} qtrain={data['y'][qtr].mean():.3f} "
              f"qval={data['y'][qva].mean():.3f} "
              f"| qval scenes w/ pos: {scenes_with_pos}/{len(qva_scenes)}")

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
        preds["parametric_desc"] = _desc_fit_predict(
            z_all[qtr], y[qtr], z_all, prior, args.seed)

        # out-of-fold versions (5-fold by log) for stacking: every stacked
        # feature trained on query_train enters the meta-model via OOF preds
        oof_noexp = oof_predict(
            lambda a, b, c: predict_proba_or_prior(safe_logreg(a, b), c, prior),
            f_all, y, qtr, log_block, args.seed)
        oof_desc = oof_predict(
            lambda a, b, c: _desc_fit_predict(a, b, c, prior, args.seed),
            z_all, y, qtr, log_block, args.seed)
        p_noexp_stack = preds["parametric_noexp"].copy()
        p_noexp_stack[qtr] = oof_noexp
        p_desc_stack = preds["parametric_desc"].copy()
        p_desc_stack[qtr] = oof_desc

        # 5. no-exp parametric + oracle-kNN features
        k_main = args.ks[-1]
        km, kv = knn_stats[k_main]
        stack = np.stack([p_noexp_stack, km, kv], axis=1)
        stack = np.nan_to_num(stack, nan=prior)
        clf3 = safe_logreg(stack[qtr], y[qtr])
        preds["noexp_plus_knn"] = predict_proba_or_prior(clf3, stack, prior)

        # 6. random memory (same stack, random neighbours)
        rm, rv = knn_predict(
            z_all[mem], y[mem], z_all, k_main,
            pool_block=log_block[mem], query_block=log_block, rng=rng,
        )
        stack_r = np.nan_to_num(np.stack([p_noexp_stack, rm, rv], axis=1), nan=prior)
        clf4 = safe_logreg(stack_r[qtr], y[qtr])
        preds["random_memory"] = predict_proba_or_prior(clf4, stack_r, prior)

        # 7. noexp + parametric_desc stack: the same-info parametric
        #    counterpart of noexp_plus_knn
        stack_d = np.nan_to_num(
            np.stack([p_noexp_stack, p_desc_stack], axis=1), nan=prior)
        clf5 = safe_logreg(stack_d[qtr], y[qtr])
        preds["noexp_plus_desc"] = predict_proba_or_prior(clf5, stack_d, prior)

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
        f"| seed={args.seed}\n\n"
        f"main_vehicle=`{args.main_vehicle}` | feature_set=`{args.feature_set}` "
        f"({', '.join(descriptor_feature_names(fields))})\n\n"
        f"Label stats:\n" + "\n".join(
            f"- {t}: pos-rate mem={s['pos_rate']['memory']:.3f} "
            f"qtrain={s['pos_rate']['query_train']:.3f} "
            f"qval={s['pos_rate']['query_val']:.3f}; "
            f"qval scenes with a positive: {s['qval_scenes_with_pos']}/{s['qval_scenes']}"
            for t, s in label_stats.items()
        ) + "\n"
    )
    md = [header]
    jsonable = {"note": "oracle upper bound", "scenes": len(scenes),
                "groups": {k: len(v) for k, v in groups.items()},
                "main_vehicle": args.main_vehicle, "feature_set": args.feature_set,
                "feature_names": descriptor_feature_names(fields),
                "label_stats": label_stats, "targets": results}
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
