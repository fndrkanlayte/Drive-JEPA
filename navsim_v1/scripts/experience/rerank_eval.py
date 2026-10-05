#!/usr/bin/env python3
"""Stage 4: experience-guided re-ranking evaluation.

Re-score each candidate as ``s = pdm_score - lambda * r_hat`` where
``r_hat = 1 - (1 - p_nc) * (1 - p_ttc)`` combines the experience module's two
predicted risk probabilities. ``lambda`` is tuned on the *navtrain query_val*
split (maximising the mean labelled final subscore of the selected candidate)
and then applied to navtest.

For navtest this reports, per risk file (variant), the mean labelled final /
NC / TTC subscore of the re-ranked selection vs the original argmax(pdm_score)
and the oracle best-of-32, on all scenes and the conflict subset, with
scene-level bootstrap 95% CIs.

The official run_pdm_score is NOT applied to re-ranked trajectories: it would
need a thin agent wrapper replaying precomputed selections end-to-end, which is
out of scope for Stage 4 -- we report labelled-subscore outcomes only.

Usage:
  python scripts/experience/rerank_eval.py \
      --labels_dir $EXP/experience/navtrain_labels_3k \
      --export_dir $EXP/experience/navtrain_export_3k \
      --val_risk_dir $EXP/experience/train_experience_out/query_val_risk \
      --navtest_labels_dir $EXP/experience/navtest_labels_smoke \
      --navtest_export_dir $EXP/experience/navtest_export_smoke \
      --navtest_risk_dir $EXP/experience/train_experience_out/navtest_risk \
      --out_dir $EXP/experience/rerank_eval_out
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
    DESCRIPTOR_FIELD_INDEX as FI,
)
from navsim.agents.drive_jepa_perception_based.experience.records import (  # noqa: E402
    bootstrap_scene_ci,
    fmt_ci,
    load_npz,
    split_logs_by_name,
)

SUB_NC, SUB_TTC, SUB_FINAL = 0, 3, 5
TARGET_NAMES = ["nc_unsafe", "ttc_bad"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--labels_dir", required=True,
                   help="navtrain labels (for lambda tuning on query_val)")
    p.add_argument("--export_dir", required=True)
    p.add_argument("--val_risk_dir", required=True,
                   help="<variant>.npz with tokens + risk (S,32,2) on query_val")
    p.add_argument("--navtest_labels_dir", required=True)
    p.add_argument("--navtest_export_dir", required=True)
    p.add_argument("--navtest_risk_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--ratios", type=float, nargs=3, default=[0.6, 0.2, 0.2])
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--num_boot", type=int, default=1000)
    p.add_argument("--dt_enter_thresh", type=float, default=2.0)
    return p.parse_args()


def r_hat(risk: np.ndarray) -> np.ndarray:
    """risk (S,K,2) -> combined risk (S,K): 1 - (1-p_nc)(1-p_ttc)."""
    p = np.clip(risk.astype(np.float64), 0.0, 1.0)
    return 1.0 - (1.0 - p[..., 0]) * (1.0 - p[..., 1])


def scene_pack(labels_dir: Path, export_dir: Path,
               dt_enter_thresh: float) -> Dict[str, np.ndarray]:
    """Per-scene arrays: subscores (S,K,6), pdm_score (S,K), subset mask (S,)."""
    subs, pdms, subsets, tokens, logs = [], [], [], [], []
    for lp in sorted(labels_dir.glob("*.npz")):
        ep = export_dir / lp.name
        if not ep.is_file():
            continue
        lab, exp = load_npz(lp), load_npz(ep)
        subs.append(lab["subscores"].astype(np.float64))
        pdms.append(np.asarray(exp["pdm_score"], dtype=np.float64))
        md = lab.get("main_desc_noatt")
        if md is not None:
            has_main = np.isfinite(md[:, FI["conflict"]])
            conf = md[:, FI["conflict"]] == 1.0
            dte = np.abs(md[:, FI["dt_enter"]])
            subsets.append(has_main & conf & (dte < dt_enter_thresh))
        else:
            subsets.append(np.zeros(lab["subscores"].shape[0], bool))
        tokens.append(str(lab["token"].item()))
        logs.append(str(lab["log_name"].item()))
    return dict(subscores=np.stack(subs), pdm_score=np.stack(pdms),
                subset=np.stack(subsets), tokens=np.asarray(tokens),
                logs=np.asarray(logs))


def select_with_risk(pdm: np.ndarray, risk: np.ndarray, lam: float) -> np.ndarray:
    """argmax over K of pdm_score - lam * r_hat."""
    return np.argmax(pdm - lam * r_hat(risk), axis=1)


def eval_sel(sel: np.ndarray, subscores: np.ndarray,
             scene_subset: np.ndarray, num_boot: int, seed: int) -> Dict[str, tuple]:
    """Mean final/NC/TTC of selected candidates (+ subset), with CIs.

    ``all``: every scene's selected candidate. ``conflict``: scenes where the
    ORIGINAL argmax candidate is in the conflict subset -- the same fixed
    scene set for every selector (same convention as headroom.py).
    """
    S, K = subscores.shape[:2]
    rows = np.arange(S)
    out = {}
    for name, keep in (("all", np.arange(S)),
                       ("conflict", np.flatnonzero(scene_subset))):
        for metric, col in (("final", SUB_FINAL), ("NC", SUB_NC), ("TTC", SUB_TTC)):
            vals = subscores[rows[keep], sel[keep], col]
            out[f"{metric}|{name}"] = bootstrap_scene_ci(vals, num_boot, seed)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- lambda tuning on navtrain query_val -------------------
    tr = scene_pack(Path(args.labels_dir), Path(args.export_dir),
                    args.dt_enter_thresh)
    logs = sorted(set(tr["logs"].tolist()))
    groups = split_logs_by_name(logs, ratios=args.ratios, seed=args.split_seed)
    group_of = {l: g for g, ls in groups.items() for l in ls}
    qva = np.array([group_of[l] == "query_val" for l in tr["logs"]])
    print(f"[rerank] tuning scenes: {int(qva.sum())} query_val of "
          f"{len(tr['tokens'])} navtrain scenes")

    lam_table: List[List[str]] = []
    best: Dict[str, float] = {}
    cv_data: Dict[str, dict] = {}  # variant -> aligned query_val arrays
    for rp in sorted(Path(args.val_risk_dir).glob("*.npz")):
        variant = rp.stem
        risk = load_npz(rp)
        tok2idx = {t: i for i, t in enumerate(tr["tokens"])}
        keep = [tok2idx[t] for t in risk["tokens"] if t in tok2idx]
        sel_rows = np.asarray(keep)
        sel_rows = sel_rows[qva[sel_rows]]
        if len(sel_rows) == 0:
            continue
        risk_s = risk["risk"][[list(risk["tokens"]).index(tr["tokens"][i])
                               for i in sel_rows]]
        pdm_s = tr["pdm_score"][sel_rows]
        sub_s = tr["subscores"][sel_rows]
        cv_data[variant] = {"rows": sel_rows, "risk": risk_s}
        best_lam, best_val = 0.0, -np.inf
        for lam in args.lambdas:
            sel = select_with_risk(pdm_s, risk_s, lam)
            m = float(sub_s[np.arange(len(sel)), sel, SUB_FINAL].mean())
            lam_table.append([variant, f"{lam:g}", f"{m:.4f}"])
            if m > best_val:
                best_val, best_lam = m, lam
        best[variant] = best_lam
        print(f"[rerank] {variant}: best lambda={best_lam:g} "
              f"(mean final {best_val:.4f} on {len(sel_rows)} scenes)")

    # ---------------- query_val rerank: 5-fold CV across logs ---------------
    # per fold: lambda tuned on the other folds, applied to held-out scenes;
    # all held-out selections are pooled for the report (no tuning leakage).
    qva_rows = np.flatnonzero(qva)
    qva_sel0 = np.argmax(tr["pdm_score"][qva_rows], axis=1)
    qva_subset = tr["subset"][qva_rows][np.arange(len(qva_rows)), qva_sel0]
    u_logs = np.unique(tr["logs"][qva_rows])
    fold_of = {l: i % 5 for i, l in enumerate(
        np.random.default_rng(args.split_seed).permutation(u_logs))}
    qva_fold = np.array([fold_of[l] for l in tr["logs"][qva_rows]])

    cv_report: Dict[str, dict] = {}
    qva_rows_report: List[List[str]] = []

    def report_qva(name: str, sel: np.ndarray):
        res = eval_sel(sel, tr["subscores"][qva_rows], qva_subset,
                       args.num_boot, args.split_seed)
        cv_report[name] = res
        for metric in ("final", "NC", "TTC"):
            for sub in ("all", "conflict"):
                qva_rows_report.append(
                    [name, metric, sub, fmt_ci(*res[f"{metric}|{sub}"])])

    report_qva("original_argmax", qva_sel0)
    report_qva("oracle_best",
               np.argmax(tr["subscores"][qva_rows][..., SUB_FINAL], axis=1))
    for variant, d in sorted(cv_data.items()):
        # map this variant's rows to positions inside qva_rows
        pos = np.array(
            [np.flatnonzero(qva_rows == r)[0] for r in d["rows"]])
        sel = np.full(len(d["rows"]), -1)
        for f in range(5):
            trn = qva_fold[pos] != f
            tst = ~trn
            if not tst.any():
                continue
            bl, bv = 0.0, -np.inf
            for lam in args.lambdas:
                s_ = np.argmax(tr["pdm_score"][d["rows"][trn]] -
                               lam * r_hat(d["risk"][trn]), axis=1)
                m = float(tr["subscores"][d["rows"][trn], s_, SUB_FINAL].mean())
                if m > bv:
                    bv, bl = m, lam
            sel[tst] = np.argmax(
                tr["pdm_score"][d["rows"][tst]] -
                bl * r_hat(d["risk"][tst]), axis=1)
        # place the held-out selections back into a qva_rows-length vector,
        # scenes without risk preds keep the original argmax
        sel_full = qva_sel0.copy()
        sel_full[pos] = sel
        report_qva(f"{variant} (CV-lam)", sel_full)

    # ---------------- apply to navtest --------------------------------------
    nt = scene_pack(Path(args.navtest_labels_dir), Path(args.navtest_export_dir),
                    args.dt_enter_thresh)
    S = len(nt["tokens"])
    print(f"[rerank] navtest scenes: {S}")

    rows_report: List[List[str]] = []
    results: Dict[str, dict] = {}

    sel0 = np.argmax(nt["pdm_score"], axis=1)
    # fixed conflict subset = scenes where the ORIGINAL argmax candidate
    # conflicts (same convention as headroom.py)
    scene_subset = nt["subset"][np.arange(S), sel0]

    def report(name: str, sel: np.ndarray):
        res = eval_sel(sel, nt["subscores"], scene_subset, args.num_boot,
                       args.split_seed)
        results[name] = res
        for metric in ("final", "NC", "TTC"):
            for sub in ("all", "conflict"):
                rows_report.append([name, metric, sub,
                                    fmt_ci(*res[f"{metric}|{sub}"])])

    report("original_argmax", sel0)
    oracle = np.argmax(nt["subscores"][..., SUB_FINAL], axis=1)
    report("oracle_best", oracle)

    for rp in sorted(Path(args.navtest_risk_dir).glob("*.npz")):
        variant = rp.stem
        risk = load_npz(rp)
        tok2idx = {t: i for i, t in enumerate(risk["tokens"].tolist())}
        idx = np.array([tok2idx[t] for t in nt["tokens"] if t in tok2idx])
        keep = np.array([i for i, t in enumerate(nt["tokens"]) if t in tok2idx])
        lam = best.get(variant, 1.0)
        sel = np.full(S, -1)
        sel[keep] = select_with_risk(nt["pdm_score"][keep],
                                     risk["risk"][idx], lam)
        # scenes without risk predictions fall back to original argmax
        sel[sel < 0] = sel0[sel < 0]
        report(f"{variant} (lam={lam:g})", sel)

    md = ["# Re-ranking evaluation (Stage 4)", ""]
    md.append(f"navtest scenes: {S} | lambda tuned on navtrain query_val "
              f"({int(qva.sum())} scenes, grid {args.lambdas}) | "
              f"split_seed={args.split_seed}")
    md.append("")
    md.append("## lambda tuning (navtrain query_val, mean final of selected)")
    md.append("| variant | lambda | mean final |")
    md.append("|---|---|---|")
    for r in lam_table:
        md.append("| " + " | ".join(r) + " |")
    md.append("")
    md.append("## query_val re-ranking (5-fold CV over logs, "
              f"{len(qva_rows)} scenes)")
    md.append("| selector | metric | subset | value [95% CI] |")
    md.append("|---|---|---|---|")
    for r in qva_rows_report:
        md.append("| " + " | ".join(r) + " |")
    md.append("")
    md.append("## navtest selected-candidate outcomes (labelled subscores)")
    md.append("| selector | metric | subset | value [95% CI] |")
    md.append("|---|---|---|---|")
    for r in rows_report:
        md.append("| " + " | ".join(r) + " |")
    md.append("")
    md.append("Note: metrics are labelled subscores of the selected "
              "candidate. For official PDMS of a re-ranked policy use "
              "scripts/experience/export_selections.py + run_pdm_score with "
              "agent=replay_selection_agent.")
    (out_dir / "rerank_results.md").write_text("\n".join(md) + "\n")
    with open(out_dir / "rerank_results.json", "w") as f:
        json.dump({"best_lambda": best,
                   "query_val_cv": {k: {m: list(v) for m, v in res.items()
                                        } for k, res in cv_report.items()},
                   "results": {k: {m: list(v) for m, v in res.items()
                                   } for k, res in results.items()}},
                  f, indent=2)
    print(f"[rerank] done -> {out_dir}")


if __name__ == "__main__":
    main()
