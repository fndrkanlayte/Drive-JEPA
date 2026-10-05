#!/usr/bin/env python3
"""Q5: PDMS-structured fusion + trust-region re-ranking (no new training).

Reuses the trained risk npz files plus the exported ``pred_logit`` (K,6)
from each scene. Column order of both pred_logit and labelled subscores:
[NC, DAC, EP, TTC, Comfort, final] (see compute_navsim_score.get_scores).
The deployed checkpoint has double_score=False, so
``pdm_score = sigmoid(pred_logit[..., 5])``.

Q5a -- PDMS-structured fusion. Replace only the NC and TTC subscore
probabilities with experience-corrected ones, keep Drive-JEPA's own
DAC/EP/comfort predictions, and recompute the score with the PDMS formula::

    p_nc'  = (1 - a) * sig(pred_nc)  + a * (1 - r_nc)
    p_ttc' = (1 - a) * sig(pred_ttc) + a * (1 - r_ttc)
    s      = p_nc' * sig(pred_dac)
             * (5 * sig(pred_ep) + 5 * p_ttc' + 2 * sig(pred_comf)) / 12

Selection = argmax(s). We also report how often the a=0 fused argmax equals
the original argmax(pdm_score).

Q5b -- trust region. Only allow switching to candidates in the top-k of the
original pdm_score (k in {2,3,5,8}) or within margin delta of its max.
Optional EP guard: a switch candidate must satisfy
sig(pred_ep)_cand >= sig(pred_ep)_orig - eps.

Params are tuned on navtrain query_val with 5-fold log-level CV (per-fold
tuning on the remaining folds) for the query_val table, and on ALL of
query_val for the navtest application (same convention as rerank_eval.py).

Usage:
  python scripts/experience/fusion_rerank.py \
      --labels_dir $EXP/experience/navtrain_labels_3k \
      --export_dir $EXP/experience/navtrain_export_3k \
      --val_risk_dir $EXP/experience/train_experience_full5_m/query_val_risk \
      --navtest_labels_dir $EXP/experience/navtest_labels_full \
      --navtest_export_dir $EXP/experience/navtest_export_full \
      --navtest_risk_dir $EXP/experience/train_experience_full5_m/navtest_risk \
      --out_dir $EXP/experience/fusion_rerank_out
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

SUB_NC, SUB_EP, SUB_TTC, SUB_FINAL = 0, 2, 3, 5
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
TOPKS = [2, 3, 5, 8]
DELTAS = [0.005, 0.01, 0.02, 0.05]
EPS = [None, 0.0, 0.05]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--export_dir", required=True)
    p.add_argument("--val_risk_dir", required=True)
    p.add_argument("--navtest_labels_dir", required=True)
    p.add_argument("--navtest_export_dir", required=True)
    p.add_argument("--navtest_risk_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ratios", type=float, nargs=3, default=[0.6, 0.2, 0.2])
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--num_boot", type=int, default=1000)
    p.add_argument("--dt_enter_thresh", type=float, default=2.0)
    p.add_argument("--variants", nargs="*",
                   default=["noexp_int", "retrieval_int",
                            "pred_desc_retrieval", "pred_desc_retrieval_pp",
                            "random"])
    return p.parse_args()


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def scene_pack(labels_dir: Path, export_dir: Path,
               dt_enter_thresh: float) -> Dict[str, np.ndarray]:
    """Per-scene arrays: subscores, pdm_score, pred_logit, subset mask."""
    subs, pdms, logits, subsets, tokens, logs = [], [], [], [], [], []
    for lp in sorted(labels_dir.glob("*.npz")):
        ep = export_dir / lp.name
        if not ep.is_file():
            continue
        lab, exp = load_npz(lp), load_npz(ep)
        subs.append(lab["subscores"].astype(np.float64))
        pdms.append(np.asarray(exp["pdm_score"], dtype=np.float64))
        logits.append(np.asarray(exp["pred_logit"], dtype=np.float64))
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
                pred_logit=np.stack(logits), subset=np.stack(subsets),
                tokens=np.asarray(tokens), logs=np.asarray(logs))


def fused_score(pred_logit: np.ndarray, risk: np.ndarray,
                alpha: float) -> np.ndarray:
    """(S,K,6) logits + (S,K,2) risk -> fused PDMS score (S,K)."""
    p = sigmoid(pred_logit)
    p_nc = (1 - alpha) * p[..., 0] + alpha * (1.0 - risk[..., 0])
    p_ttc = (1 - alpha) * p[..., 3] + alpha * (1.0 - risk[..., 1])
    return p_nc * p[..., 1] * (5.0 * p[..., 2] + 5.0 * p_ttc
                             + 2.0 * p[..., 4]) / 12.0


def region_mask(pdm: np.ndarray, region: Tuple[str, float]) -> np.ndarray:
    """Allowed-candidate mask (S,K). region: ('all',0), ('topk',k),
    ('margin',delta). The original argmax is always allowed."""
    S, K = pdm.shape
    kind, val = region
    if kind == "all":
        return np.ones((S, K), bool)
    if kind == "topk":
        order = np.argsort(-pdm, axis=1)
        rank = np.empty_like(order)
        rank[np.arange(S)[:, None], order] = np.arange(K)[None, :]
        return rank < int(val)
    thr = pdm.max(axis=1, keepdims=True) - float(val)
    return pdm >= thr


def ep_guard_mask(pred_logit: np.ndarray, sel0: np.ndarray,
                  eps: Optional[float]) -> np.ndarray:
    """(S,K) bool: candidate's predicted EP >= original's predicted EP - eps."""
    S, K = pred_logit.shape[:2]
    if eps is None:
        return np.ones((S, K), bool)
    p_ep = sigmoid(pred_logit)[..., SUB_EP]
    return p_ep >= (p_ep[np.arange(S), sel0][:, None] - eps)


def fused_select(pred_logit: np.ndarray, risk: np.ndarray, alpha: float,
                 pdm: np.ndarray, region: Tuple[str, float],
                 eps: Optional[float], sel0: np.ndarray) -> np.ndarray:
    s = fused_score(pred_logit, risk, alpha)
    allowed = region_mask(pdm, region) & ep_guard_mask(pred_logit, sel0, eps)
    allowed[np.arange(len(sel0)), sel0] = True
    s = np.where(allowed, s, -np.inf)
    return np.argmax(s, axis=1)


def eval_sel(sel, subscores, scene_subset, num_boot, seed):
    S = subscores.shape[0]
    rows = np.arange(S)
    out = {}
    for name, keep in (("all", np.arange(S)),
                       ("conflict", np.flatnonzero(scene_subset))):
        for metric, col in (("final", SUB_FINAL), ("NC", SUB_NC),
                            ("TTC", SUB_TTC), ("EP", SUB_EP)):
            vals = subscores[rows[keep], sel[keep], col]
            out[f"{metric}|{name}"] = bootstrap_scene_ci(vals, num_boot, seed)
    return out


def load_risk_aligned(risk_dir: Path, tokens: np.ndarray,
                      variants: List[str]):
    """variant -> (rows_into_scene_pack, risk(S',K,2)) aligned by token."""
    out = {}
    for rp in sorted(Path(risk_dir).glob("*.npz")):
        if rp.stem not in variants:
            continue
        risk = load_npz(rp)
        if risk["risk"].shape[-1] != 2:
            continue  # residual npz: skip
        tok2idx = {t: i for i, t in enumerate(risk["tokens"].tolist())}
        keep_pack = np.array([i for i, t in enumerate(tokens)
                              if t in tok2idx])
        keep_risk = np.array([tok2idx[t] for t in tokens
                              if t in tok2idx])
        out[rp.stem] = (keep_pack, risk["risk"][keep_risk])
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = scene_pack(Path(args.labels_dir), Path(args.export_dir),
                    args.dt_enter_thresh)
    groups = split_logs_by_name(sorted(set(tr["logs"].tolist())),
                                ratios=args.ratios, seed=args.split_seed)
    group_of = {l: g for g, ls in groups.items() for l in ls}
    qva = np.array([group_of[l] == "query_val" for l in tr["logs"]])
    qva_rows = np.flatnonzero(qva)
    print(f"[fusion] tuning scenes: {len(qva_rows)} query_val of "
          f"{len(tr['tokens'])} navtrain scenes")

    qv_risk = load_risk_aligned(Path(args.val_risk_dir), tr["tokens"],
                                args.variants)
    # restrict aligned risk rows to query_val scenes
    for v, (pk, rk) in list(qv_risk.items()):
        m = np.isin(pk, qva_rows)
        qv_risk[v] = (pk[m], rk[m])

    sel0_qv = np.argmax(tr["pdm_score"][qva_rows], axis=1)
    # ---- alpha=0 fidelity check ------------------------------------------------
    s0_qv = fused_score(tr["pred_logit"][qva_rows],
                        np.zeros((len(qva_rows),
                                  tr["pred_logit"].shape[1], 2)), 0.0)
    a0_agree = float((np.argmax(s0_qv, axis=1) == sel0_qv).mean())
    print(f"[fusion] alpha=0 fused argmax == original argmax on "
          f"query_val: {a0_agree:.3f}")

    # region grid: unrestricted, topk, margin
    regions = [("all", 0.0)] + [("topk", float(k)) for k in TOPKS] \
        + [("margin", float(d)) for d in DELTAS]

    def tune_eval(rows_local, risk_map_local, grid_fusion_only=False):
        """Return dict variant -> best (alpha, region, eps, mean_final)."""
        sub = tr["subscores"][rows_local]
        pdm = tr["pdm_score"][rows_local]
        logit = tr["pred_logit"][rows_local]
        sel0 = np.argmax(pdm, axis=1)
        best = {}
        for v, (pk_all, rk_all) in risk_map_local.items():
            pos = np.array([np.flatnonzero(rows_local == r)[0]
                            for r in pk_all])
            if len(pos) == 0:
                continue
            bv, bp = -np.inf, (0.0, ("all", 0.0), None)
            reg_list = regions[:1] if grid_fusion_only else regions
            for a in ALPHAS:
                for reg in reg_list:
                    for eps in EPS:
                        sel = fused_select(logit[pos], rk_all, a,
                                           pdm[pos], reg, eps, sel0[pos])
                        m = float(sub[pos, sel, SUB_FINAL].mean())
                        if m > bv:
                            bv, bp = m, (a, reg, eps)
            best[v] = (bp, bv)
        return best

    # ---- query_val: 5-fold log CV --------------------------------------------
    u_logs = np.unique(tr["logs"][qva_rows])
    fold_of = {l: i % 5 for i, l in enumerate(
        np.random.default_rng(args.split_seed).permutation(u_logs))}
    qva_fold = np.array([fold_of[l] for l in tr["logs"][qva_rows]])
    qv_subset = tr["subset"][qva_rows][np.arange(len(qva_rows)), sel0_qv]

    cv_report: Dict[str, dict] = {}
    md_rows: List[List[str]] = []

    def report_cv(name, sel):
        res = eval_sel(sel, tr["subscores"][qva_rows], qv_subset,
                       args.num_boot, args.split_seed)
        res["frac_changed"] = float((sel != sel0_qv).mean())
        cv_report[name] = res
        for metric in ("final", "NC", "TTC", "EP"):
            for subn in ("all", "conflict"):
                md_rows.append([name, metric, subn,
                                fmt_ci(*res[f"{metric}|{subn}"]),
                                f"{res['frac_changed']:.3f}"])

    report_cv("original_argmax", sel0_qv)
    report_cv("oracle_best",
              np.argmax(tr["subscores"][qva_rows][..., SUB_FINAL], axis=1))
    # Q5a (fusion only) and Q5b (fusion + region + guard), CV-tuned
    for v, (pk, rk) in sorted(qv_risk.items()):
        pos_all = np.array([np.flatnonzero(qva_rows == r)[0]
                            for r in pk])
        for mode, fusion_only in (("Q5a", True), ("Q5b", False)):
            sel = sel0_qv.copy()
            for f in range(5):
                trn = qva_fold != f
                tst = ~trn
                if not tst.any():
                    continue
                rows_trn, rows_tst = qva_rows[trn], qva_rows[tst]
                sub_t = tr["subscores"][rows_trn]
                pdm_t = tr["pdm_score"][rows_trn]
                log_t = tr["pred_logit"][rows_trn]
                sel0_t = np.argmax(pdm_t, axis=1)
                pos_trn = np.array([np.flatnonzero(rows_trn == r)[0]
                                    for r in pk if r in set(rows_trn.tolist())])
                rk_trn = rk[[i for i, r in enumerate(pk)
                             if r in set(rows_trn.tolist())]]
                bv, bp = -np.inf, (0.0, ("all", 0.0), None)
                for a in ALPHAS:
                    for reg in (regions[:1] if fusion_only else regions):
                        for eps in EPS:
                            s_ = fused_select(log_t[pos_trn], rk_trn, a,
                                              pdm_t[pos_trn], reg, eps,
                                              sel0_t[pos_trn])
                            m = float(sub_t[pos_trn, s_, SUB_FINAL].mean())
                            if m > bv:
                                bv, bp = m, (a, reg, eps)
                a_, reg_, eps_ = bp
                pos_tst = np.array([np.flatnonzero(qva_rows == r)[0]
                                    for r in pk if r in set(rows_tst.tolist())])
                rk_tst = rk[[i for i, r in enumerate(pk)
                             if r in set(rows_tst.tolist())]]
                sel[pos_tst] = fused_select(
                    tr["pred_logit"][qva_rows][pos_tst], rk_tst, a_,
                    tr["pdm_score"][qva_rows][pos_tst], reg_, eps_,
                    sel0_qv[pos_tst])
            report_cv(f"{v} ({mode})", sel)

    # ---- navtest --------------------------------------------------------------
    nt = scene_pack(Path(args.navtest_labels_dir),
                    Path(args.navtest_export_dir), args.dt_enter_thresh)
    S = len(nt["tokens"])
    sel0 = np.argmax(nt["pdm_score"], axis=1)
    scene_subset = nt["subset"][np.arange(S), sel0]
    s0_nt = fused_score(nt["pred_logit"],
                        np.zeros((S, nt["pred_logit"].shape[1], 2)), 0.0)
    a0_agree_nt = float((np.argmax(s0_nt, axis=1) == sel0).mean())
    print(f"[fusion] navtest scenes: {S}; alpha=0 agreement {a0_agree_nt:.3f}")

    nt_risk = load_risk_aligned(Path(args.navtest_risk_dir), nt["tokens"],
                                args.variants)
    # navtest params: tuned on ALL of query_val
    full_best = tune_eval(qva_rows, qv_risk, grid_fusion_only=False)
    full_best_a = tune_eval(qva_rows, qv_risk, grid_fusion_only=True)

    results: Dict[str, dict] = {}
    nt_rows: List[List[str]] = []

    def report(name, sel):
        res = eval_sel(sel, nt["subscores"], scene_subset, args.num_boot,
                       args.split_seed)
        res["frac_changed"] = float((sel != sel0).mean())
        results[name] = res
        for metric in ("final", "NC", "TTC", "EP"):
            for subn in ("all", "conflict"):
                nt_rows.append([name, metric, subn,
                                fmt_ci(*res[f"{metric}|{subn}"]),
                                f"{res['frac_changed']:.3f}"])

    report("original_argmax", sel0)
    report("fused_alpha0_argmax", np.argmax(s0_nt, axis=1))
    report("oracle_best",
           np.argmax(nt["subscores"][..., SUB_FINAL], axis=1))

    tuned: Dict[str, dict] = {}
    for v, (pk, rk) in sorted(nt_risk.items()):
        for mode, bestd in (("Q5a", full_best_a), ("Q5b", full_best)):
            if v not in bestd:
                continue
            (a_, reg_, eps_), val_ = bestd[v]
            sel = sel0.copy()
            sel[pk] = fused_select(nt["pred_logit"][pk], rk, a_,
                                   nt["pdm_score"][pk], reg_, eps_, sel0[pk])
            tag = f"a={a_:g}"
            if mode == "Q5b":
                tag += f",region={reg_[0]}:{reg_[1]:g},eps={eps_}"
            report(f"{v} ({mode} {tag})", sel)
            tuned[f"{v}|{mode}"] = {"alpha": a_, "region": list(reg_),
                                   "eps": eps_, "qval_mean": val_}

    md = ["# Q5 fusion + trust-region re-ranking", ""]
    md.append(f"navtest scenes: {S} | query_val scenes: {len(qva_rows)} | "
              f"alpha=0 agreement qval {a0_agree:.3f} / navtest {a0_agree_nt:.3f} | "
              f"split_seed={args.split_seed}")
    md.append("")
    md.append("## tuned params (all of query_val)")
    md.append("| variant | mode | alpha | region | eps | qval mean final |")
    md.append("|---|---|---|---|---|---|")
    for k, d in sorted(tuned.items()):
        v, m = k.split("|")
        md.append(f"| {v} | {m} | {d['alpha']:g} | {d['region'][0]}:"
                  f"{d['region'][1]:g} | {d['eps']} | {d['qval_mean']:.4f} |")
    md.append("")
    md.append("## query_val re-ranking (5-fold log CV)")
    md.append("| selector | metric | subset | value [95% CI] | frac_changed |")
    md.append("|---|---|---|---|---|")
    for r in md_rows:
        md.append("| " + " | ".join(r) + " |")
    md.append("")
    md.append("## navtest selected-candidate outcomes (labelled subscores)")
    md.append("| selector | metric | subset | value [95% CI] | frac_changed |")
    md.append("|---|---|---|---|---|")
    for r in nt_rows:
        md.append("| " + " | ".join(r) + " |")
    (out_dir / "fusion_rerank_results.md").write_text("\n".join(md) + "\n")
    with open(out_dir / "fusion_rerank_results.json", "w") as f:
        json.dump({"a0_agree_qval": a0_agree, "a0_agree_navtest": a0_agree_nt,
                   "tuned": tuned,
                   "query_val_cv": {k: {m: (list(v) if isinstance(v, tuple)
                                            else v)
                                        for m, v in res.items()}
                                    for k, res in cv_report.items()},
                   "results": {k: {m: (list(v) if isinstance(v, tuple)
                                       else v)
                                   for m, v in res.items()}
                               for k, res in results.items()}},
                  f, indent=2)
    print(f"[fusion] done -> {out_dir}")


if __name__ == "__main__":
    main()
