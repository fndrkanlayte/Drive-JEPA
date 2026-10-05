#!/usr/bin/env python3
"""Q6: rarity stratification of navtest (no training).

Scene feature = mean-pooled proposal_feature (K,256) -> (256,), L2-normalised.
Rarity of a navtest scene = mean cosine distance to its k=10 nearest navtrain
scenes, excluding same-log scenes. Navtest is split into rarity deciles; for
each decile we report labelled final/NC/TTC/EP of the original argmax, the
best risk variant, and the best Q5b variant, with scene-bootstrap CIs and
selection-changed fractions. Also reports the conflict ∩ top-30%-rarity
subset.

Usage:
  python scripts/experience/rarity_strat.py \
      --labels_dir $EXP/experience/navtrain_labels_3k \
      --export_dir $EXP/experience/navtrain_export_3k \
      --navtest_labels_dir $EXP/experience/navtest_labels_full \
      --navtest_export_dir $EXP/experience/navtest_export_full \
      --navtest_risk_dir $EXP/experience/train_experience_full5_m/navtest_risk \
      --out_dir $EXP/experience/rarity_strat_out
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
# risk-penalty rows: (variant, lambda tuned on query_val 5-fold CV)
RISK_ROWS = [("noexp_int", 0.5), ("retrieval_int", 0.5),
             ("pred_desc_retrieval", 0.5), ("pred_desc_retrieval_pp", 0.5),
             ("retrieval", 1.0), ("random", 0.5)]
FUSION_ROWS = [("pred_desc_retrieval_pp", 0.2, ("topk", 2.0), 0.05),
               ("random", 0.2, ("margin", 0.005), 0.0)]
KNN = 10


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--export_dir", required=True)
    p.add_argument("--navtest_labels_dir", required=True)
    p.add_argument("--navtest_export_dir", required=True)
    p.add_argument("--navtest_risk_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rarity_npz", default=None,
                   help="reuse precomputed rarity.npz (tokens + rarity) "
                        "instead of recomputing scene features")
    p.add_argument("--num_boot", type=int, default=1000)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--dt_enter_thresh", type=float, default=2.0)
    p.add_argument("--device",
                   default="cuda" if __import__("torch").cuda.is_available()
                   else "cpu")
    return p.parse_args()


def scene_feats(export_dir: Path, label_dir: Path):
    """Per labelled scene: token, log, mean-pooled L2-normed feature."""
    feats, tokens, logs = [], [], []
    for lp in sorted(label_dir.glob("*.npz")):
        ep = export_dir / lp.name
        if not ep.is_file():
            continue
        lab, exp = load_npz(lp), load_npz(ep)
        f = np.asarray(exp["proposal_feature"], dtype=np.float32).mean(0)
        n = np.linalg.norm(f)
        feats.append(f / n if n > 0 else f)
        tokens.append(str(lab["token"].item()))
        logs.append(str(lab["log_name"].item()))
    return np.stack(feats), np.asarray(tokens), np.asarray(logs)


def rarity_scores(q_feats, q_logs, m_feats, m_logs, device):
    """Mean cosine distance to k=10 nearest memory scenes (other logs)."""
    import torch
    q = torch.from_numpy(q_feats).to(device)
    m = torch.from_numpy(m_feats).to(device)
    out = np.zeros(len(q), dtype=np.float64)
    for i in range(0, len(q), 512):
        s = q[i:i + 512] @ m.T  # cosine sim (normed feats)
        same = torch.from_numpy(
            q_logs[i:i + 512, None] == m_logs[None, :]).to(device)
        s = s.masked_fill(same, -2.0)
        top = torch.topk(s, KNN, dim=1).values
        out[i:i + 512] = (1.0 - top).mean(1).cpu().numpy()
    return out


def eval_subset(sel, subscores, keep, num_boot, seed, sel0):
    rows = np.arange(subscores.shape[0])[keep]
    if len(rows) == 0:
        return {}
    out = {}
    for metric, col in (("final", SUB_FINAL), ("NC", SUB_NC),
                        ("TTC", SUB_TTC), ("EP", SUB_EP)):
        out[metric] = bootstrap_scene_ci(subscores[rows, sel[rows], col],
                                         num_boot, seed)
    out["frac_changed"] = float((sel[rows] != sel0[rows]).mean())
    out["n"] = int(len(rows))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[rarity] loading navtest feats + labels", flush=True)
    nt = fr.scene_pack(Path(args.navtest_labels_dir),
                       Path(args.navtest_export_dir), args.dt_enter_thresh)
    if args.rarity_npz:
        rz = np.load(args.rarity_npz, allow_pickle=True)
        t2r = {t: i for i, t in enumerate(rz["tokens"].tolist())}
        rare = np.asarray(rz["rarity"])[
            [t2r[t] for t in nt["tokens"]]]
        print(f"[rarity] reused rarity scores from {args.rarity_npz}")
    else:
        print("[rarity] loading navtrain feats", flush=True)
        m_feats, _, m_logs = scene_feats(Path(args.export_dir),
                                         Path(args.labels_dir))
        print(f"[rarity] navtrain scenes: {len(m_feats)}")
        q_feats, q_tokens, q_logs = scene_feats(
            Path(args.navtest_export_dir), Path(args.navtest_labels_dir))
        t2i = {t: i for i, t in enumerate(q_tokens)}
        q_feats = q_feats[[t2i[t] for t in nt["tokens"]]]
        q_logs = q_logs[[t2i[t] for t in nt["tokens"]]]
        print("[rarity] scoring rarity", flush=True)
        rare = rarity_scores(q_feats, q_logs, m_feats, m_logs, args.device)
    dec = np.quantile(rare, np.linspace(0, 1, 11))

    S = len(nt["tokens"])
    sel0 = np.argmax(nt["pdm_score"], axis=1)
    scene_subset = nt["subset"][np.arange(S), sel0]

    # selectors -----------------------------------------------------------
    risks = {}
    for v, _ in RISK_ROWS + [(v, None) for v, _, _, _ in FUSION_ROWS]:
        if v in risks:
            continue
        risk = load_npz(Path(args.navtest_risk_dir) / f"{v}.npz")
        t2r = {t: i for i, t in enumerate(risk["tokens"].tolist())}
        idx = np.array([t2r[t] for t in nt["tokens"] if t in t2r])
        keep = np.array([i for i, t in enumerate(nt["tokens"]) if t in t2r])
        risks[v] = (risk["risk"][idx], keep)

    selectors = [("original_argmax", sel0)]
    for v, lam in RISK_ROWS:
        sel = sel0.copy()
        rr, keep = risks[v]
        corr = 1.0 - (1.0 - rr[..., 0].astype(np.float64)) \
            * (1.0 - rr[..., 1].astype(np.float64))
        sel[keep] = np.argmax(
            nt["pdm_score"][keep] - lam * corr, axis=1)
        selectors.append((f"{v}(lam={lam})", sel))
    for v, alpha, region, eps in FUSION_ROWS:
        sel = sel0.copy()
        rr, keep = risks[v]
        sel[keep] = fr.fused_select(
            nt["pred_logit"][keep], rr, alpha,
            nt["pdm_score"][keep], region, eps, sel0[keep])
        selectors.append((f"{v}(Q5b)", sel))

    rows_md = []
    results = {}
    for name, sel in selectors:
        for d in range(10):
            lo, hi = dec[d], dec[d + 1]
            keep_d = (rare >= lo) & (rare <= hi) if d == 9 \
                else (rare >= lo) & (rare < hi)
            res = eval_subset(sel, nt["subscores"], keep_d,
                              args.num_boot, args.split_seed, sel0)
            results[f"{name}|decile{d}"] = res
            for m in ("final", "NC", "TTC", "EP"):
                rows_md.append([name, f"d{d} (n={res['n']})", m,
                                fmt_ci(*res[m]),
                                f"{res['frac_changed']:.3f}"])

    # conflict ∩ top-30% rarity + top-10% (all scenes)
    top30 = rare >= np.quantile(rare, 0.7)
    top10 = rare >= np.quantile(rare, 0.9)
    extra_subsets = [("conflict ∩ top-30% rare", scene_subset & top30),
                     ("top-10% rare", top10),
                     ("conflict ∩ top-10% rare", scene_subset & top10)]
    rows_md2 = []
    for sname, keep_x in extra_subsets:
        for name, sel in selectors:
            res = eval_subset(sel, nt["subscores"], keep_x, args.num_boot,
                              args.split_seed, sel0)
            results[f"{name}|{sname}"] = res
            for m in ("final", "NC", "TTC", "EP"):
                rows_md2.append([sname, name, m, fmt_ci(*res[m]),
                                 f"{res['frac_changed']:.3f}"])

    # candidate-level NC/TTC AUPRC per decile (risk npz mean over seeds)
    def auprc(y_true, score):
        order = np.argsort(-score)
        tp = np.cumsum(y_true[order])
        fp = np.cumsum(1 - y_true[order])
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / max(tp[-1], 1)
        return float(np.sum(
            (rec[1:] - rec[:-1]) * (prec[1:] + prec[:-1]) / 2)
            + rec[0] * prec[0]) if tp[-1] > 0 else np.nan

    y_nc = (nt["subscores"][..., 0] < 1.0).ravel()
    y_ttc = (nt["subscores"][..., 3] < 1.0).ravel()
    rows_auprc = []
    for v, _ in RISK_ROWS:
        if v not in risks:
            continue
        rr, keep = risks[v]
        rh_nc = np.clip(rr[..., 0], 0, 1).ravel()
        rh_ttc = np.clip(rr[..., 1], 0, 1).ravel()
        has = np.zeros(S, dtype=bool)
        has[keep] = True
        for d in range(10):
            lo, hi = dec[d], dec[d + 1]
            dm = ((rare >= lo) & (rare <= hi) if d == 9
                  else (rare >= lo) & (rare < hi))
            sel_c = np.repeat(dm & has, rr.shape[1])
            rows_auprc.append([v, f"d{d}",
                               f"{auprc(y_nc[sel_c], rh_nc[sel_c]):.4f}",
                               f"{auprc(y_ttc[sel_c], rh_ttc[sel_c]):.4f}"])
        for sname, keep_x in extra_subsets:
            sel_c = np.repeat(keep_x & has, rr.shape[1])
            rows_auprc.append([v, sname,
                               f"{auprc(y_nc[sel_c], rh_nc[sel_c]):.4f}",
                               f"{auprc(y_ttc[sel_c], rh_ttc[sel_c]):.4f}"])

    md = ["# Q6 rarity stratification", ""]
    md.append(f"navtest scenes: {S} | rarity = mean cosine dist to "
              f"{KNN}NN navtrain scenes (excl. same log) | "
              f"rarity range [{rare.min():.4f}, {rare.max():.4f}], "
              f"decile edges: {np.round(dec, 4).tolist()}")
    md.append("")
    md.append("## per-decile labelled subscores")
    md.append("| selector | decile | metric | value [95% CI] | frac_changed |")
    md.append("|---|---|---|---|---|")
    for r in rows_md:
        md.append("| " + " | ".join(r) + " |")
    md.append("")
    md.append("## tail subsets (all scenes + conflict ∩ rare)")
    md.append("| subset | selector | metric | value [95% CI] | frac_changed |")
    md.append("|---|---|---|---|---|")
    for r in rows_md2:
        md.append("| " + " | ".join(r) + " |")
    md.append("")
    md.append("## candidate-level AUPRC per decile (nc_unsafe / ttc_bad)")
    md.append("| variant | decile | NC AUPRC | TTC AUPRC |")
    md.append("|---|---|---|---|")
    for r in rows_auprc:
        md.append("| " + " | ".join(r) + " |")
    (out_dir / "rarity_results.md").write_text("\n".join(md) + "\n")
    np.savez(out_dir / "rarity.npz", rarity=rare, tokens=nt["tokens"],
             deciles=dec, subset=scene_subset)
    print(f"[rarity] done -> {out_dir}")


if __name__ == "__main__":
    main()
