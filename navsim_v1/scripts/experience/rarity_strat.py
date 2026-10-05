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
RISK_VARIANT = "noexp_int"          # best ungated risk variant
RISK_LAM = 0.5
FUSION_VARIANT = "pred_desc_retrieval_pp"  # best Q5b on navtest
FUSION_ALPHA, FUSION_REGION, FUSION_EPS = 0.2, ("topk", 2.0), 0.05
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

    print("[rarity] loading navtrain feats", flush=True)
    m_feats, _, m_logs = scene_feats(Path(args.export_dir),
                                     Path(args.labels_dir))
    print(f"[rarity] navtrain scenes: {len(m_feats)}")

    print("[rarity] loading navtest feats + labels", flush=True)
    nt = fr.scene_pack(Path(args.navtest_labels_dir),
                       Path(args.navtest_export_dir), args.dt_enter_thresh)
    q_feats, q_tokens, q_logs = scene_feats(
        Path(args.navtest_export_dir), Path(args.navtest_labels_dir))
    # align q_feats to nt token order
    t2i = {t: i for i, t in enumerate(q_tokens)}
    q_feats = q_feats[[t2i[t] for t in nt["tokens"]]]
    q_logs = q_logs[[t2i[t] for t in nt["tokens"]]]

    print("[rarity] scoring rarity", flush=True)
    rare = rarity_scores(q_feats, q_logs, m_feats, m_logs, args.device)
    dec = np.quantile(rare, np.linspace(0, 1, 11))

    S = len(nt["tokens"])
    sel0 = np.argmax(nt["pdm_score"], axis=1)
    scene_subset = nt["subset"][np.arange(S), sel0]

    # selectors
    sel_risk = sel0.copy()
    sel_fus = sel0.copy()
    risk = load_npz(Path(args.navtest_risk_dir) / f"{RISK_VARIANT}.npz")
    t2r = {t: i for i, t in enumerate(risk["tokens"].tolist())}
    idx = np.array([t2r[t] for t in nt["tokens"] if t in t2r])
    keep = np.array([i for i, t in enumerate(nt["tokens"]) if t in t2r])
    corr = 1.0 - (1.0 - risk["risk"][idx][..., 0]) \
        * (1.0 - risk["risk"][idx][..., 1])
    sel_risk[keep] = np.argmax(
        nt["pdm_score"][keep] - RISK_LAM * corr.astype(np.float64), axis=1)
    sel_fus[keep] = fr.fused_select(
        nt["pred_logit"][keep], risk["risk"][idx], FUSION_ALPHA,
        nt["pdm_score"][keep], FUSION_REGION, FUSION_EPS, sel0[keep])
    selectors = [("original_argmax", sel0),
                 (f"{RISK_VARIANT}(lam={RISK_LAM})", sel_risk),
                 (f"{FUSION_VARIANT}(Q5b)", sel_fus)]

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

    # conflict ∩ top-30% rarity
    top30 = rare >= np.quantile(rare, 0.7)
    keep_c = scene_subset & top30
    rows_md2 = []
    for name, sel in selectors:
        res = eval_subset(sel, nt["subscores"], keep_c, args.num_boot,
                          args.split_seed, sel0)
        results[f"{name}|conflict_top30"] = res
        for m in ("final", "NC", "TTC", "EP"):
            rows_md2.append([name, m, fmt_ci(*res[m]),
                             f"{res['frac_changed']:.3f}"])

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
    md.append(f"## conflict ∩ top-30% rarity (n={int(keep_c.sum())})")
    md.append("| selector | metric | value [95% CI] | frac_changed |")
    md.append("|---|---|---|---|")
    for r in rows_md2:
        md.append("| " + " | ".join(r) + " |")
    (out_dir / "rarity_results.md").write_text("\n".join(md) + "\n")
    np.savez(out_dir / "rarity.npz", rarity=rare, tokens=nt["tokens"],
             deciles=dec, subset=scene_subset)
    print(f"[rarity] done -> {out_dir}")


if __name__ == "__main__":
    main()
