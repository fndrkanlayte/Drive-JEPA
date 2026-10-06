#!/usr/bin/env python
"""Select trajectories with the EWM readout head scores (no kNN).

Input: an eval_readout.py dump with tokens/pdm/final/subs/readout/rare10.

Scoring variants over readout probs r (K,6), order [NC,DAC,EP,TTC,C,final]:
  - "composed": s_hat = NC * DAC * (5*EP + 5*TTC + 2*C) / 12  (PDM formula)
  - "finalcol": s_hat = r[..., 5]                            (learned final)

Selection rules inside B0's pdm top-K (K in {2,4,8,32}):
  - rule "a": argmax s_hat                       (ignore pdm entirely)
  - rule "b": argmax z(pdm) + beta * z(s_hat),   beta in {0.5,1,2,4}
              z = standardise within the scene's top-K

Metrics on the picked candidates' TRUE final (labels subscores[:,5]):
all / b0_wrong(K) / rare10 / sw / win / lose / dwin / dlose, plus a
danger-recall AUC on the pdm top-8: candidates with true NC<1|DAC<1|TTC<1
vs the readout-derived p_unsafe = 1 - NC*DAC*TTC.

Config is selected on val by `all`; navtest reports only that config.
"""
import sys
from pathlib import Path

import numpy as np

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

KS = (2, 4, 8, 32)
BETAS = (0.5, 1.0, 2.0, 4.0)
VARIANTS = ("composed", "finalcol")


def topk_idx(pdm_s, k):
    return np.argsort(-pdm_s)[:k]


def s_hat_scores(readout, variant):
    """(S,K) score per candidate from readout probs (S,K,6)."""
    r = readout
    if variant == "composed":
        return r[..., 0] * r[..., 1] * (5 * r[..., 2] + 5 * r[..., 3]
                                        + 2 * r[..., 4]) / 12.0
    if variant == "finalcol":
        return r[..., 5].copy()
    raise ValueError(variant)


def pick_scene(pdm_s, shat_s, k, rule, beta=0.0):
    tk = topk_idx(pdm_s, k)
    zp = pdm_s[tk] - pdm_s[tk].mean()
    zs = shat_s[tk] - shat_s[tk].mean()
    zp = zp / (pdm_s[tk].std() + 1e-9)
    zs = zs / (shat_s[tk].std() + 1e-9)
    if rule == "a":
        score = shat_s[tk]
    else:
        score = zp + beta * zs
    return int(tk[np.argmax(score)])


def evaluate(z, k, rule, beta, variant, b0w_k):
    pdm, final, rare = z["pdm"], z["final"], z["rare10"].astype(bool)
    shat = s_hat_scores(z["readout"], variant)
    S = len(pdm)
    pick = np.array([pick_scene(pdm[s], shat[s], k, rule, beta)
                     for s in range(S)])
    t1 = pdm.argmax(1)
    rows = final[np.arange(S), pick]
    dl = rows - final[np.arange(S), t1]
    sw = pick != t1
    win = sw & (dl > 1e-6)
    lose = sw & (dl < -1e-6)
    b0w = b0w_k[k]
    return {
        "all": float(rows.mean()),
        "b0_wrong": float(rows[b0w].mean()),
        "rare10": float(rows[rare].mean()),
        "sw": float(sw.mean()),
        "win": float(win.sum() / max(sw.sum(), 1)),
        "lose": float(lose.sum() / max(sw.sum(), 1)),
        "dwin": float(dl[win].mean()) if win.any() else 0.0,
        "dlose": float(dl[lose].mean()) if lose.any() else 0.0,
        "b0_top1": float(final[np.arange(S), t1].mean()),
    }


def b0_wrong_masks(z):
    pdm, final = z["pdm"], z["final"]
    out = {}
    for k in KS:
        out[k] = np.array([
            final[s, pdm[s].argmax()] <
            final[s, topk_idx(pdm[s], k)].max() - 1e-6
            for s in range(len(pdm))])
    return out


def danger_auc(z, k=8):
    """AUC separating truly-dangerous top-K candidates by readout-derived
    p_unsafe = 1 - NC*DAC*TTC; same protocol as the memory readout."""
    pdm, subs, r = z["pdm"], z["subs"], z["readout"]
    pun = 1.0 - r[..., 0] * r[..., 1] * r[..., 3]
    num = den = 0
    pd_, ps_ = [], []
    for s in range(len(pdm)):
        tk = topk_idx(pdm[s], k)
        sb = subs[s]
        dg = (sb[tk, 0] < 1) | (sb[tk, 1] < 1) | (sb[tk, 3] < 1)
        for j, d in zip(tk, dg):
            (pd_ if d else ps_).append(pun[s, j])
        for j in tk[dg]:
            for j2 in tk[~dg]:
                den += 1
                num += (pun[s, j] > pun[s, j2]) + 0.5 * (pun[s, j] == pun[s, j2])
    return {"auc": float(num / den), "p_unsafe_danger": float(np.mean(pd_)),
            "p_unsafe_safe": float(np.mean(ps_)), "n_danger": len(pd_)}


def grid(z):
    b0w = b0_wrong_masks(z)
    res = {}
    for variant in VARIANTS:
        for k in KS:
            res[(variant, k, "a", 0.0)] = evaluate(z, k, "a", 0.0, variant, b0w)
            for beta in BETAS:
                res[(variant, k, "b", beta)] = evaluate(
                    z, k, "b", beta, variant, b0w)
    return res


def select_config(val_grid):
    best = max(val_grid.items(), key=lambda kv: kv[1]["all"])
    return best[0]


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--val_dump", required=True)
    p.add_argument("--nt_dump", required=True)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    zv = np.load(args.val_dump, allow_pickle=True)
    zn = np.load(args.nt_dump, allow_pickle=True)
    vg = grid(zv)
    cfg = select_config(vg)
    out = {
        "sel_cfg": {"variant": cfg[0], "K": cfg[1], "rule": cfg[2],
                    "beta": cfg[3]},
        "val_sel": vg[cfg],
        "navtest_sel": evaluate(zn, cfg[1], cfg[2], cfg[3], cfg[0],
                               b0_wrong_masks(zn)),
        "navtest_danger": danger_auc(zn),
        "val_danger": danger_auc(zv),
        "navtest_grid": {f"{c[0]}|K{c[1]}|{c[2]}|{c[3]}": v
                         for c, v in grid(zn).items()},
    }
    txt = json.dumps(out, indent=1)
    if args.out_json:
        Path(args.out_json).write_text(txt)
    print(txt)
