#!/usr/bin/env python
"""R3: conservative flip re-ranking inside B0's top-K using kNN subscore
readouts dumped by eval_yhat.py --dump_npz.

Rule (per scene): default pick = B0's top1 by pdm. A challenger j inside
B0's pdm top-K replaces it only if ALL hold:
    a) fhat_j - fhat_top1 > m
    b) p_nc_j <= p_nc_top1 + eps AND p_dac_j <= p_dac_top1 + eps
    c) cos_j >= q_thr   (q_thr = global quantile of per-candidate cos)
Multiple valid challengers -> pick max fhat.

Metrics: mean true final of pick over {all, rare10, b0_wrong(K)} plus
oracle@K, B0 top1, and flip stats sw/win/lose/dwin/dlose. Shuffle control
uses the *_shuf readouts (same gate).

Grid: K {2,4,8} x m {0,.02,.05,.1} x eps {0,.05} x q-quantile {0,.25,.5}.
(K,m,eps,q) selected on VAL by 'all' mean final ONLY; navtest reported once
for the selected config (full grid appended, marked post-hoc).

Usage:
    python scripts/experience/rerank_yhat.py \
        --val_npz val_dump.npz --navtest_npz navtest_dump.npz \
        --variant b0prop --out_json rerank_r3.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

KS = (2, 4, 8)
MS = (0.0, 0.02, 0.05, 0.1)
EPS = (0.0, 0.05)
QS = (0.0, 0.25, 0.5)


def topk_idx(pdm_row, k):
    return np.argpartition(-pdm_row, k - 1)[:k]


def flip_eval(pdm, final, fhat, pnc, pdac, cos, splits, k, m, eps, qthr):
    S = len(pdm)
    pick = np.empty(S, np.int64)
    for s in range(S):
        t1 = int(pdm[s].argmax())
        tk = topk_idx(pdm[s], k)
        best, best_f = t1, -np.inf
        for j in tk:
            j = int(j)
            if j == t1:
                continue
            if (fhat[s, j] - fhat[s, t1] > m
                    and pnc[s, j] <= pnc[s, t1] + eps
                    and pdac[s, j] <= pdac[s, t1] + eps
                    and cos[s, j] >= qthr
                    and fhat[s, j] > best_f):
                best, best_f = j, fhat[s, j]
        pick[s] = best
    out = {}
    delta = final[np.arange(S), pick] - final[np.arange(S),
                                               pdm.argmax(1)]
    sw = pick != pdm.argmax(1)
    win = sw & (delta > 1e-6)
    lose = sw & (delta < -1e-6)
    for name, msk in splits.items():
        idx = np.where(msk)[0]
        out[name] = float(np.mean(final[idx, pick[idx]])) if len(idx) \
            else float("nan")
    out["sw"] = float(sw.mean())
    out["win"] = float(win.sum() / max(sw.sum(), 1))
    out["lose"] = float(lose.sum() / max(sw.sum(), 1))
    out["dwin"] = float(delta[win].mean()) if win.any() else 0.0
    out["dlose"] = float(delta[lose].mean()) if lose.any() else 0.0
    # bounds
    orac = [final[s, topk_idx(pdm[s], k)[
        final[s, topk_idx(pdm[s], k)].argmax()]] for s in range(S)]
    out["oracle@K"] = float(np.mean(orac))
    out["b0_top1"] = float(final[np.arange(S), pdm.argmax(1)].mean())
    return out


def grid(npz, variant):
    pdm, final = npz["pdm"], npz["final"]
    rare10 = npz["rare10"].astype(bool)
    pre = f"{variant}__"
    fhat = npz[pre + "fhat"]
    pnc = npz[pre + "p_nc"]
    pdac = npz[pre + "p_dac"]
    cos = npz[pre + "cos"]
    fs, pnc_s, pdac_s = (npz[pre + "fhat_shuf"], npz[pre + "p_nc_shuf"],
                         npz[pre + "p_dac_shuf"])
    res = {}
    for k in KS:
        b0w = np.array([
            final[s, pdm[s].argmax()] <
            final[s, topk_idx(pdm[s], k)].max() - 1e-6
            for s in range(len(pdm))])
        splits = {"all": np.ones(len(pdm), bool), "rare10": rare10,
                  "b0_wrong": b0w}
        for m in MS:
            for eps in EPS:
                for q in QS:
                    qt = float(np.quantile(cos, q)) if q > 0 else -np.inf
                    key = f"K{k}/m{m}/e{eps}/q{q}"
                    res[key] = flip_eval(pdm, final, fhat, pnc, pdac, cos,
                                         splits, k, m, eps, qt)
        res[f"K{k}/shuf"] = flip_eval(pdm, final, fs, pnc_s, pdac_s, cos,
                                      splits, k, 0.02, 0.0, -np.inf)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_npz", required=True)
    p.add_argument("--navtest_npz", default=None)
    p.add_argument("--variant", default="b0prop")
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    val = np.load(args.val_npz, allow_pickle=True)
    vres = grid(val, args.variant)
    report = {"variant": args.variant, "val": vres}

    # select on val 'all' mean final (exclude shuf rows)
    cand = {k: v for k, v in vres.items() if "/shuf" not in k}
    bkey = max(cand, key=lambda k: cand[k]["all"])
    report["selected"] = {"config": bkey, "val_all": cand[bkey]["all"],
                          "val_b0": cand[bkey]["b0_top1"]}

    if args.navtest_npz:
        nt = np.load(args.navtest_npz, allow_pickle=True)
        nres = grid(nt, args.variant)
        report["navtest_selected"] = {"config": bkey, **nres[bkey]}
        k_of_sel = bkey.split("/")[0]
        report["navtest_shuf_control"] = nres[f"{k_of_sel}/shuf"]
        report["navtest_grid_posthoc"] = nres

    txt = json.dumps(report, indent=1)
    if args.out_json:
        Path(args.out_json).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
