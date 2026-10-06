#!/usr/bin/env python
"""R2: does f_hat re-ranking inside B0's top-K raise true final (PDMS proxy)?

Input: npz dumped by eval_yhat.py --dump_npz (per run, per bank variant):
    tokens (S,), rare10 (S,), pdm (S,K), final (S,K),
    {variant}__fhat (S,K), {variant}__fhat_shuf (S,K)

Rule: take B0's top-K candidates by pdm; score = z(pdm) + beta*z(fhat) where
z is standardisation WITHIN the scene's top-K; pick argmax. Metric = mean true
final of the pick over scene splits {all, rare10, b0_wrong(K)} plus oracle@K
(best final in top-K) and the B0 pick (beta=0 must reproduce it).

(K, beta) are selected on the VAL dump only; navtest is reported once for the
selected config (full grid appended, marked post-hoc).

Usage:
    python scripts/experience/rerank_yhat.py \
        --val_npz val_dump.npz --navtest_npz navtest_dump.npz \
        --variant b0prop --out_json rerank_yhat.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

KS = (2, 4, 8)
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def topk_mask(pdm_row, k):
    tk = np.argpartition(-pdm_row, k - 1)[:k]
    return tk


def zscore(v):
    s = v.std()
    return (v - v.mean()) / (s + 1e-8)


def eval_config(pdm, final, fhat, splits, k, beta):
    """Mean true final of fused pick within top-K; splits = {name: mask(S)}."""
    S = len(pdm)
    out = {}
    for name, msk in splits.items():
        idx = np.where(msk)[0]
        vals = []
        for s in idx:
            tk = topk_mask(pdm[s], k)
            sc = zscore(pdm[s][tk]) + beta * zscore(fhat[s][tk])
            pick = tk[sc.argmax()]
            vals.append(final[s, pick])
        out[name] = float(np.mean(vals)) if vals else float("nan")
    # oracle and B0 within top-K (computed on 'all' for reference)
    orac = [final[s, topk_mask(pdm[s], k)[final[s, topk_mask(pdm[s], k)].argmax()]]
            for s in range(S)]
    b0 = [final[s, pdm[s].argmax()] for s in range(S)]
    out["oracle@K"] = float(np.mean(orac))
    out["b0_top1"] = float(np.mean(b0))
    return out


def grid_eval(npz, variant):
    pdm, final = npz["pdm"], npz["final"]
    rare10 = npz["rare10"].astype(bool)
    fhat = npz[f"{variant}__fhat"]
    fhat_sh = npz[f"{variant}__fhat_shuf"]
    res = {}
    for k in KS:
        b0_wrong = np.array([
            final[s, pdm[s].argmax()] < final[s, topk_mask(pdm[s], k)].max() - 1e-6
            for s in range(len(pdm))])
        splits = {"all": np.ones(len(pdm), bool), "rare10": rare10,
                  "b0_wrong": b0_wrong}
        for beta in BETAS:
            res[f"K{k}/b{beta}"] = eval_config(pdm, final, fhat, splits, k, beta)
        # shuffle control at beta=1 reference + best-beta slot filled later
        res[f"K{k}/shuf_b1"] = eval_config(pdm, final, fhat_sh, splits, k, 1.0)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_npz", required=True)
    p.add_argument("--navtest_npz", default=None)
    p.add_argument("--variant", default="b0prop")
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    val = np.load(args.val_npz, allow_pickle=True)
    vres = grid_eval(val, args.variant)
    report = {"variant": args.variant, "val": vres}

    # select (K, beta) on val: maximise b0_wrong gain over B0 top1
    best, best_gain = None, -1
    for k in KS:
        base = vres[f"K{k}/b0.0"]["b0_wrong"]
        for beta in BETAS[1:]:
            g = vres[f"K{k}/b{beta}"]["b0_wrong"] - base
            if g > best_gain:
                best_gain, best = g, (k, beta)
    k, beta = best
    report["selected"] = {"K": k, "beta": beta,
                          "val_b0_wrong_gain": best_gain}

    if args.navtest_npz:
        nt = np.load(args.navtest_npz, allow_pickle=True)
        nres = grid_eval(nt, args.variant)
        report["navtest_selected"] = nres[f"K{k}/b{beta}"]
        report["navtest_shuf_control"] = nres[f"K{k}/shuf_b1"]
        report["navtest_grid_posthoc"] = nres

    txt = json.dumps(report, indent=1)
    if args.out_json:
        Path(args.out_json).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
