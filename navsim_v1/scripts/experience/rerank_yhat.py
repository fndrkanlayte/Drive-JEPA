#!/usr/bin/env python
"""R3/R4: conservative flip re-ranking inside B0's top-K using kNN subscore
readouts dumped by eval_yhat.py --dump_npz (plus augment_dump.py fields).

Rule (per scene): default pick = B0's top1 by pdm. A challenger j inside
B0's pdm top-K replaces it only if ALL hold:
    a) fhat_j - fhat_top1 > m
    b) p_nc_j <= p_nc_top1 + eps AND p_dac_j <= p_dac_top1 + eps
    c) cos_j >= q_thr   (q_thr = global quantile of per-candidate cos)
    d) when b0g=1 and b0_sub present: B0's own predicted NC/DAC probs of j
       are >= top1's minus eps (both scorer and memory must agree j is not
       more dangerous)
Multiple valid challengers -> pick max fhat.

Metrics: mean true final of pick over {all, rare10, b0_wrong(K)} plus
oracle@K, B0 top1, and flip stats sw/win/lose/dwin/dlose. Shuffle control
uses the *_shuf readouts (same gate). --diagnose emits per-losing-flip
detail rows (true subs of the pick, neighbour estimates, cos).

Grid: K {2,4,8} x m {0,.02,.05,.1} x eps {0,.05} x q-quantile {0,.25,.5}
x b0g {0,1}. (K,m,eps,q,b0g) selected on VAL by 'all' mean final ONLY;
navtest reported once for the selected config (full grid appended, marked
post-hoc).

Usage:
    python scripts/experience/rerank_yhat.py \
        --val_npz val_aug.npz --navtest_npz navtest_aug.npz \
        --variant b0prop --out_json rerank_r3.json --diagnose
"""

import argparse
import json
from pathlib import Path

import numpy as np

KS = (2, 4, 8)
MS = (0.0, 0.02, 0.05, 0.1)
EPS = (0.0,)
QS = (0.0, 0.25, 0.5)


def topk_idx(pdm_row, k):
    return np.argpartition(-pdm_row, k - 1)[:k]


def pick_one(pdm_s, fhat_s, pnc_s, pdac_s, pttc_s, cos_s, k, m, eps, qthr):
    """Return picked candidate index for one scene."""
    t1 = int(pdm_s.argmax())
    best, best_f = t1, -np.inf
    for j in topk_idx(pdm_s, k):
        j = int(j)
        if j == t1:
            continue
        if not (fhat_s[j] - fhat_s[t1] > m):
            continue
        if not (pnc_s[j] <= pnc_s[t1] + eps
                and pdac_s[j] <= pdac_s[t1] + eps
                and pttc_s[j] <= pttc_s[t1] + eps):
            continue
        if cos_s[j] < qthr:
            continue
        if fhat_s[j] > best_f:
            best, best_f = j, fhat_s[j]
    return best


def flip_eval(pdm, final, fhat, pnc, pdac, pttc, cos, splits, k, m, eps,
              qthr, diagnose=None, cfg_key=None):
    S = len(pdm)
    pick = np.empty(S, np.int64)
    for s in range(S):
        pick[s] = pick_one(pdm[s], fhat[s], pnc[s], pdac[s], pttc[s], cos[s],
                           k, m, eps, qthr)
    out = {}
    t1 = pdm.argmax(1)
    delta = final[np.arange(S), pick] - final[np.arange(S), t1]
    sw = pick != t1
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
    orac = [final[s, topk_idx(pdm[s], k)[
        final[s, topk_idx(pdm[s], k)].argmax()]] for s in range(S)]
    out["oracle@K"] = float(np.mean(orac))
    out["b0_top1"] = float(final[np.arange(S), t1].mean())
    if diagnose is not None:
        rows = []
        for s in np.where(lose)[0]:
            j = int(pick[s])
            rows.append({
                "scene": int(s), "pick": j, "top1": int(t1[s]),
                "true_subs": None, "delta": float(delta[s]),
                "fhat": float(fhat[s, j]), "p_nc": float(pnc[s, j]),
                "p_dac": float(pdac[s, j]), "cos": float(cos[s, j])})
        diagnose.append((cfg_key, rows))
    return out


def grid(npz, variant, diagnose_rows=None, selected_only=None):
    pdm, final = npz["pdm"], npz["final"]
    rare10 = npz["rare10"].astype(bool)
    pre = f"{variant}__"
    fhat = npz[pre + "fhat"]
    pnc = npz[pre + "p_nc"]
    pdac = npz[pre + "p_dac"]
    pttc = npz[pre + "p_ttc"]
    cos = npz[pre + "cos"]
    fs, pnc_s, pdac_s, pttc_s = (npz[pre + "fhat_shuf"], npz[pre + "p_nc_shuf"],
                               npz[pre + "p_dac_shuf"], npz[pre + "p_ttc_shuf"])
    res = {"_cos_stats": {"median": float(np.median(cos)),
                          "p10": float(np.quantile(cos, .10)),
                          "p90": float(np.quantile(cos, .90))}}
    for k in KS:
        if f"b0_wrong_K{k}" in npz.files:
            b0w = npz[f"b0_wrong_K{k}"].astype(bool)
        else:
            b0w = np.array([
                final[s, pdm[s].argmax()] <
                final[s, topk_idx(pdm[s], k)].max() - 1e-6
                for s in range(len(pdm))])
        splits = {"all": np.ones(len(pdm), bool), "rare10": rare10,
                  "b0_wrong": b0w}
        for m in MS:
            for eps in EPS:
                for q in QS:
                    qt = float(np.quantile(cos, q)) if q > 0 \
                        else -np.inf
                    key = f"K{k}/m{m}/e{eps}/q{q}"
                    res[key] = flip_eval(
                        pdm, final, fhat, pnc, pdac, pttc, cos, splits,
                        k, m, eps, qt,
                        diagnose=diagnose_rows, cfg_key=key)
        res[f"K{k}/shuf"] = flip_eval(pdm, final, fs, pnc_s, pdac_s, pttc_s,
                                      cos, splits, k, 0.02, 0.0, -np.inf)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_npz", required=True)
    p.add_argument("--navtest_npz", default=None)
    p.add_argument("--variant", default="b0prop")
    p.add_argument("--out_json", default=None)
    p.add_argument("--diagnose", action="store_true",
                   help="emit losing-flip detail rows for the selected navtest config")
    args = p.parse_args()

    val = np.load(args.val_npz, allow_pickle=True)
    vres = grid(val, args.variant)
    report = {"variant": args.variant, "val": vres}

    cand = {k: v for k, v in vres.items()
            if "/shuf" not in k and not k.startswith("_")}
    bkey = max(cand, key=lambda k: cand[k]["all"])
    report["selected"] = {"config": bkey, "val_all": cand[bkey]["all"],
                          "val_b0": cand[bkey]["b0_top1"]}

    if args.navtest_npz:
        nt = np.load(args.navtest_npz, allow_pickle=True)
        diag = [] if args.diagnose else None
        nres = grid(nt, args.variant, diagnose_rows=diag)
        report["navtest_selected"] = {"config": bkey, **nres[bkey]}
        k_of_sel = bkey.split("/")[0]
        report["navtest_shuf_control"] = nres[f"{k_of_sel}/shuf"]
        report["navtest_grid_posthoc"] = nres
        if args.diagnose and "subs" in nt.files:
            subs = nt["subs"]
            for (kk, rows) in diag:
                if kk != bkey:
                    continue
                for r in rows:
                    r["true_subs"] = subs[r["scene"], r["pick"]].tolist()
                    r["nc0"] = bool(subs[r["scene"], r["pick"], 0] < 1)
                    r["dac0"] = bool(subs[r["scene"], r["pick"], 1] < 1)
                report["navtest_losing_flips"] = rows
                nc0 = sum(r["nc0"] for r in rows)
                dac0 = sum(r["dac0"] for r in rows)
                both = sum(r["nc0"] and r["dac0"] for r in rows)
                report["navtest_losing_summary"] = {
                    "loses": len(rows), "nc0": nc0, "dac0": dac0,
                    "both": both,
                    "cos_mean": float(np.mean([r["cos"] for r in rows]))
                    if rows else None,
                    "pnc_mean": float(np.mean([r["p_nc"] for r in rows]))
                    if rows else None,
                    "pdac_mean": float(np.mean([r["p_dac"] for r in rows]))
                    if rows else None}

    txt = json.dumps(report, indent=1)
    if args.out_json:
        Path(args.out_json).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
