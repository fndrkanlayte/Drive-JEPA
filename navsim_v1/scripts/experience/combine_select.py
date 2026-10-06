#!/usr/bin/env python
"""Combine: readout-head danger filter + kNN fhat pick ("路数1: 分工融合").

Two signal sources, joined by token:
  * danger: rd_dump readout probs -> p_unsafe = 1 - min(sig NC, sig DAC,
    sig TTC) (label order [NC,DAC,EP,TTC,C,final]); or B0's own pred_logit
    (b0_sub in the key dump, already sigmoid) as the control source.
  * ranking: yhat_key dump's b0prop__fhat (memory kNN outcome readout).

Rule inside B0's pdm top-K:
  1. drop candidates with p_unsafe > tau; B0 top1 is always kept;
  2. pick argmax fhat among survivors; switch only if
     fhat_best - fhat_B0top1 > m, else stay with B0.

Grid K x tau x m is selected on val by `all`; navtest reports the selected
config per danger source. tau <= 0 keeps only B0's top1 (== B0 exactly).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from readout_select import topk_idx  # noqa: E402

KS = (2, 4, 8)
TAUS = (0.05, 0.1, 0.2, 0.3)
MS = (0.0, 0.01, 0.02, 0.05)


def join_by_token(z_key, z_rd):
    """Reorder z_rd rows into z_key order; assert pdm identical."""
    idx_of = {t: i for i, t in enumerate(z_rd["tokens"])}
    try:
        idx = np.array([idx_of[t] for t in z_key["tokens"]])
    except KeyError as e:
        raise SystemExit(f"token join failed: {e} missing")
    if not np.allclose(z_key["pdm"], z_rd["pdm"][idx]):
        raise SystemExit("pdm mismatch after token join")
    return idx


def p_unsafe_from(probs):
    """(S,K,6) probs in label order -> 1 - min(NC, DAC, TTC)."""
    return 1.0 - np.minimum.reduce([probs[..., 0], probs[..., 1], probs[..., 3]])


def pick_one(pdm_s, fhat_s, pun_s, k, tau, m):
    tk = topk_idx(pdm_s, k)
    t1 = tk[0]
    if tau > 0:
        keep = np.array([j for j in tk if pun_s[j] <= tau or j == t1])
    else:
        keep = np.array([t1])
    j = keep[np.argmax(fhat_s[keep])]
    if fhat_s[j] - fhat_s[t1] > m:
        return int(j)
    return int(t1)


def evaluate(z, pun, fhat, k, tau, m, b0w_k):
    pdm, final, rare = z["pdm"], z["final"], z["rare10"].astype(bool)
    S = len(pdm)
    pick = np.array([pick_one(pdm[s], fhat[s], pun[s], k, tau, m)
                     for s in range(S)])
    t1 = pdm.argmax(1)
    rows = final[np.arange(S), pick]
    dl = rows - final[np.arange(S), t1]
    sw = pick != t1
    win = sw & (dl > 1e-6)
    lose = sw & (dl < -1e-6)
    b0w = b0w_k[k]
    return {"all": float(rows.mean()), "b0_wrong": float(rows[b0w].mean()),
            "rare10": float(rows[rare].mean()), "sw": float(sw.mean()),
            "win": float(win.sum() / max(sw.sum(), 1)),
            "lose": float(lose.sum() / max(sw.sum(), 1)),
            "dlose": float(dl[lose].mean()) if lose.any() else 0.0,
            "dwin": float(dl[win].mean()) if win.any() else 0.0,
            "b0_top1": float(final[np.arange(S), t1].mean())}


def b0_wrong_masks(z):
    pdm, final = z["pdm"], z["final"]
    out = {}
    for k in KS:
        out[k] = np.array([
            final[s, pdm[s].argmax()] <
            final[s, topk_idx(pdm[s], k)].max() - 1e-6
            for s in range(len(pdm))])
    return out


def grid(z, pun, fhat, b0w):
    res = {}
    for k in KS:
        for tau in TAUS:
            for m in MS:
                res[(k, tau, m)] = evaluate(z, pun, fhat, k, tau, m, b0w)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--key_val", required=True)
    p.add_argument("--key_nt", required=True)
    p.add_argument("--rd_val", action="append", default=[],
                   help="name:path rd dumps (repeatable)")
    p.add_argument("--rd_nt", action="append", default=[])
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    zkv = np.load(args.key_val, allow_pickle=True)
    zkn = np.load(args.key_nt, allow_pickle=True)
    fhat_v, fhat_n = zkv["b0prop__fhat"], zkn["b0prop__fhat"]
    sources = {"b0": (p_unsafe_from(zkv["b0_sub"]),
                      p_unsafe_from(zkn["b0_sub"]))}
    for spec_v, spec_n in zip(args.rd_val, args.rd_nt):
        name_v, path_v = spec_v.split(":", 1)
        name_n, path_n = spec_n.split(":", 1)
        zrv = np.load(path_v, allow_pickle=True)
        zrn = np.load(path_n, allow_pickle=True)
        iv, in_ = join_by_token(zkv, zrv), join_by_token(zkn, zrn)
        sources[name_v] = (p_unsafe_from(zrv["readout"][iv]),
                           p_unsafe_from(zrn["readout"][in_]))

    out = {"b0_navtest": float(
        zkn["final"][np.arange(len(zkn["final"])),
                     zkn["pdm"].argmax(1)].mean())}
    b0w_v, b0w_n = b0_wrong_masks(zkv), b0_wrong_masks(zkn)
    for name, (pv, pn) in sources.items():
        vg = grid(zkv, pv, fhat_v, b0w_v)
        cfg = max(vg.items(), key=lambda kv: kv[1]["all"])[0]
        out[name] = {
            "sel_cfg": {"K": cfg[0], "tau": cfg[1], "m": cfg[2]},
            "val_sel": vg[cfg],
            "navtest_sel": evaluate(zkn, pn, fhat_n, cfg[0], cfg[1], cfg[2],
                                    b0w_n),
            "navtest_grid": {f"K{c[0]}|t{c[1]}|m{c[2]}": v["all"]
                             for c, v in grid(zkn, pn, fhat_n, b0w_n).items()},
        }
    txt = json.dumps(out, indent=1)
    if args.out_json:
        Path(args.out_json).write_text(txt)
    print(txt)
