#!/usr/bin/env python
"""Attach label subscores + B0 scorer predictions to an eval_yhat dump.

Reads the dump npz (tokens + existing arrays), then for each scene token
loads ``pred_logit`` from the latents shard and ``subscores`` from the
labels shard. Writes a new npz with the original arrays plus:
    subs      (S,K,6) true subscores [NC,DAC,EP,TTC,Comfort,final]
    b0_sub    (S,K,6) sigmoid(pred_logit)
    b0_wrong_K{2,4,8} bool

Usage:
    python scripts/experience/augment_dump.py \
        --dump yhat_dump_x.npz --latents_dir LAT --labels_dir LAB \
        --out yhat_dump_x_aug.npz
"""

import argparse
from pathlib import Path

import numpy as np


def topk_idx(row, k):
    return np.argpartition(-row, k - 1)[:k]


def build_index(d):
    idx = {}
    for f in Path(d).rglob("*.npz"):
        idx[f.stem] = f
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--latents_dir", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    z = np.load(args.dump, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    tokens = [str(t) for t in z["tokens"]]
    lat_idx = build_index(args.latents_dir)
    lab_idx = build_index(args.labels_dir)

    subs = np.zeros((len(tokens), z["pdm"].shape[1], 6), np.float32)
    b0 = np.zeros_like(subs)
    miss_l = miss_b = 0
    for i, t in enumerate(tokens):
        lp, bp = lat_idx.get(t), lab_idx.get(t)
        if lp is None or bp is None:
            miss_l += lp is None
            miss_b += bp is None
            continue
        subs[i] = np.asarray(np.load(bp)["subscores"], np.float32)
        pl = np.asarray(np.load(lp)["pred_logit"], np.float32)
        b0[i] = 1.0 / (1.0 + np.exp(-pl))
    print(f"[augment] missing latents={miss_l} labels={miss_b} "
          f"of {len(tokens)}")
    out["subs"] = subs
    out["b0_sub"] = b0
    pdm, final = z["pdm"], z["final"]
    for ks in (2, 4, 8):
        out[f"b0_wrong_K{ks}"] = np.array([
            final[s, pdm[s].argmax()] <
            final[s, topk_idx(pdm[s], ks)].max() - 1e-6
            for s in range(len(pdm))])
    np.savez(args.out, **out)
    print(f"[augment] wrote {args.out}")


if __name__ == "__main__":
    main()
