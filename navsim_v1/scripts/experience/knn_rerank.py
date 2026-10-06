#!/usr/bin/env python
"""kNN-outcome reranking: almost parameter-free Delta from retrieved outcomes.

For each of the top-K candidates by B0 logit, retrieve the n nearest
BANK candidates (candidate-level, same-log excluded), take a
distance-weighted mean of their true `final` scores -> fhat_k, and rerank:

    score_k = logit(B0_k) + beta * (fhat_k - mean_topK fhat)

Candidates outside the top-K get score=-1e9 (never selected).

Retrieval spaces:
    yhat_ego    y_hat[:, slot0, :]        (64d, ego predicted latent)
    yhat_mean   y_hat.mean over slots     (64d)
    traj_z      [traj.flatten(), z_pool]  (24+256d; z-scored by bank stats)

Controls: shuffle (fhat deranged across different-log scenes),
random (neighbour finals drawn uniformly from other-log candidates),
scale{S} (random S% of the bank).

  val grid:  python knn_rerank.py --query_cache $E/b3cache_navtrain \
      --bank_cache $E/b3cache_navtrain \
      --tokens $E/split_navtrain/val_tokens.txt \
      --bank_tokens $E/split_navtrain/train_tokens.txt \
      --grid --out $E/knn_val.json

  navtest:   python knn_rerank.py --query_cache $E/b3cache_navtest \
      --bank_cache $E/b3cache_navtrain --tokens /tmp/navtest_all.txt \
      --K 8 --n 32 --t .2 --beta 4 --space yhat_ego \
      --labels_dir $E/navtest_labels_full --latents_dir $E/latents_navtest \
      --ref_labels_dir $E/navtrain_labels_full \
      --ref_latents_dir $E/latents_navtrain \
      --ref_tokens $E/split_navtrain/train_tokens.txt --out $E/knn_nt.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader  # noqa: E402
from train_memory import CacheDataset, collate_cache  # noqa: E402
from eval_ewm import (  # noqa: E402
    LatentDataset, build_index, knn_mean_dist, scene_desc, KNN_Q)


# --------------------------------------------------------------------------
# candidate-level features
# --------------------------------------------------------------------------

def cand_feats(yhat, traj, z_pool, space):
    """yhat (S,K,5,64), traj (S,K,8,3), z_pool (S,256) -> (S,K,d)."""
    if space == "yhat_ego":
        return np.asarray(yhat[:, :, 0, :], np.float32)
    if space == "yhat_mean":
        return np.asarray(yhat.mean(2), np.float32)
    if space == "traj_z":
        K = traj.shape[1]
        return np.concatenate(
            [traj.reshape(len(traj), K, -1).astype(np.float32),
             np.repeat(z_pool[:, None, :], K, 1).astype(np.float32)], -1)
    raise ValueError(space)


def knn_topn(q_feat, bank_feat, bank_log, q_log,
             n, bank_mask=None, chunk=256, device="cpu"):
    """Indices + squared distances of the n nearest non-same-log bank
    candidates for each query candidate. Returns (ni, dn) (S,Kq,n)."""
    if bank_mask is not None:
        keep = np.flatnonzero(bank_mask)
        bank_feat = bank_feat[keep]
        bank_log = bank_log[keep]
    Kq = q_feat.shape[1]
    q = torch.from_numpy(np.asarray(q_feat, np.float32)).to(device)
    bf = torch.from_numpy(np.asarray(bank_feat, np.float32)).to(device)
    ni_all = np.zeros((len(q), Kq, n), np.int64)
    dn_all = np.zeros((len(q), Kq, n), np.float32)
    for s in range(0, len(q), chunk):
        sl = slice(s, min(s + chunk, len(q)))
        qc = q[sl].reshape(-1, q.shape[-1])                    # (B*Kq,d)
        d2 = torch.cdist(qc, bf).pow(2)                        # (B*Kq,Nb)
        row_log = np.repeat(q_log[sl], Kq)
        same = torch.from_numpy(row_log[:, None] == bank_log[None, :])
        d2[same.to(device)] = torch.inf
        dn, ni = d2.topk(n, dim=1, largest=False)              # (B*Kq,n)
        ni_all[sl] = ni.reshape(sl.stop - sl.start, Kq, n).cpu().numpy()
        dn_all[sl] = dn.reshape(sl.stop - sl.start, Kq, n).cpu().numpy()
    return ni_all, dn_all


def fhat_from_topn(ni, dn, bank_final, n, t):
    """fhat from precomputed top-n lists (prefix subsets allowed)."""
    ni, dn = ni[:, :, :n], dn[:, :, :n]
    w = torch.softmax(-torch.from_numpy(dn) / t, dim=-1).numpy()
    return (w * bank_final[ni]).sum(-1).astype(np.float32)


def knn_fhat(q_feat, bank_feat, bank_final, bank_log, q_log,
             n, t, bank_mask=None, chunk=256, device="cpu"):
    """Distance-weighted mean of the n nearest bank candidates' finals."""
    ni, dn = knn_topn(q_feat, bank_feat, bank_log, q_log, n=n,
                      bank_mask=bank_mask, chunk=chunk, device=device)
    if bank_mask is not None:
        bank_final = bank_final[np.flatnonzero(bank_mask)]
    return fhat_from_topn(ni, dn, bank_final, n, t)


def rerank_scores(b0, fhat_topk, tk, beta):
    """score = logit(B0) + beta*(fhat - mean_topK fhat), -1e9 outside topK.

    b0 (S,Kall); fhat_topk (S,K); tk (S,K) top-K candidate indices."""
    lb = np.log(np.clip(b0, 1e-6, 1 - 1e-6) /
                np.clip(1 - b0, 1e-6, 1 - 1e-6))
    sc = np.full(lb.shape, -1e9, np.float64)
    lb_t = np.take_along_axis(lb, tk, 1)
    s_t = lb_t + beta * (fhat_topk - fhat_topk.mean(1, keepdims=True))
    np.put_along_axis(sc, tk, s_t, 1)
    return sc


def metrics(final, b0, picks, wrong_margin=0.05):
    fpick = final[np.arange(len(final)), picks]
    b0_pick = b0.argmax(1)
    fb0 = final[np.arange(len(final)), b0_pick]
    fbest = final.max(1)
    wrong = fb0 < fbest - wrong_margin
    d = fpick - fb0
    nf = float(d[wrong].mean() * wrong.mean()) if wrong.any() else 0.0
    nb = float(d[~wrong].mean() * (~wrong).mean()) if (~wrong).any() else 0.0
    return dict(val=float(fpick.mean()), flip=float((picks != b0_pick).mean()),
                net_fix=nf, net_break=nb)


def derange_fhat(fhat, logs, seed=0):
    """Swap fhat rows between different-log scenes (memory-control)."""
    rng = np.random.default_rng(seed)
    out = fhat.copy()
    perm = rng.permutation(len(fhat))
    ok = logs[perm] != logs
    out[ok] = fhat[perm[ok]]
    return out


def topk_idx(b0, K):
    lb = np.log(np.clip(b0, 1e-6, 1 - 1e-6) /
                np.clip(1 - b0, 1e-6, 1 - 1e-6))
    K = min(K, b0.shape[1])
    return np.argpartition(-lb, K - 1, axis=1)[:, :K]


# --------------------------------------------------------------------------

def load_cache(cache_dir, tokens):
    ds = CacheDataset(tokens, Path(cache_dir))
    zp, yh, tr, sub, b0, oc, logs = [], [], [], [], [], [], []
    for b in DataLoader(ds, batch_size=64, shuffle=False, num_workers=8,
                        collate_fn=collate_cache):
        zp.append(b["z_pool"].numpy()); yh.append(b["yhat"].numpy())
        tr.append(b["traj"].numpy()); sub.append(b["sub"].numpy())
        b0.append(b["b0"].numpy()); logs.extend(b["log_names"])
        oc.append(b["outcomes"].numpy())
    return dict(z_pool=np.concatenate(zp), yhat=np.concatenate(yh),
                traj=np.concatenate(tr), sub=np.concatenate(sub),
                b0=np.concatenate(b0), outcomes=np.concatenate(oc),
                logs=np.asarray(logs),
                tokens=np.asarray([p.stem for p in ds.items]))


def strat_masks(q, tokens, args):
    """all / conflict / rare10 / tert* / b0_wrong masks."""
    m = {"all": np.ones(len(q["logs"]), bool)}
    m["conflict"] = (q["outcomes"][:, :, 0] > 0).any(1)
    b0_pick = q["b0"].argmax(1)
    m["b0_wrong"] = q["sub"][..., 5][np.arange(len(q["sub"])),
                                    b0_pick] < q["sub"][..., 5].max(1) - 0.05
    if args.labels_dir:
        q_lab = LatentDataset(tokens, build_index(Path(args.latents_dir)),
                              Path(args.labels_dir))
        ref_lab = LatentDataset(
            [l.strip() for l in open(args.ref_tokens) if l.strip()],
            build_index(Path(args.ref_latents_dir)),
            Path(args.ref_labels_dir))
        ref_desc = np.stack([scene_desc(ref_lab[i]["outcomes"])
                             for i in range(len(ref_lab))])
        rmu, rsd = ref_desc.mean(0), ref_desc.std(0) + 1e-6
        rd = (ref_desc - rmu) / rsd
        ref_den = knn_mean_dist(rd, rd, KNN_Q + 1) * (KNN_Q + 1) / KNN_Q
        t_e = np.quantile(ref_den, [1 / 3, 2 / 3])
        d9 = np.quantile(ref_den, 0.9)
        q_desc = np.stack([scene_desc(q_lab[i]["outcomes"])
                           for i in range(len(q_lab))])
        q_den = knn_mean_dist((q_desc - rmu) / rsd, rd, KNN_Q)
        # align LatentDataset order -> cache order via token name
        lab_pos = {q_lab.items[i][0]: i for i in range(len(q_lab))}
        order = np.asarray([lab_pos[t] for t in q["tokens"]], dtype=int)
        q_den = q_den[order]
        m["tert0"] = np.digitize(q_den, t_e) == 0
        m["tert1"] = np.digitize(q_den, t_e) == 1
        m["tert2"] = np.digitize(q_den, t_e) == 2
        m["rare10"] = q_den > d9
    return m


def boot(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    mm = d[idx].mean(1)
    return (float(mm.mean()), float(np.percentile(mm, 2.5)),
            float(np.percentile(mm, 97.5)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query_cache", required=True)
    p.add_argument("--bank_cache", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--bank_tokens", default=None)
    p.add_argument("--grid", action="store_true")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--t", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--space", default="yhat_ego")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--labels_dir", default=None)
    p.add_argument("--latents_dir", default=None)
    p.add_argument("--ref_labels_dir", default=None)
    p.add_argument("--ref_latents_dir", default=None)
    p.add_argument("--ref_tokens", default=None)
    args = p.parse_args()
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokens = [l.strip() for l in open(args.tokens) if l.strip()]
    q = load_cache(args.query_cache, tokens)
    print(f"[knn] query scenes={len(q['logs'])}", flush=True)
    if args.bank_tokens:
        b_tokens = [l.strip() for l in open(args.bank_tokens) if l.strip()]
    else:
        b_tokens = [f.stem for f in Path(args.bank_cache).rglob("*.npz")]
    bank = load_cache(args.bank_cache, b_tokens)
    print(f"[knn] bank scenes={len(bank['logs'])}", flush=True)

    q_final = q["sub"][..., 5]
    b_final = bank["sub"][..., 5].reshape(-1)
    b_clog = np.repeat(bank["logs"], bank["sub"].shape[1])
    out = {"args": vars(args),
           "b0": metrics(q_final, q["b0"], q["b0"].argmax(1))}
    print(f"[knn] b0 val={out['b0']['val']:.4f}", flush=True)

    spaces = ["yhat_ego", "yhat_mean", "traj_z"]
    # z-score each space by bank stats, keep both scene-level and flat views
    qf, bf = {}, {}
    for sp in spaces:
        braw = cand_feats(bank["yhat"], bank["traj"], bank["z_pool"], sp)
        flat = braw.reshape(-1, braw.shape[-1])
        mu, sd = flat.mean(0), flat.std(0) + 1e-6
        bf[sp] = (flat - mu) / sd
        qf[sp] = (cand_feats(q["yhat"], q["traj"], q["z_pool"], sp)
                  - mu) / sd
    print("[knn] features done", flush=True)

    if args.grid:
        results = []
        for sp in spaces:
            for K in (4, 8):
                tk = topk_idx(q["b0"], K)
                qft = qf[sp][np.arange(len(tk))[:, None], tk]   # (S,K,d)
                # one retrieval pass at n=32; (n,t) variants are prefixes
                ni32, dn32 = knn_topn(qft, bf[sp], b_clog, q["logs"],
                                      n=32, device=device)
                for n in (8, 32):
                    for t in (0.05, 0.2):
                        fh = fhat_from_topn(ni32, dn32, b_final, n, t)
                        fh_s = derange_fhat(fh, q["logs"], args.seed)
                        for beta in (0, 1, 2, 4, 8, 16):
                            m = metrics(q_final, q["b0"], rerank_scores(
                                q["b0"], fh, tk, beta).argmax(1))
                            ms = metrics(q_final, q["b0"], rerank_scores(
                                q["b0"], fh_s, tk, beta).argmax(1))
                            r = dict(space=sp, K=K, n=n, t=t, beta=beta,
                                     val=m["val"], val_shuf=ms["val"],
                                     flip=m["flip"], net_fix=m["net_fix"],
                                     net_break=m["net_break"])
                            results.append(r)
                            print(f"[grid] {sp} K{K} n{n} t{t} b{beta} "
                                  f"val={m['val']:.4f} shuf={ms['val']:.4f} "
                                  f"nf={m['net_fix']:+.5f} "
                                  f"nb={m['net_break']:+.5f} "
                                  f"flip={m['flip']:.3f}", flush=True)
        out["grid"] = results
        results.sort(key=lambda r: -r["val"])
        out["best"] = results[0]
        print(f"[knn] grid best: {results[0]}", flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        return

    # ---- single config ---------------------------------------------------
    sp, K, n, t, beta = (args.space, args.K, args.n, args.t, args.beta)
    tk = topk_idx(q["b0"], K)
    qft = qf[sp][np.arange(len(tk))[:, None], tk]
    print(f"[knn] retrieving {sp} K{K} n{n} t{t} ...", flush=True)
    fh = knn_fhat(qft, bf[sp], b_final, b_clog, q["logs"], n=n, t=t,
                  device=device)
    fh_s = derange_fhat(fh, q["logs"], args.seed)
    rng = np.random.default_rng(args.seed)
    fh_r = np.zeros_like(fh)
    for i in range(len(fh)):
        ok = np.flatnonzero(b_clog != q["logs"][i])
        fh_r[i] = b_final[ok[rng.integers(0, len(ok), (K, n))]].mean(1)

    arms = {}
    for name, src in [("knn", fh), ("shuffle", fh_s), ("random", fh_r)]:
        arms[name] = rerank_scores(q["b0"], src, tk, beta).argmax(1)
    for s_pct in (25, 50, 100):
        mask = np.zeros(len(bf[sp]), bool)
        mask[np.random.default_rng(args.seed + s_pct).choice(
            len(mask), int(len(mask) * s_pct / 100), replace=False)] = True
        fh_c = knn_fhat(qft, bf[sp], b_final, b_clog, q["logs"], n=n, t=t,
                        bank_mask=mask, device=device)
        arms[f"scale{s_pct}"] = rerank_scores(
            q["b0"], fh_c, tk, beta).argmax(1)
    arms["b0"] = q["b0"].argmax(1)

    masks = strat_masks(q, tokens, args)
    f_b0v = q_final[np.arange(len(q_final)), q["b0"].argmax(1)]
    out["arms"] = {}
    for name, pk in arms.items():
        fp = q_final[np.arange(len(q_final)), pk]
        row = {mn: round(float(fp[mm].mean()), 5)
               for mn, mm in masks.items() if mm.any()}
        dv, lo, hi = boot(fp, f_b0v)
        row["diff_vs_b0"] = [round(dv, 5), round(lo, 5), round(hi, 5)]
        row["flip"] = round(float((pk != q["b0"].argmax(1)).mean()), 4)
        out["arms"][name] = row
        print(f"[eval] {name}: " + " ".join(
            f"{k}={v}" for k, v in row.items()), flush=True)

    # ---- diagnostics -----------------------------------------------------
    # b0_wrong scenes, top-8: spearman(fhat, final) vs spearman(b0, final);
    # pairwise AUC of "fhat prefers j over b0pick" against true final
    from scipy.stats import spearmanr
    Kd = 8
    tk8 = topk_idx(q["b0"], Kd)
    fh8 = knn_fhat(qf[sp][np.arange(len(tk8))[:, None], tk8],
                   bf[sp], b_final, b_clog, q["logs"], n=n, t=t,
                   device=device)
    lb = np.log(np.clip(q["b0"], 1e-6, 1 - 1e-6) /
                np.clip(1 - q["b0"], 1e-6, 1 - 1e-6))
    wrong = masks["b0_wrong"]
    sp_f, sp_b = [], []
    auc_hit = auc_tot = acc_hit = acc_tot = 0
    b0_pick = q["b0"].argmax(1)
    for i in np.flatnonzero(wrong):
        cand = tk8[i]
        fnl = q_final[i][cand]
        if fnl.std() < 1e-9:
            continue
        sf = spearmanr(fh8[i], fnl).correlation
        sb = spearmanr(lb[i][cand], fnl).correlation
        if np.isfinite(sf):
            sp_f.append(sf)
        if np.isfinite(sb):
            sp_b.append(sb)
        pj = int(np.flatnonzero(cand == b0_pick[i])[0])
        for j in range(Kd):
            if j == pj:
                continue
            better = fnl[j] > fnl[pj]
            prefer = fh8[i][j] > fh8[i][pj]
            auc_hit += float(prefer == better)
            auc_tot += 1
            if prefer:
                acc_hit += float(better)
                acc_tot += 1
    out["diag"] = dict(
        n_wrong=int(wrong.sum()),
        spearman_fhat=[round(float(np.mean(sp_f)), 4),
                       round(float(np.percentile(sp_f, 25)), 4),
                       round(float(np.percentile(sp_f, 75)), 4)],
        spearman_b0=[round(float(np.mean(sp_b)), 4),
                     round(float(np.percentile(sp_b, 25)), 4),
                     round(float(np.percentile(sp_b, 75)), 4)],
        pairwise_auc=round(auc_hit / max(auc_tot, 1), 4),
        prefer_accuracy=round(acc_hit / max(acc_tot, 1), 4))
    print(f"[diag] {out['diag']}", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"[knn] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
