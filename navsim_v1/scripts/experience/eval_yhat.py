#!/usr/bin/env python
"""T2: does the learned y_hat (ego slot) carry outcome information B0 misses?

Metrics (per run):
  M1  within-scene action sensitivity: Spearman(pairwise ||dy_ego||,
      pairwise ||dlabels||) per scene (geometry_bank.py G3 definition);
      plus cross-log top-16 kNN finalMAE in y_ego space.
  M2  within B0 top-K (K=4,8 by pdm_score): pairwise AUC of f_hat (cross-log
      kNN estimate of final) and of B0 score against true final
      (pairs with final_i != final_j only). Splits: all / b0_wrong / rare10.
  M3  partial Spearman of f_hat vs true final controlling for B0 score,
      within top-K.
  Control: bank finals shuffled within scene -> M2 should fall to ~0.5.

Memory = train split B0 proposal y_hat. With --bank_npz, a second memory
variant adds bank trajectories (B0 + bank) for models trained with bank
(A2/A3) -- requires the model be --no_pfeat.

Usage:
    python scripts/experience/eval_yhat.py \
        --latents_dir $E/latents_navtrain --labels_dir $E/navtrain_labels_full \
        --ref_latents_dir $E/latents_navtrain --ref_labels_dir $E/navtrain_labels_full \
        --train_tokens $E/split_navtrain/train_tokens.txt \
        --tokens $E/split_navtrain/val_tokens.txt \
        --runs $E/ewm_runs/full_b3_s0 \
        --bank_npz $E/clover/bank_ours.npz --out_json out.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
from scipy.stats import rankdata
from torch.utils.data import DataLoader

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_ewm import (  # noqa: E402
    LatentDataset, build_index, collate, load_bank_npz, knn_val_mae,
)
from eval_ewm import (  # noqa: E402
    KNN_Q, knn_mean_dist, load_model, scene_desc, spearman,
)


@torch.no_grad()
def collect_query(model, loader, device):
    """-> y_full (S,K,n_sl,L), slot_logit (S,K,n_ag) or None,
       labels (S,K,6), pdm (S,K), log per scene (S,), pred_logit.

    For unstructured models y_hat is (B,K,L) and we add a length-1 slot
    axis so downstream code can treat both cases uniformly."""
    ys, sl_, ll, pp, lg, pl = [], [], [], [], [], []
    for b in loader:
        out = model(b["image_feature"].to(device),
                    b["proposal_feature"].to(device),
                    b["proposals"].to(device))
        yh = out["y_hat"].float().cpu().numpy()
        ys.append(yh if yh.ndim == 4 else yh[:, :, None, :])
        if "slot_logit" in out:
            sl_.append(out["slot_logit"].float().cpu().numpy())
        ll.append(b["labels"].numpy())
        pp.append(b["pdm_score"].numpy())
        pl.append(b["pred_logit"].numpy())
        lg.extend(b["log_names"])
    return (np.concatenate(ys),
            np.concatenate(sl_) if sl_ else None,
            np.concatenate(ll), np.concatenate(pp),
            np.asarray(lg), np.concatenate(pl))


@torch.no_grad()
def collect_bank(model, loader, device, bank=None):
    """-> y (N,L), finals (N,), cand-level logs (N,), scene tokens (N,).

    With ``bank`` (token->(trajs,subs)), each scene's bank trajectories are
    forwarded as extra candidates (model must be no_pfeat) and appended with
    their true subscore finals.
    """
    ys, sl_, fin, lg, sc, sub = [], [], [], [], [], []
    for b in loader:
        img = b["image_feature"].to(device)
        out = model(img, b["proposal_feature"].to(device),
                    b["proposals"].to(device))
        yh = out["y_hat"].float().cpu().numpy()
        ys.append(yh if yh.ndim == 4 else yh[:, :, None, :])
        if "slot_logit" in out:
            sl_.append(out["slot_logit"].float().cpu().numpy()
                       .reshape(-1, out["slot_logit"].shape[-1]))
        fin.append(b["labels"][..., 5].numpy())
        sub.append(b["labels"].numpy().reshape(-1, 6))
        for i, ln in enumerate(b["log_names"]):
            lg.extend([ln] * b["labels"].shape[1])
            sc.extend([b["tokens"][i]] * b["labels"].shape[1])
        if bank is not None:
            bt_list, bs_list, have = [], [], []
            for i, t in enumerate(b["tokens"]):
                e = bank.get(t)
                if e is not None and len(e[0]) > 0:
                    have.append(i)
                    bt_list.append(e[0])
                    bs_list.append(e[1])
            if have:
                im = img[have]
                # scenes may have different NB -> fall back to per-scene loop
                # if ragged
                lens = [len(x) for x in bt_list]
                if len(set(lens)) == 1:
                    B = im.shape[0]
                    bt = torch.from_numpy(np.concatenate(bt_list)).float().to(device)
                    out2 = model(im, torch.zeros(B, lens[0], 256, device=device),
                                 bt.view(B, lens[0], 8, 3))
                    yb = out2["y_hat"].float().cpu().numpy()
                    yb = yb if yb.ndim == 4 else yb[:, :, None, :]
                    if "slot_logit" in out2:
                        sl_.append(out2["slot_logit"].float().cpu().numpy()
                                   .reshape(-1, out2["slot_logit"].shape[-1]))
                    for i, s in enumerate(have):
                        ys.append(yb[i])
                        fin.append(np.asarray(bs_list[i])[:, 5])
                        sub.append(np.asarray(bs_list[i])[:, :6])
                        lg.extend([b["log_names"][s]] * lens[i])
                        sc.extend([b["tokens"][s]] * lens[i])
                else:
                    for i, s in enumerate(have):
                        tr = torch.from_numpy(bt_list[i]).float().to(device)
                        out2 = model(im[i:i + 1],
                                     torch.zeros(1, tr.shape[0], 256, device=device),
                                     tr.unsqueeze(0))
                        yh2 = out2["y_hat"].float().cpu().numpy()
                        ys.append(yh2[0] if yh2.ndim == 4
                                  else yh2[0, :, None, :])
                        if "slot_logit" in out2:
                            sl_.append(out2["slot_logit"].float().cpu().numpy()
                                       .reshape(-1, out2["slot_logit"].shape[-1]))
                        fin.append(np.asarray(bs_list[i])[:, 5])
                        sub.append(np.asarray(bs_list[i])[:, :6])
                        lg.extend([b["log_names"][s]] * lens[i])
                        sc.extend([b["tokens"][s]] * lens[i])
    return (np.concatenate([a.reshape(-1, a.shape[-2], a.shape[-1])
                            for a in ys]),
            np.concatenate(sl_) if sl_ else None,
            np.concatenate([np.asarray(f).reshape(-1) for f in fin]),
            np.asarray(lg), np.asarray(sc),
            np.concatenate(sub).astype(np.float32))


def action_sensitivity(y_ego, labels):
    """Per-scene Spearman of pairwise latent distances vs label L2 distances."""
    vals = []
    for s in range(len(y_ego)):
        R = torch.from_numpy(y_ego[s]).float()
        dr = torch.cdist(R, R).numpy()
        O = labels[s]
        do = np.linalg.norm(O[:, None, :] - O[None, :, :], axis=-1)
        iu = np.triu_indices(len(R), 1)
        if np.std(dr[iu]) == 0 or np.std(do[iu]) == 0:
            continue
        vals.append(spearman(dr[iu], do[iu]))
    vals = np.asarray(vals)
    return float(np.nanmean(vals)), int(np.sum(~np.isnan(vals)))


def pairwise_auc(pred, true):
    """P(pred orders a pair the same as true) over pairs with true_i != true_j."""
    a, b = np.triu_indices(len(pred), 1)
    d = true[a] - true[b]
    keep = d != 0
    a, b, d = a[keep], b[keep], d[keep]
    if len(d) == 0:
        return float("nan"), 0
    dp = pred[a] - pred[b]
    conc = np.sum(dp * d > 0)
    tie = np.sum(dp == 0)
    return float((conc + 0.5 * tie) / len(d)), int(len(d))


def partial_spearman(x, y, c):
    """Spearman(x,y | c): residuals of ranks regressed on rank(c), then Pearson."""
    rx, ry, rc = rankdata(x), rankdata(y), rankdata(c)
    A = np.stack([rc, np.ones_like(rc)], 1)
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if np.std(ex) == 0 or np.std(ey) == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def topk_indices(pdm_row, k):
    return np.argpartition(-pdm_row, k - 1)[:k]


def knn_readout(q_y, q_logs, b_y, b_subs, b_logs, k=16, t=0.1, device="cpu",
                whiten="meanstd"):
    """Per-query cross-log top-k cosine kNN -> dict of weighted subscore
    readouts: fhat (w·final), p_nc/p_dac/p_ttc (w·frac(sub<1)), f_ep (w·EP),
    cos (mean top-k similarity, masked neighbours clamped to 0).

    whiten="meanstd": centre+scale query and bank ŷ by the bank's per-dim
    mean/std before the cosine (removes the dominant common direction).
    whiten="off": raw ŷ (previous behaviour)."""
    if whiten == "meanstd":
        mu = b_y.mean(0)
        sd = b_y.std(0) + 1e-6
        q_y = (q_y - mu) / sd
        b_y = (b_y - mu) / sd
    elif whiten != "off":
        raise ValueError(f"unknown whiten mode {whiten!r}")
    qn = q_y / (np.linalg.norm(q_y, axis=1, keepdims=True) + 1e-8)
    bn = b_y / (np.linalg.norm(b_y, axis=1, keepdims=True) + 1e-8)
    bt = torch.from_numpy(bn).to(device)
    bs = torch.from_numpy(b_subs).float().to(device)
    uniq = np.unique(np.concatenate([np.asarray(q_logs), np.asarray(b_logs)]))
    lid = {v: i for i, v in enumerate(uniq)}
    ql = torch.from_numpy(np.array([lid[v] for v in q_logs])).to(device)
    bl = torch.from_numpy(np.array([lid[v] for v in b_logs])).to(device)
    n = len(qn)
    out = {k_: np.zeros(n, np.float32) for k_ in
           ("fhat", "p_nc", "p_dac", "p_ttc", "p_any", "f_ep", "cos")}
    CH = 1024
    for s in range(0, n, CH):
        qc = torch.from_numpy(qn[s:s + CH]).to(device)
        sim = qc @ bt.T
        same = ql[s:s + CH, None] == bl[None, :]
        sim = sim.masked_fill(same, -1e9)
        tk = sim.topk(min(k, sim.shape[1]), dim=-1)
        w = torch.softmax(tk.values / t, dim=-1)
        nb = bs[tk.indices]                       # (ch,k,6)
        sl = slice(s, s + CH)
        out["fhat"][sl] = (w * nb[..., 5]).sum(-1).cpu().numpy()
        out["p_nc"][sl] = (w * (nb[..., 0] < 1).float()).sum(-1).cpu().numpy()
        out["p_dac"][sl] = (w * (nb[..., 1] < 1).float()).sum(-1).cpu().numpy()
        out["p_ttc"][sl] = (w * (nb[..., 3] < 1).float()).sum(-1).cpu().numpy()
        out["p_any"][sl] = (w * ((nb[..., 0] < 1) | (nb[..., 1] < 1)
                                 | (nb[..., 3] < 1)).float()).sum(-1).cpu().numpy()
        out["f_ep"][sl] = (w * nb[..., 2]).sum(-1).cpu().numpy()
        out["cos"][sl] = tk.values.clamp(-1, 1).mean(-1).cpu().numpy()
    return out


def dump_tokens(ds):
    """Tokens in the same order as DataLoader output (ds.items is sorted
    internally) so every dump row stays aligned with its token."""
    return [it[0] for it in ds.items]


def slot_stats(b_y_full):
    """Whitening stats per slot from the train memory (N,n_sl,L)."""
    return {"eg_mu": b_y_full[:, 0].mean(0),
            "eg_sd": b_y_full[:, 0].std(0) + 1e-6,
            "ag_mu": b_y_full[:, 1:].mean(0),
            "ag_sd": b_y_full[:, 1:].std(0) + 1e-6}


def make_key(y_full, slot_logit, key, stats=None):
    """Retrieval key per candidate.

    ego:        slot-0 vector, returned unchanged (knn_readout whitens).
    agent:      per-slot whiten (bank stats), * sigmoid(slot_logit), concat
                -> already whitened, call knn_readout with whiten='off'.
    ego_agent:  ego whitened L2-normed concat agent-key L2-normed.
    """
    if key == "ego":
        return y_full[:, 0]
    if key == "pool":
        # slot-order-invariant pooled agent key (pre-normalized)
        import torch
        from navsim.agents.drive_jepa_perception_based.experience. \
            ewm_structured import pool_key
        if slot_logit is None:
            raise SystemExit("--key pool requires slot_logit (b3 only)")
        y = torch.as_tensor(y_full, dtype=torch.float32)
        sl = torch.as_tensor(slot_logit, dtype=torch.float32)
        return pool_key(y, sl).numpy()
    ag = (y_full[:, 1:] - stats["ag_mu"]) / stats["ag_sd"]
    if slot_logit is not None:
        p = 1.0 / (1.0 + np.exp(-slot_logit))
        ag = ag * p[..., None]
    agk = ag.reshape(len(y_full), -1)
    if key == "agent":
        return agk
    eg = (y_full[:, 0] - stats["eg_mu"]) / stats["eg_sd"]
    eg = eg / (np.linalg.norm(eg, axis=1, keepdims=True) + 1e-8)
    agn = agk / (np.linalg.norm(agk, axis=1, keepdims=True) + 1e-8)
    return np.concatenate([eg, agn], axis=1)


def knn_fhat(q_y, q_logs, b_y, b_fin, b_logs, k=16, t=0.1, device="cpu",
             whiten="meanstd"):
    b_subs = np.zeros((len(b_fin), 6), np.float32)
    b_subs[:, 5] = b_fin
    return knn_readout(q_y, q_logs, b_y, b_subs, b_logs, k, t, device,
                       whiten=whiten)["fhat"]


def run_metrics(tag, y_ego, q_sl, labels, pdm, q_logs, bank_variants, tert,
                rare10, device, dumps=None, whiten="meanstd", key="ego"):
    """bank_variants: {name: (b_y_full, b_sl, b_fin, b_logs, b_scene, b_subs)}

    y_ego is the full-slot query tensor (S,K,n_sl,L); q_sl its slot logits
    (S,K,n_ag) or None."""
    res = {}
    S, K = labels.shape[:2]

    # M1a: within-scene action sensitivity (always on the ego slot)
    sp, nok = action_sensitivity(y_ego[:, :, 0], labels)
    res["M1_action_sensitivity"] = dict(spearman=sp, n_scenes=nok)

    # candidate-level flat arrays for kNN
    q_yf = y_ego.reshape(-1, y_ego.shape[-2], y_ego.shape[-1])
    q_slf = q_sl.reshape(-1, q_sl.shape[-1]) if q_sl is not None else None
    q_fin = labels[..., 5].reshape(-1)
    q_cand_logs = np.repeat(q_logs, K)

    # b0_wrong per K: B0's top1 final is strictly below the top-K best final
    b0_wrong = {}
    for Ksel in (4, 8):
        m = np.zeros(S, bool)
        for s in range(S):
            tk = topk_indices(pdm[s], Ksel)
            m[s] = labels[s, pdm[s].argmax(), 5] < \
                labels[s, tk, 5].max() - 1e-6
        b0_wrong[Ksel] = m

    for bname, (b_yf, b_sl, b_fin, b_logs, b_scene,
                b_subs) in bank_variants.items():
        if key == "ego":
            q_key, b_key, wmode = q_yf[:, 0], b_yf[:, 0], whiten
        else:
            st = slot_stats(b_yf)
            q_key = make_key(q_yf, q_slf, key, st)
            b_key = make_key(b_yf, b_sl, key, st)
            wmode = "off"
        rd = knn_readout(q_key, q_cand_logs, b_key, b_subs, b_logs,
                         device=device, whiten=wmode)
        fhat_flat = rd["fhat"]
        fhat = fhat_flat.reshape(S, K)
        # shuffle control: permute whole subscore rows within each bank SCENE
        rng = np.random.default_rng(0)
        b_subs_sh = b_subs.copy()
        for sc in np.unique(b_scene):
            m = b_scene == sc
            b_subs_sh[m] = b_subs_sh[m][rng.permutation(int(m.sum()))]
        rd_sh = knn_readout(q_key, q_cand_logs, b_key, b_subs_sh, b_logs,
                            device=device, whiten=wmode)
        fhat_sh = rd_sh["fhat"].reshape(S, K)
        if dumps is not None:
            d = {"fhat": fhat.astype(np.float32),
                 "fhat_shuf": fhat_sh.astype(np.float32)}
            for k_ in ("p_nc", "p_dac", "p_ttc", "p_any", "f_ep", "cos"):
                d[k_] = rd[k_].reshape(S, K).astype(np.float32)
                d[k_ + "_shuf"] = rd_sh[k_].reshape(S, K).astype(np.float32)
            dumps[bname] = d

        # M1b: cross-log kNN finalMAE
        mae = float(np.abs(fhat_flat - q_fin).mean())

        m2, m3 = {}, {}
        for Ksel in (4, 8):
            scene_mask = {"all": np.ones(S, bool),
                          "b0_wrong": b0_wrong[Ksel], "rare10": rare10}
            for sub, msk in scene_mask.items():
                auc_f, auc_b, ps = [], [], []
                npairs = 0
                for s in np.where(msk)[0]:
                    tk = topk_indices(pdm[s], Ksel)
                    af, n1 = pairwise_auc(fhat[s][tk], labels[s, tk, 5])
                    ab, n2 = pairwise_auc(pdm[s][tk], labels[s, tk, 5])
                    if not np.isnan(af):
                        auc_f.append(af); npairs += n1
                    if not np.isnan(ab):
                        auc_b.append(ab)
                    ps.append(partial_spearman(fhat[s][tk], labels[s, tk, 5],
                                               pdm[s][tk]))
                m2[f"K{Ksel}/{sub}"] = dict(
                    auc_fhat=float(np.mean(auc_f)) if auc_f else float("nan"),
                    auc_b0=float(np.mean(auc_b)) if auc_b else float("nan"),
                    pairs=npairs)
                m3[f"K{Ksel}/{sub}"] = dict(
                    partial_spearman=float(np.nanmean(ps)))
        # shuffle-control AUC (K=8, all)
        auc_c = []
        for s in range(S):
            tk = topk_indices(pdm[s], 8)
            af, _ = pairwise_auc(fhat_sh[s][tk], labels[s, tk, 5])
            if not np.isnan(af):
                auc_c.append(af)
        res[f"M1_knn_finalMAE/{bname}"] = mae
        res[f"M2_pairwise_auc/{bname}"] = m2
        res[f"M3_partial_spearman/{bname}"] = m3
        res[f"M2_shuffle_control_K8/{bname}"] = float(np.mean(auc_c))
        cv = rd["cos"]
        res[f"knn_cos_stats/{bname}"] = {
            "mean": float(cv.mean()), "std": float(cv.std()),
            "p05": float(np.quantile(cv, .05)),
            "p50": float(np.quantile(cv, .50)),
            "p95": float(np.quantile(cv, .95))}
        print(f"[{tag}/{bname}] finalMAE={mae:.4f} "
              f"auc_f(all,K8)={m2['K8/all']['auc_fhat']:.3f} "
              f"auc_b0={m2['K8/all']['auc_b0']:.3f} "
              f"shuf={res[f'M2_shuffle_control_K8/{bname}']:.3f}", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_dir", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--ref_latents_dir", required=True)
    p.add_argument("--ref_labels_dir", required=True)
    p.add_argument("--train_tokens", required=True)
    p.add_argument("--tokens", required=True, help="query token list")
    p.add_argument("--runs", required=True, help="comma-separated run dirs")
    p.add_argument("--ckpt", default="model.pt")
    p.add_argument("--bank_npz", default=None)
    p.add_argument("--whiten", default="meanstd",
                   choices=("meanstd", "off"),
                   help="ŷ centre+scale by train-memory stats before cosine")
    p.add_argument("--key", default="ego",
                   choices=("ego", "agent", "ego_agent", "pool"),
                   help="retrieval key: ego slot only (default), agent slots "
                        "1..4 whitened+exist-weighted concat, or both")
    p.add_argument("--mem_cache_dir", default=None,
                   help="cache train-memory y_hat (all slots) per run/bank "
                        "variant as npz so later runs/splits reuse it")
    p.add_argument("--dump_npz", default=None,
                   help="per-query dump: tokens/pdm/final/fhat(+shuf)/rare10 "
                        "per run & bank variant, for rerank_yhat.py")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tokens = [l.strip() for l in open(args.train_tokens) if l.strip()]

    ref_ds = LatentDataset(train_tokens, build_index(Path(args.ref_latents_dir)),
                           Path(args.ref_labels_dir))
    ref_desc = np.stack([scene_desc(ref_ds[i]["outcomes"]) for i in range(len(ref_ds))])
    mu, sd = ref_desc.mean(0), ref_desc.std(0) + 1e-6
    ref_desc = (ref_desc - mu) / sd
    ref_density = knn_mean_dist(ref_desc, ref_desc, KNN_Q + 1) * (KNN_Q + 1) / KNN_Q
    t_edges = np.quantile(ref_density, [1 / 3, 2 / 3])
    d9_edge = np.quantile(ref_density, 0.9)

    lat_index = build_index(Path(args.latents_dir))
    tokens = [l.strip() for l in open(args.tokens) if l.strip()]
    ds = LatentDataset(tokens, lat_index, Path(args.labels_dir))
    desc = np.stack([scene_desc(ds[i]["outcomes"]) for i in range(len(ds))])
    density = knn_mean_dist((desc - mu) / sd, ref_desc, KNN_Q)
    tert = np.digitize(density, t_edges)
    rare10 = density > d9_edge
    print(f"[eval] query scenes={len(ds)} rare10={int(rare10.sum())}", flush=True)

    q_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate)
    ref_loader = DataLoader(ref_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)

    bank = load_bank_npz(args.bank_npz) if args.bank_npz else None
    report = {}
    dump_runs = {}
    pdm_all, final_all, b0_all, labels_full = {}, {}, {}, {}
    for run in args.runs.split(","):
        model, direct, tag = load_model(Path(run), device, ckpt_file=args.ckpt)
        y_full, q_sl, labels, pdm, q_logs, pred_logit = collect_query(
            model, q_loader, device)
        if args.key != "ego" and y_full.shape[2] == 1:
            raise ValueError(f"--key {args.key} needs agent slots; model "
                             f"{tag} has an unstructured y_hat")
        pdm_all[tag] = pdm.copy()
        final_all[tag] = labels[..., 5].copy()
        labels_full[tag] = labels.copy()
        b0_all[tag] = (1.0 / (1.0 + np.exp(-pred_logit))).astype(np.float32)

        safe_tag = tag.replace(":", "_").replace("/", "_")

        def bank_variant(bname, use_bank):
            cp = None
            if args.mem_cache_dir:
                cp = (Path(args.mem_cache_dir)
                      / f"mem_{safe_tag}_{bname}.npz")
                if cp.exists():
                    z = np.load(cp, allow_pickle=True)
                    return (z["y_full"],
                            z["slot_logit"] if "slot_logit" in z.files
                            else None,
                            z["fin"], z["logs"], z["scene"], z["subs"])
            print(f"[{tag}] collecting train bank"
                  + (" + clover trajs ..." if use_bank else " ..."),
                  flush=True)
            by, bs_, bf, bl, bsc, bu = collect_bank(
                model, ref_loader, device, bank=bank if use_bank else None)
            if cp is not None:
                cp.parent.mkdir(parents=True, exist_ok=True)
                tmp = cp.with_name(cp.stem + ".tmp.npz")
                np.savez(tmp, y_full=by, fin=bf, logs=bl, scene=bsc, subs=bu,
                         **({"slot_logit": bs_} if bs_ is not None else {}))
                os.replace(tmp, cp)
            return (by, bs_, bf, bl, bsc, bu)

        variants = {"b0prop": bank_variant("b0prop", False)}
        if bank is not None and getattr(model.action, "no_pfeat", False):
            variants["b0prop+bank"] = bank_variant("b0prop+bank", True)

        dumps = {} if args.dump_npz else None
        report[tag] = run_metrics(tag, y_full, q_sl, labels, pdm.copy(),
                                  q_logs, variants, tert, rare10, device,
                                  dumps=dumps, whiten=args.whiten,
                                  key=args.key)
        if dumps is not None:
            dump_runs[tag] = dumps

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=1))
    if args.dump_npz:
        for tag, dumps in dump_runs.items():
            pth = Path(args.dump_npz.format(tag=tag))
            pdm_t, final_t = pdm_all[tag], final_all[tag]
            bw = {}
            for ks in (2, 4, 8):
                bw[f"b0_wrong_K{ks}"] = np.array([
                    final_t[s, pdm_t[s].argmax()] <
                    final_t[s, topk_indices(pdm_t[s], ks)].max() - 1e-6
                    for s in range(len(pdm_t))])
            np.savez(
                pth,
                tokens=np.asarray(dump_tokens(ds)),
                rare10=rare10.astype(bool),
                pdm=pdm_t,
                final=final_t,
                subs=labels_full[tag],
                b0_sub=b0_all[tag],
                **bw,
                **{f"{bn}__{k}": v for bn, d in dumps.items()
                   for k, v in d.items()},
            )
            print(f"[eval] dumped {pth}", flush=True)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
