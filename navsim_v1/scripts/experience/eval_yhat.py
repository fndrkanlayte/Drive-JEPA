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
import sys
from pathlib import Path

import numpy as np
import torch
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
    """-> y_ego (S,K,L), labels (S,K,6), pdm (S,K), log per scene (S,)."""
    ys, ll, pp, lg = [], [], [], []
    for b in loader:
        out = model(b["image_feature"].to(device),
                    b["proposal_feature"].to(device),
                    b["proposals"].to(device))
        ys.append(out["y_hat"][:, :, 0].float().cpu().numpy())
        ll.append(b["labels"].numpy())
        pp.append(b["pdm_score"].numpy())
        lg.extend(b["log_names"])
    return (np.concatenate(ys), np.concatenate(ll), np.concatenate(pp),
            np.asarray(lg))


@torch.no_grad()
def collect_bank(model, loader, device, bank=None):
    """-> y (N,L), finals (N,), cand-level logs (N,), scene tokens (N,).

    With ``bank`` (token->(trajs,subs)), each scene's bank trajectories are
    forwarded as extra candidates (model must be no_pfeat) and appended with
    their true subscore finals.
    """
    ys, fin, lg, sc = [], [], [], []
    for b in loader:
        img = b["image_feature"].to(device)
        out = model(img, b["proposal_feature"].to(device),
                    b["proposals"].to(device))
        ys.append(out["y_hat"][:, :, 0].float().cpu().numpy())
        fin.append(b["labels"][..., 5].numpy())
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
                    yb = out2["y_hat"][:, :, 0].float().cpu().numpy()
                    for i, s in enumerate(have):
                        ys.append(yb[i])
                        fin.append(np.asarray(bs_list[i])[:, 5])
                        lg.extend([b["log_names"][s]] * lens[i])
                        sc.extend([b["tokens"][s]] * lens[i])
                else:
                    for i, s in enumerate(have):
                        tr = torch.from_numpy(bt_list[i]).float().to(device)
                        out2 = model(im[i:i + 1],
                                     torch.zeros(1, tr.shape[0], 256, device=device),
                                     tr.unsqueeze(0))
                        ys.append(out2["y_hat"][0, :, 0].float().cpu().numpy())
                        fin.append(np.asarray(bs_list[i])[:, 5])
                        lg.extend([b["log_names"][s]] * lens[i])
                        sc.extend([b["tokens"][s]] * lens[i])
    return (np.concatenate([a.reshape(-1, a.shape[-1]) for a in ys]),
            np.concatenate([np.asarray(f).reshape(-1) for f in fin]),
            np.asarray(lg), np.asarray(sc))


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


def knn_fhat(q_y, q_logs, b_y, b_fin, b_logs, k=16, t=0.1, device="cpu"):
    """Per-query cross-log top-k cosine kNN -> weighted f_hat (N,)."""
    qn = q_y / (np.linalg.norm(q_y, axis=1, keepdims=True) + 1e-8)
    bn = b_y / (np.linalg.norm(b_y, axis=1, keepdims=True) + 1e-8)
    bt = torch.from_numpy(bn).to(device)
    bf = torch.from_numpy(b_fin).float().to(device)
    uniq = np.unique(np.concatenate([np.asarray(q_logs), np.asarray(b_logs)]))
    lid = {v: i for i, v in enumerate(uniq)}
    ql = torch.from_numpy(np.array([lid[v] for v in q_logs])).to(device)
    bl = torch.from_numpy(np.array([lid[v] for v in b_logs])).to(device)
    out = np.zeros(len(qn), np.float32)
    CH = 1024
    for s in range(0, len(qn), CH):
        qc = torch.from_numpy(qn[s:s + CH]).to(device)
        sim = qc @ bt.T
        same = ql[s:s + CH, None] == bl[None, :]
        sim = sim.masked_fill(same, -1e9)
        tk = sim.topk(min(k, sim.shape[1]), dim=-1)
        w = torch.softmax(tk.values / t, dim=-1)
        out[s:s + CH] = (w * bf[tk.indices]).sum(-1).cpu().numpy()
    return out


def run_metrics(tag, y_ego, labels, pdm, q_logs, bank_variants, tert, rare10,
                device):
    """bank_variants: {name: (b_y, b_fin, b_logs, b_scene)}"""
    res = {}
    S, K = labels.shape[:2]

    # M1a: within-scene action sensitivity
    sp, nok = action_sensitivity(y_ego, labels)
    res["M1_action_sensitivity"] = dict(spearman=sp, n_scenes=nok)

    # candidate-level flat arrays for kNN
    q_y = y_ego.reshape(-1, y_ego.shape[-1])
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

    for bname, (b_y, b_fin, b_logs, b_scene) in bank_variants.items():
        fhat_flat = knn_fhat(q_y, q_cand_logs, b_y, b_fin, b_logs, device=device)
        fhat = fhat_flat.reshape(S, K)
        # shuffle control: permute finals within each bank SCENE (token)
        rng = np.random.default_rng(0)
        b_fin_sh = b_fin.copy()
        for sc in np.unique(b_scene):
            m = b_scene == sc
            b_fin_sh[m] = rng.permutation(b_fin[m])
        fhat_sh = knn_fhat(q_y, q_cand_logs, b_y, b_fin_sh, b_logs, device=device)
        fhat_sh = fhat_sh.reshape(S, K)

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
    for run in args.runs.split(","):
        model, direct, tag = load_model(Path(run), device, ckpt_file=args.ckpt)
        y_ego, labels, pdm, q_logs = collect_query(model, q_loader, device)

        # bank memory: train split B0 proposals
        print(f"[{tag}] collecting train bank ...", flush=True)
        b_y, b_fin, b_logs, b_scene = collect_bank(model, ref_loader, device)
        variants = {"b0prop": (b_y, b_fin, b_logs, b_scene)}
        if bank is not None and getattr(model.action, "no_pfeat", False):
            print(f"[{tag}] collecting train bank + clover trajs ...", flush=True)
            # second pass reusing same loader for bank trajectories
            by2, bf2, bl2, bs2 = collect_bank(model, ref_loader, device,
                                             bank=bank)
            variants["b0prop+bank"] = (by2, bf2, bl2, bs2)

        report[tag] = run_metrics(tag, y_ego, labels, pdm, q_logs, variants,
                                  tert, rare10, device)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
