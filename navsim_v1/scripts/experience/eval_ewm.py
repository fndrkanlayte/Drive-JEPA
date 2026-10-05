#!/usr/bin/env python
"""Step 3 eval: candidate-level val metrics + navtest selection metrics.

Usage:
    # candidate-level on navtrain val
    python scripts/experience/eval_ewm.py --mode val \
        --latents_dir $E/latents_navtrain --labels_dir $E/navtrain_labels \
        --train_tokens $E/split_navtrain/train_tokens.txt \
        --tokens $E/split_navtrain/val_tokens.txt \
        --runs $E/ewm_runs/b1_s0,$E/ewm_runs/b2_s0

    # navtest selection (existing navtest labels, latents_navtest)
    python scripts/experience/eval_ewm.py --mode navtest \
        --latents_dir $E/latents_navtest --labels_dir $E/navtest_labels \
        --ref_latents_dir $E/latents_navtrain --ref_labels_dir $E/navtrain_labels \
        --train_tokens $E/split_navtrain/train_tokens.txt \
        --runs $E/ewm_runs/b2_s0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))

from navsim.agents.drive_jepa_perception_based.experience.descriptors import (  # noqa: E402
    DESCRIPTOR_FIELD_INDEX,
    TIMING_FIELDS,
)
from navsim.agents.drive_jepa_perception_based.experience.ewm import EWMJEPA  # noqa: E402

from train_ewm import (  # noqa: E402
    LatentDataset,
    build_model,
    auprc,
    build_index,
    collate,
    spearman,
)

TIMING_IDX = np.array([DESCRIPTOR_FIELD_INDEX[f] for f in TIMING_FIELDS])
KNN_Q = 20


def load_model(run_dir: Path, device, ckpt_name: str = "model.pt"):
    ckpt = torch.load(Path(run_dir) / ckpt_name, map_location="cpu",
                      weights_only=False)
    name = ckpt["args"]["model"]
    direct = name in ("b1", "b1aux")
    model = build_model(name, ckpt["args"]["n_layers"],
                        use_future=ckpt["args"].get("use_future", False)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # tag by run dir so several seeds of one arm do not overwrite each other
    return model, direct, f"{name}:{Path(run_dir).name}"


@torch.no_grad()
def infer(model, loader, device, direct):
    """-> list of per-scene dicts with model probs (K,6) + labels + meta."""
    rows = []
    for b in loader:
        img = b["image_feature"].to(device)
        pf = b["proposal_feature"].to(device)
        tr = b["proposals"].to(device)
        out = model(img, pf, tr)
        logits = out["logits"] if "logits" in out else out["readout"]
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = b["labels"].numpy()
        outs = b["outcomes"].numpy()
        pdm = b["pdm_score"].numpy()
        for i, t in enumerate(b["tokens"]):
            rows.append(dict(token=t, probs=probs[i], labels=labels[i],
                             outcomes=outs[i], pdm=pdm[i]))
    return rows


@torch.no_grad()
def embed(model, loader, device):
    """-> z_pool (S,D), a (S,K,D), labels (S,K,6), outcomes, log_names, pdm."""
    zs, aa, ll, oo, lg, pp = [], [], [], [], [], []
    for b in loader:
        img = b["image_feature"].to(device)
        pf = b["proposal_feature"].to(device)
        tr = b["proposals"].to(device)
        a, z = model.forward_trunk(img, pf, tr)
        zs.append(z.mean(1).cpu().numpy())
        aa.append(a.cpu().numpy())
        ll.append(b["labels"].numpy())
        oo.append(b["outcomes"].numpy())
        lg.extend(b["log_names"])
        pp.append(b["pdm_score"].numpy())
    return (np.concatenate(zs), np.concatenate(aa), np.concatenate(ll),
            np.concatenate(oo), np.asarray(lg), np.concatenate(pp))


def lknn_probs(q_z, q_a, q_logs, b_z, b_a, b_lab, b_logs, k: int = 8):
    """L-kNN: per query scene retrieve k=8 train neighbours by cosine on
    pooled z (same log excluded), then score each candidate with the label
    of its nearest neighbour in the joint (z, a_k) embedding."""
    S, K, D = q_a.shape
    qz_n = q_z / (np.linalg.norm(q_z, axis=1, keepdims=True) + 1e-8)
    bz_n = b_z / (np.linalg.norm(b_z, axis=1, keepdims=True) + 1e-8)
    sim = qz_n @ bz_n.T                                    # (S,T)
    probs = np.zeros((S, K, b_lab.shape[-1]), dtype=np.float32)
    for i in range(S):
        s = sim[i].copy()
        s[b_logs == q_logs[i]] = -np.inf
        nb = np.argpartition(-s, k)[:k]                    # k neighbour scenes
        nb_a = b_a[nb].reshape(-1, D)                      # (k*K, D)
        nb_z = np.repeat(b_z[nb], K, axis=0)               # (k*K, D)
        nb_joint = np.concatenate([nb_z, nb_a], axis=-1)   # (k*K, 2D)
        nb_lab = b_lab[nb].reshape(-1, b_lab.shape[-1])    # (k*K, 6)
        q_joint = np.concatenate(
            [np.repeat(q_z[i:i + 1], K, axis=0), q_a[i]], axis=-1)  # (K, 2D)
        nj = nb_joint / (np.linalg.norm(nb_joint, axis=1, keepdims=True) + 1e-8)
        qj = q_joint / (np.linalg.norm(q_joint, axis=1, keepdims=True) + 1e-8)
        nn = (qj @ nj.T).argmax(1)                         # (K,)
        probs[i] = nb_lab[nn]
    return probs


def scene_desc(outcomes: np.ndarray) -> np.ndarray:
    """Scene descriptor = mean over candidates of the 12 timing features."""
    return np.nanmean(outcomes[:, : len(TIMING_FIELDS)], axis=0)  # (12,)


def knn_mean_dist(q: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
    """Mean euclid distance to k nearest ref points (per query row)."""
    ref_n2 = (ref ** 2).sum(1)[None, :]
    q_n2 = (q ** 2).sum(1)[:, None]
    d2 = np.maximum(q_n2 + ref_n2 - 2 * (q @ ref.T), 0.0)
    part = np.partition(d2, min(k, ref.shape[0] - 1), axis=1)[:, :k]
    return np.sqrt(part).mean(1)


def bootstrap_ci(vals: np.ndarray, n: int = 1000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    m = vals[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["val", "navtest"], required=True)
    p.add_argument("--latents_dir", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--tokens", default=None, help="bare-token list (val mode)")
    p.add_argument("--train_tokens", required=True,
                   help="train tokens for density reference")
    p.add_argument("--ref_latents_dir", default=None,
                   help="latents dir of train reference (navtest mode)")
    p.add_argument("--ref_labels_dir", default=None)
    p.add_argument("--runs", required=True, help="comma-separated run dirs")
    p.add_argument("--lknn", action="store_true",
                   help="also score L-kNN (needs a B1 run in --runs)")
    p.add_argument("--lknn_k", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out_json", default=None)
    p.add_argument("--ckpt", default="model.pt", help="model.pt (last epoch) or model_best.pt (val-selected)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- reference density (train scenes) ----------------------------------
    train_tokens = [l.strip() for l in open(args.train_tokens) if l.strip()]
    ref_lat_dir = Path(args.ref_latents_dir or args.latents_dir)
    ref_lab_dir = Path(args.ref_labels_dir or args.labels_dir)
    ref_index = build_index(ref_lat_dir)
    ref_ds = LatentDataset(train_tokens, ref_index, ref_lab_dir)
    ref_desc = []
    for i in range(len(ref_ds)):
        ref_desc.append(scene_desc(ref_ds[i]["outcomes"]))
    ref_desc = np.stack(ref_desc)
    mu, sd = ref_desc.mean(0), ref_desc.std(0) + 1e-6
    ref_desc = (ref_desc - mu) / sd
    # self-kNN includes the query itself (d=0): take KNN_Q+1 and rescale
    ref_density = knn_mean_dist(ref_desc, ref_desc, KNN_Q + 1) * (KNN_Q + 1) / KNN_Q
    t_edges = np.quantile(ref_density, [1 / 3, 2 / 3])
    d9_edge = np.quantile(ref_density, 0.9)
    print(f"[eval] ref scenes={len(ref_ds)} tertiles={t_edges} d9={d9_edge:.4f}",
          flush=True)

    # ---- eval tokens -------------------------------------------------------
    if args.mode == "val":
        tokens = [l.strip() for l in open(args.tokens) if l.strip()]
    else:
        tokens = None  # all tokens present in labels+latents
    lat_index = build_index(Path(args.latents_dir))
    if tokens is None:
        tokens = sorted(set(lat_index) &
                        {f.stem for f in Path(args.labels_dir).glob("*.npz")})
    ds = LatentDataset(tokens, lat_index, Path(args.labels_dir))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)
    ref_loader = DataLoader(ref_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)
    desc = np.stack([scene_desc(ds[i]["outcomes"]) for i in range(len(ds))])
    desc = (desc - mu) / sd
    density = knn_mean_dist(desc, ref_desc, KNN_Q)
    tert = np.digitize(density, t_edges)          # 0/1/2 sparse->dense
    rare10 = density > d9_edge
    print(f"[eval] eval scenes={len(ds)}", flush=True)

    # ---- per-run inference + metrics ---------------------------------------
    report = {}
    for run in args.runs.split(","):
        model, direct, tag = load_model(Path(run), device, args.ckpt)
        rows = infer(model, loader, device, direct)

        probs = np.stack([r["probs"] for r in rows])      # (S,K,6)
        labels = np.stack([r["labels"] for r in rows])    # (S,K,6)
        pdm = np.stack([r["pdm"] for r in rows])          # (S,K)
        S, K = probs.shape[:2]

        sel_model = probs[..., 5].argmax(1)
        sel_native = pdm.argmax(1)
        sel_blend_cache = {}
        s_flip = sel_model != sel_native

        # candidate-level metrics ------------------------------------------------
        p_flat = probs.reshape(-1, 6)
        l_flat = labels.reshape(-1, 6)
        dens_flat = np.repeat(tert, K)
        res = {}
        for name, mask in [("all", np.ones(S * K, bool)),
                           ("tert0", dens_flat == 0),
                           ("tert1", dens_flat == 1),
                           ("tert2", dens_flat == 2)]:
            res[name] = dict(
                nc_auprc=auprc(1 - p_flat[mask, 0], l_flat[mask, 0] < 1),
                ttc_auprc=auprc(1 - p_flat[mask, 3], l_flat[mask, 3] < 1),
                spearman=spearman(p_flat[mask, 5], l_flat[mask, 5]),
            )
        res["top1_unsafe_model"] = float(np.mean(
            labels[np.arange(S), sel_model, 0] < 1))
        res["top1_unsafe_native"] = float(np.mean(
            labels[np.arange(S), sel_native, 0] < 1))
        res["s_flip_frac"] = float(np.mean(s_flip))

        # scene-level selected-candidate outcomes ---------------------------------
        def sel_metrics(sel, mask):
            idx = np.where(mask)[0]
            if len(idx) == 0:
                return {}
            lab_sel = labels[idx, sel[idx]]              # (n,6)
            out = {}
            for j, nm in enumerate(["NC", "DAC", "EP", "TTC", "C", "final"]):
                lo, hi = bootstrap_ci(lab_sel[:, j])
                out[nm] = dict(mean=float(lab_sel[:, j].mean()), ci=[lo, hi])
            return out

        subsets = dict(
            all=np.ones(S, bool),
            conflict=np.asarray([(r["outcomes"][:, 0] > 0).any() for r in rows]),
            rare10=rare10,
            flip=s_flip,
        )
        sel_res = {nm: sel_metrics(sel_model, m) for nm, m in subsets.items()}
        nat_res = {nm: sel_metrics(sel_native, m) for nm, m in subsets.items()}
        res["selection_model"] = sel_res
        res["selection_native"] = nat_res

        # L-kNN baseline (uses the B1 model's embeddings) ---------------------
        if args.lknn and direct:
            print(f"[lknn] embedding bank from {run}", flush=True)
            b_z, b_a, b_lab, _, b_logs, _ = embed(model, ref_loader, device)
            q_z, q_a, _, _, q_logs, _ = embed(model, loader, device)
            kp = lknn_probs(q_z, q_a, q_logs, b_z, b_a, b_lab, b_logs,
                            k=args.lknn_k)
            sel_knn = kp[..., 5].argmax(1)
            p_flat_k = kp.reshape(-1, kp.shape[-1])
            kres = {}
            for name, mask in [("all", np.ones(S * K, bool)),
                               ("tert0", dens_flat == 0),
                               ("tert1", dens_flat == 1),
                               ("tert2", dens_flat == 2)]:
                kres[name] = dict(
                    nc_auprc=auprc(1 - p_flat_k[mask, 0], l_flat[mask, 0] < 1),
                    ttc_auprc=auprc(1 - p_flat_k[mask, 3], l_flat[mask, 3] < 1),
                    spearman=spearman(p_flat_k[mask, 5], l_flat[mask, 5]),
                )
            kres["top1_unsafe_model"] = float(np.mean(
                labels[np.arange(S), sel_knn, 0] < 1))
            kres["selection_model"] = {
                nm: sel_metrics(sel_knn, m) for nm, m in subsets.items()}
            report[f"lknn_{tag}"] = kres
            print(f"[lknn] nc_auprc={kres['all']['nc_auprc']:.4f} "
                  f"top1_unsafe={kres['top1_unsafe_model']:.4f}", flush=True)

        # alpha blend scan (navtest) -------------------------------------------------
        if args.mode == "navtest":
            blends = {}
            for a in np.arange(0.0, 1.01, 0.1):
                pdm_n = (pdm - pdm.min(1, keepdims=True)) / \
                    (pdm.max(1, keepdims=True) - pdm.min(1, keepdims=True) + 1e-8)
                sc = a * probs[..., 5] + (1 - a) * pdm_n
                sel = sc.argmax(1)
                lab_sel = labels[np.arange(S), sel]
                blends[f"{a:.1f}"] = dict(
                    final=float(lab_sel[:, 5].mean()),
                    NC=float(lab_sel[:, 0].mean()),
                    TTC=float(lab_sel[:, 3].mean()),
                    EP=float(lab_sel[:, 2].mean()),
                )
            res["blend_scan"] = blends

        report[tag or run] = res
        print(f"[eval] {run}: nc_auprc={res['all']['nc_auprc']:.4f} "
              f"ttc={res['all']['ttc_auprc']:.4f} "
              f"top1_unsafe={res['top1_unsafe_model']:.4f} "
              f"(native {res['top1_unsafe_native']:.4f})", flush=True)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v.get("all", v) for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
