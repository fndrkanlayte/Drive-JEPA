#!/usr/bin/env python
"""Dump the EWM readout head's per-candidate probabilities for B0 proposals.

For each scene we forward the 32 B0 proposals through a trained arm and store
sigmoid(out["readout"]) (K,6) in the labels column order
[NC, DAC, EP, TTC, Comfort, final] — the readout head is trained with BCE
against `labels` (subscores) directly, so its columns follow the labels order.

The dump keeps every row aligned with its token: `tokens` comes from
dump_tokens(ds) (the DataLoader's sorted item order), `final` is
labels[...,5] taken from the same batch, and `subs` is the full true
subscore matrix so downstream analysis needs no re-join.
"""
import argparse
import json
import sys
from pathlib import Path

import multiprocessing as mp

try:
    mp.set_sharing_strategy("file_system")
except Exception:  # pragma: no cover - platform default
    pass

import numpy as np
import torch
from torch.utils.data import DataLoader

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_ewm import LatentDataset, build_index, collate  # noqa: E402
from eval_ewm import KNN_Q, knn_mean_dist, load_model, scene_desc  # noqa: E402
from eval_yhat import dump_tokens  # noqa: E402


@torch.no_grad()
def collect_readout(model, loader, device):
    """-> readout probs (S,K,6), labels (S,K,6), pdm (S,K), log per scene."""
    rd, ll, pp, lg = [], [], [], []
    for b in loader:
        out = model(b["image_feature"].to(device),
                    b["proposal_feature"].to(device),
                    b["proposals"].to(device))
        key = "readout" if "readout" in out else "logits"
        rd.append(torch.sigmoid(out[key]).float().cpu().numpy())
        ll.append(b["labels"].numpy())
        pp.append(b["pdm_score"].numpy())
        lg.extend(b["log_names"])
    return (np.concatenate(rd), np.concatenate(ll), np.concatenate(pp),
            np.asarray(lg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_dir", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--ref_latents_dir", required=True)
    p.add_argument("--ref_labels_dir", required=True)
    p.add_argument("--train_tokens", required=True)
    p.add_argument("--tokens", required=True, help="query token list")
    p.add_argument("--run", required=True, help="run dir with checkpoint")
    p.add_argument("--ckpt", default="model_best.pt")
    p.add_argument("--dump_npz", required=True)
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
    d9_edge = np.quantile(ref_density, 0.9)

    lat_index = build_index(Path(args.latents_dir))
    tokens = [l.strip() for l in open(args.tokens) if l.strip()]
    ds = LatentDataset(tokens, lat_index, Path(args.labels_dir))
    desc = np.stack([scene_desc(ds[i]["outcomes"]) for i in range(len(ds))])
    density = knn_mean_dist((desc - mu) / sd, ref_desc, KNN_Q)
    rare10 = density > d9_edge
    print(f"[eval] query scenes={len(ds)} rare10={int(rare10.sum())}", flush=True)

    q_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate)
    model, direct, tag = load_model(Path(args.run), device,
                                  ckpt_file=args.ckpt)
    print(f"[eval] model tag={tag} ckpt={args.ckpt}", flush=True)
    rd, labels, pdm, q_logs = collect_readout(model, q_loader, device)
    print(f"[eval] readout collected {rd.shape}", flush=True)

    np.savez(
        args.dump_npz,
        tokens=np.asarray(dump_tokens(ds)),
        pdm=pdm,
        final=labels[..., 5].astype(np.float32),
        subs=labels.astype(np.float32),
        readout=rd.astype(np.float32),
        rare10=rare10,
        logs=q_logs,
    )
    report = {"tag": tag, "scenes": len(ds), "rare10": int(rare10.sum())}
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
