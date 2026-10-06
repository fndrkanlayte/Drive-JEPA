#!/usr/bin/env python
"""Step 4 pre-pass: cache frozen b3 outputs per scene.

Saves one npz per token under <out_dir>/<log>/<token>.npz with:
    a        (32,256)  f16   trunk action embeddings
    z_pool   (256,)    f16   pooled scene z
    yhat     (32,5,64) f16   predicted outcome latents (slot0=ego)
    yt       (32,5,64) f16   encode_outcome(ema=True) of real outcome
    sub      (32,6)    f32   true subscores
    b0       (32,)     f32   B0 pdm_score
    traj     (32,8,3)  f32   candidate proposals
    key_src  (512,)    f16   [mean-pool(img), mean-pool(pf)]
    o        (64,6)    f16   action-aligned outcome signature (if vocab given)
    o_mask   (64,)     bool

    python scripts/experience/cache_b3_feats.py \
        --latents_dir $E/latents_navtrain --labels_dir $E/navtrain_labels_full \
        --tokens $E/navtrain_all.txt --b3_run $E/ewm_runs/full_b3_s0 \
        --vocab $E/proposal_vocab_64.npy --out_dir $E/b3cache_navtrain \
        --batch_size 32 --num_workers 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from train_ewm import LatentDataset, build_index, collate  # noqa: E402
from eval_ewm import load_model  # noqa: E402
from navsim.agents.drive_jepa_perception_based.experience.ewm_memory import (  # noqa: E402
    scene_o,
)

p = argparse.ArgumentParser()
p.add_argument("--latents_dir", required=True)
p.add_argument("--labels_dir", required=True)
p.add_argument("--tokens", required=True)
p.add_argument("--b3_run", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--vocab", default=None, help="proposal_vocab_64.npy + thr file")
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--num_workers", type=int, default=8)
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, tag = load_model(Path(args.b3_run), device)
for q in model.parameters():
    q.requires_grad_(False)

vocab = thr = None
if args.vocab:
    vz = np.load(args.vocab)
    vocab, thr = vz["vocab"], float(vz["thresh"])

tokens = [l.strip() for l in open(args.tokens) if l.strip()]
ds = LatentDataset(tokens, build_index(Path(args.latents_dir)),
                   Path(args.labels_dir), structured=True)
print(f"[cache] scenes={len(ds)} model={tag}", flush=True)
out_root = Path(args.out_dir)

n_done = n_skip = 0
with torch.no_grad():
    for b in DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate):
        img = b["image_feature"].to(device)
        pf = b["proposal_feature"].to(device)
        tr = b["proposals"].to(device)
        out = model(img, pf, tr)
        a, z = model.forward_trunk(img, pf, tr)
        yt = model.encode_outcome(b["agent_vals"].to(device),
                                  b["agent_valid"].to(device),
                                  b["agent_slot"].to(device),
                                  b["labels"].to(device), ema=True)
        key = torch.cat([img.mean(1), pf.mean(1)], -1).cpu().numpy()
        for i, tok in enumerate(b["tokens"]):
            log = b["log_names"][i]
            d = out_root / log
            d.mkdir(parents=True, exist_ok=True)
            fp = d / f"{tok}.npz"
            if fp.exists():
                n_skip += 1
                continue
            kv = dict(
                a=a[i].cpu().numpy().astype(np.float16),
                z_pool=z[i].mean(0).cpu().numpy().astype(np.float16),
                yhat=out["y_hat"][i].cpu().numpy().astype(np.float16),
                yt=yt[i].cpu().numpy().astype(np.float16),
                sub=b["labels"][i].numpy().astype(np.float32),
                outcomes=b["outcomes"][i].numpy().astype(np.float32),
                b0=b["pdm_score"][i].numpy().astype(np.float32),
                traj=b["proposals"][i].numpy().astype(np.float32),
                key_src=key[i].astype(np.float16),
                log_name=log,
            )
            if vocab is not None:
                o, om = scene_o(b["proposals"][i].numpy(),
                                b["labels"][i].numpy(), vocab, thr)
                kv.update(o=o.astype(np.float16), o_mask=om)
            np.savez_compressed(fp, **kv)
            n_done += 1
        if (n_done + n_skip) % 200 == 0:
            print(f"[cache] {n_done} saved {n_skip} skipped", flush=True)
print(f"[cache] DONE {n_done} saved {n_skip} skipped", flush=True)
