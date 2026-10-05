#!/usr/bin/env python
"""Build the action-aligned proposal vocabulary for o(x).

k-means (K=64, fixed seed) over ALL navtrain candidates' proposals flattened
to 16 dims (8 poses x,y in ego frame). Saves proposal_vocab_64.npy as npz:
    vocab  (64,16) cluster centres
    thresh scalar = 75th percentile of nearest-centre distance over a sample

    python scripts/experience/build_proposal_vocab.py \
        --latents_dir $E/latents_navtrain --tokens $E/navtrain_all.txt \
        --out $E/proposal_vocab_64.npy [--sample_scenes 4000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_ewm import build_index  # noqa: E402
sys.path.insert(0, str(HERE.parent.parent))
from navsim.agents.drive_jepa_perception_based.experience.ewm_memory import (  # noqa: E402
    build_proposal_vocab,
)

p = argparse.ArgumentParser()
p.add_argument("--latents_dir", required=True)
p.add_argument("--tokens", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n_clusters", type=int, default=64)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--sample_scenes", type=int, default=4000,
               help="scenes sampled for k-means fit + thresh (0 = all)")
args = p.parse_args()

tokens = [l.strip() for l in open(args.tokens) if l.strip()]
index = build_index(Path(args.latents_dir))
tokens = [t for t in tokens if t in index]
rng = np.random.default_rng(args.seed)
if args.sample_scenes and len(tokens) > args.sample_scenes:
    tokens = list(rng.choice(tokens, args.sample_scenes, replace=False))
print(f"[vocab] fitting on {len(tokens)} scenes", flush=True)

props = []
for t in tokens:
    props.append(np.asarray(np.load(index[t])["proposals"], np.float32))
props = np.concatenate(props)                       # (S*32,8,3)
flat = props[:, :, :2].reshape(len(props), -1)

vocab = build_proposal_vocab(props, args.n_clusters, args.seed)
d = ((flat[:, None, :] - vocab[None]) ** 2).sum(-1)
thr = float(np.quantile(np.sqrt(d.min(1)), 0.75))
np.savez(args.out, vocab=vocab, thresh=thr,
         n_scenes=len(tokens), n_clusters=args.n_clusters)
print(f"[vocab] saved {args.out} thresh={thr:.3f}", flush=True)
