#!/usr/bin/env python
"""Split cached latent tokens into train/val by log (85/15, seeded).

Scans ``--latents_dir`` for ``<log>/<token>.npz`` shards, groups by log,
seeds an 85/15 log-level split, and writes bare-token lists:

    python scripts/experience/split_logs.py \
        --latents_dir $NAVSIM_EXP_ROOT/experience/latents_navtrain \
        --out_dir $NAVSIM_EXP_ROOT/experience/split_navtrain --seed 0
"""

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--latents_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.15)
    args = p.parse_args()

    latents_dir = Path(args.latents_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_tokens = {}
    for log_dir in sorted(latents_dir.iterdir()):
        if not log_dir.is_dir():
            continue
        toks = [f.stem for f in log_dir.glob("*.npz")]
        if toks:
            log_tokens[log_dir.name] = toks

    import numpy as np

    rng = np.random.default_rng(args.seed)
    logs = sorted(log_tokens)
    order = rng.permutation(len(logs))
    n_val = int(round(len(logs) * args.val_frac))
    val_logs = {logs[i] for i in order[:n_val]}
    train_logs = set(logs) - val_logs

    train_tokens = [t for l in train_logs for t in log_tokens[l]]
    val_tokens = [t for l in val_logs for t in log_tokens[l]]

    (out_dir / "train_tokens.txt").write_text("\n".join(sorted(train_tokens)) + "\n")
    (out_dir / "val_tokens.txt").write_text("\n".join(sorted(val_tokens)) + "\n")
    (out_dir / "train_logs.txt").write_text("\n".join(sorted(train_logs)) + "\n")
    (out_dir / "val_logs.txt").write_text("\n".join(sorted(val_logs)) + "\n")
    print(f"[split] {len(logs)} logs -> {len(train_logs)} train / {len(val_logs)} val")
    print(f"[split] {len(train_tokens)} train tokens / {len(val_tokens)} val tokens")


if __name__ == "__main__":
    main()
