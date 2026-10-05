#!/usr/bin/env python
"""Step 4 trainer: KeyNet (L_ret) + MemoryDelta on frozen b3 features.

Data = cache_b3_feats.py output (per-token npz with a/z_pool/yhat/yt/sub/b0/
traj/key_src/o/o_mask). Memory bank = navtrain cache (all logs; same-log
queries excluded at retrieval). Queries = train split; val split for ckpt.

Losses (per spec):
    L_main = listwise CE vs softmax(true_final/0.05)
           + pairwise hinge on |Dfinal|>0.2 pairs
    L_c    = max(0, mu - (L(M~) - L(M))), M~ = randomly swapped neighbours
    L_ret  = KL(softmax(w_ab) || softmax(cos(g_a,g_b)/t)) in-batch, cross-log
    memory dropout p=0.2; hard-scene weight x5 (B0 top1 final < best-0.05)
    ckpt by val selection-level final -> model_best.pt

    python scripts/experience/train_memory.py \
        --cache_dir $E/b3cache_navtrain --labels_dir $E/navtrain_labels_full \
        --latents_dir $E/latents_navtrain \
        --train_tokens .../train_tokens.txt --val_tokens .../val_tokens.txt \
        --mode joint --epochs 10 --batch_size 256 --lr 1e-4 --seed 0 \
        --out_dir $E/mem_runs/mem_s0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from train_ewm import build_index  # noqa: E402
from eval_ewm import KNN_Q, knn_mean_dist, scene_desc  # noqa: E402
from navsim.agents.drive_jepa_perception_based.experience.ewm_memory import (  # noqa: E402
    K_RETR,
    KeyNet,
    MemoryBank,
    MemoryDelta,
    MemTokenEnc,
    listwise_ce,
    make_memory_tokens,
    pairwise_hinge,
    retrieval_kl,
    score_with_delta,
)

MU = 0.05          # memory-dependence margin
P_DROP = 0.2       # memory token dropout
T_O = 1.0          # w_ab temperature
T_G = 0.1          # cosine temperature
HARD_W = 5.0


class CacheDataset(Dataset):
    """Per-scene cached b3 features."""

    def __init__(self, tokens, cache_dir: Path, labels_dir: Path = None,
                 lat_index: dict = None):
        self.items = []
        cache_dir = Path(cache_dir)
        for t in tokens:
            for lp in cache_dir.glob(f"*/{t}.npz"):
                self.items.append(lp)
        # fall back to recursive scan result order; sort for determinism
        self.items.sort()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        z = np.load(self.items[i], allow_pickle=True)
        d = {k: z[k] for k in z.files}
        return d


def collate_cache(batch):
    keys = [k for k in batch[0] if k not in ("log_name",)]
    out = {k: torch.from_numpy(np.stack([np.asarray(b[k]) for b in batch])
                                 .astype(np.float32))
           for k in keys}
    out["log_names"] = [str(b["log_name"]) for b in batch]
    return out


def build_bank(ds: CacheDataset, keynet, device) -> MemoryBank:
    z_pool, key, yt, sub, b0, traj, logs = [], [], [], [], [], [], []
    for b in DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                        collate_fn=collate_cache):
        z_pool.append(b["z_pool"].numpy())
        key.append(b["key_src"].numpy())
        yt.append(b["yt"].numpy())
        sub.append(b["sub"].numpy())
        b0.append(b["b0"].numpy())
        traj.append(b["traj"].numpy())
        logs.extend(b["log_names"])
    bank = MemoryBank(np.concatenate(key), np.concatenate(z_pool), logs,
                      np.concatenate(yt).reshape(len(logs), -1, 5 * 64),
                      np.concatenate(sub), np.concatenate(b0),
                      np.concatenate(traj))
    if keynet is not None:
        with torch.no_grad():
            ks = torch.from_numpy(bank.key_src).to(device)
            bank.g_keys = torch.cat([keynet(ks[i:i + 4096]).cpu()
                                     for i in range(0, len(ks), 4096)]
                                    ).numpy()
    return bank


def selection_final(scores: np.ndarray, final: np.ndarray) -> float:
    return float(final[np.arange(len(final)), scores.argmax(1)].mean())


def eval_selection(bank, keynet, memenc, delta, ds, device,
                   key_mode="lret", shuffle=False, scale=1.0, seed=0):
    """Vectorised selection eval on a CacheDataset; returns per-scene
    (final_chosen, labels, conflict scenes idx arrays for subsets)."""
    keynet.eval(); memenc.eval(); delta.eval()
    key, zp, a, yh, sub, b0, logs = [], [], [], [], [], [], []
    for b in DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                        collate_fn=collate_cache):
        key.append(b["key_src"].numpy()); zp.append(b["z_pool"].numpy())
        a.append(b["a"].numpy()); yh.append(b["yhat"].numpy())
        sub.append(b["sub"].numpy()); b0.append(b["b0"].numpy())
        logs.extend(b["log_names"])
    key = np.concatenate(key); zp = np.concatenate(zp)
    a = np.concatenate(a); yh = np.concatenate(yh)
    sub = np.concatenate(sub); b0 = np.concatenate(b0)
    logs = np.asarray(logs)

    if scale < 1.0:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(bank.logs),
                                  int(len(bank.logs) * scale), replace=False))
        bank = MemoryBank(bank.key_src[keep], bank.z_pool[keep],
                          bank.logs[keep], bank.yt_flat[keep],
                          bank.sub[keep], bank.b0[keep], bank.traj[keep],
                          bank.g_keys[keep] if bank.g_keys is not None else None)

    if key_mode == "lret":
        with torch.no_grad():
            qg = torch.cat([keynet(torch.from_numpy(key[i:i + 4096]).to(device))
                            for i in range(0, len(key), 4096)]).cpu().numpy()
            bg = torch.cat([keynet(torch.from_numpy(
                bank.key_src[i:i + 4096]).to(device))
                for i in range(0, len(bank.key_src), 4096)]).cpu().numpy()
        bank.g_keys = bg
        cand = bank.stage1(key)
        nb = bank.stage2(qg, cand, logs)
    elif key_mode == "appearance":
        nb = bank.retrieve_appearance(key, logs)
    elif key_mode == "random":
        nb = bank.retrieve_random(logs, seed=seed)
    else:
        raise ValueError(key_mode)

    g = bank.gather(nb)
    B, kk, K = g["sub"].shape[:3]
    finals = np.zeros((B, K), np.float32)
    chunk = 128
    with torch.no_grad():
        for s in range(0, B, chunk):
            sl = slice(s, min(s + chunk, B))
            gs = {k: v[sl] for k, v in g.items()}
            mem = make_memory_tokens(memenc, gs, device)
            if shuffle:
                perm = torch.randperm(mem.shape[1], device=device)
                mem = mem[:, perm]
            # mask invalid scenes' tokens
            valid = torch.from_numpy(gs["valid"]).to(device).float()
            valid = valid[:, :, None].expand(-1, -1, K).reshape(
                valid.shape[0], -1, 1)
            mem = mem * valid
            at = torch.from_numpy(a[sl]).to(device)
            yf = torch.from_numpy(yh[sl]).to(device).reshape(
                at.shape[0], at.shape[1], -1)
            dl = delta(at, yf, mem)
            sc = score_with_delta(torch.from_numpy(b0[sl]).to(device), dl)
            finals[sl] = torch.from_numpy(sub[sl])[
                torch.arange(len(sc)), sc.argmax(1)].numpy()
    return finals, sub, b0, logs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--train_tokens", required=True)
    p.add_argument("--val_tokens", required=True)
    p.add_argument("--mode", choices=["joint", "key_only", "delta_only"],
                   default="joint")
    p.add_argument("--key_ckpt", default=None,
                   help="pretrained KeyNet state for delta_only/joint-warm")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lam_ret", type=float, default=1.0)
    p.add_argument("--lam_hinge", type=float, default=1.0)
    p.add_argument("--lam_c", type=float, default=1.0)
    p.add_argument("--k", type=int, default=K_RETR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_scenes", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = []

    train_tokens = [l.strip() for l in open(args.train_tokens) if l.strip()]
    val_tokens = [l.strip() for l in open(args.val_tokens) if l.strip()]
    if args.max_scenes:
        train_tokens = train_tokens[: args.max_scenes]
        val_tokens = val_tokens[: max(256, args.max_scenes // 8)]
    cache = Path(args.cache_dir)
    train_ds = CacheDataset(train_tokens, cache)
    val_ds = CacheDataset(val_tokens, cache)
    print(f"[mem] train={len(train_ds)} val={len(val_ds)}", flush=True)

    # bank = everything in cache dir (navtrain all), matching spec
    bank_ds = CacheDataset([t for t in train_tokens + val_tokens], cache)
    keynet = KeyNet().to(device)
    memenc = MemTokenEnc().to(device)
    delta = MemoryDelta().to(device)
    if args.key_ckpt:
        keynet.load_state_dict(torch.load(args.key_ckpt, map_location="cpu"))
        print(f"[mem] loaded key_ckpt {args.key_ckpt}", flush=True)
    bank = build_bank(bank_ds, keynet, device)
    print(f"[mem] bank scenes={len(bank.logs)}", flush=True)

    params = []
    if args.mode in ("joint", "delta_only"):
        params += list(memenc.parameters()) + list(delta.parameters())
    if args.mode in ("joint", "key_only"):
        params += list(keynet.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    # per-token log ids for L_ret cross-log check
    log_lut = {}
    def log_id(l):
        if l not in log_lut:
            log_lut[l] = len(log_lut)
        return log_lut[l]

    best_val = -1.0
    for ep in range(args.epochs):
        t0 = time.time()
        keynet.train(); memenc.train(); delta.train()
        ep_loss = ep_ret = ep_c = ep_main = 0.0
        nb = 0
        loader = DataLoader(train_ds, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            collate_fn=collate_cache, drop_last=True)
        for b in loader:
            B = b["key_src"].shape[0]
            ks = b["key_src"].to(device)
            q_logs = np.asarray(b["log_names"])

            # ---- L_ret on in-batch o(x) ------------------------------------
            lret = ks.new_zeros(())
            if args.mode in ("joint", "key_only") and "o" in b:
                lid = torch.as_tensor([log_id(l) for l in b["log_names"]],
                                      device=device)
                lret = retrieval_kl(keynet, ks, b["o"].to(device),
                                    b["o_mask"].to(device).bool(), lid,
                                    t=T_G, T=T_O)

            lmain = lce = lhinge = lc = ks.new_zeros(())
            if args.mode in ("joint", "delta_only"):
                # refresh bank g keys (cheap) then retrieve
                with torch.no_grad():
                    bg = torch.cat([keynet(torch.from_numpy(
                        bank.key_src[i:i + 4096]).to(device))
                        for i in range(0, len(bank.key_src), 4096)]).cpu().numpy()
                    bank.g_keys = bg
                    qg = keynet(ks).detach().cpu().numpy()
                cand = bank.stage1(ks.cpu().numpy())
                nb_s = bank.stage2(qg, cand, q_logs, k=args.k)
                g = bank.gather(nb_s)
                mem = make_memory_tokens(memenc, g, device)
                valid = torch.from_numpy(g["valid"]).to(device).float()
                Kq = b["sub"].shape[1]
                valid = valid[:, :, None].expand(-1, -1, Kq).reshape(
                    B, -1, 1)
                # memory dropout p=0.2
                drop = (torch.rand(mem.shape[:2], device=device) > P_DROP
                        ).float().unsqueeze(-1)
                mem = mem * drop * valid

                at = b["a"].to(device)
                yf = b["yhat"].to(device).flatten(2)
                dl = delta(at, yf, mem)
                scores = score_with_delta(b["b0"].to(device), dl)
                final = b["sub"][..., 5].to(device)

                # hard-scene weight: B0 top1 missed by >0.05
                b0_pick = b["b0"].argmax(1)
                f_best = final.max(1).values
                f_b0 = final.gather(1, b0_pick[:, None].to(device)).squeeze(1)
                hw = torch.where(f_b0 < f_best - 0.05,
                                 torch.tensor(HARD_W, device=device),
                                 torch.tensor(1.0, device=device))

                lce = listwise_ce(scores, final, weight=hw)
                lhinge = pairwise_hinge(scores, final, weight=hw)
                lmain = lce + args.lam_hinge * lhinge

                # memory-dependence constraint: random-swap neighbours
                if args.lam_c > 0:
                    nb_shuf = bank.retrieve_random(q_logs, k=args.k,
                                                   seed=np.random.randint(1e9))
                    gs = bank.gather(nb_shuf)
                    mem_s = make_memory_tokens(memenc, gs, device)
                    dl_s = delta(at, yf, mem_s)
                    sc_s = score_with_delta(b["b0"].to(device), dl_s)
                    l_s = listwise_ce(sc_s, final, weight=hw)
                    lc = F.relu(MU - (l_s - lce))
                    lmain = lmain + args.lam_c * lc

            loss = lmain + args.lam_ret * lret
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss); ep_ret += float(lret)
            ep_c += float(lc); ep_main += float(lce)
            nb += 1

        # ---- val: selection final ------------------------------------------
        vf, sub, b0v, _ = eval_selection(bank, keynet, memenc, delta,
                                         val_ds, device)
        rec = dict(epoch=ep, loss=ep_loss / nb, lret=ep_ret / nb,
                   lmain=ep_main / nb, lc=ep_c / nb,
                   val_final=vf.mean(), secs=round(time.time() - t0, 1))
        hist.append(rec)
        print(f"[mem] ep{ep} loss={rec['loss']:.4f} lret={rec['lret']:.4f} "
              f"lc={rec['lc']:.4f} val_final={rec['val_final']:.4f} "
              f"({rec['secs']}s)", flush=True)
        ck = dict(args=vars(args), keynet=keynet.state_dict(),
                  memenc=memenc.state_dict(), delta=delta.state_dict(),
                  epoch=ep, val_final=rec["val_final"])
        torch.save(ck, out_dir / "model.pt")
        if rec["val_final"] > best_val:
            best_val = rec["val_final"]
            torch.save(ck, out_dir / "model_best.pt")
        (out_dir / "history.json").write_text(json.dumps(hist, indent=1))
    print(f"[mem] DONE best_val_final={best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
