#!/usr/bin/env python
"""Step 4 eval: memory-augmented selection on navtest (or val).

Arms:
    b0          native argmax on pdm_score
    lret        learned g retrieval + MemoryDelta (main method)
    appearance  appearance-cosine retrieval + MemoryDelta
    random      random scenes + MemoryDelta
    shuffle     lret retrieval, neighbour lists deranged across scenes
    scale{25,50}  lret with 25/50% of bank scenes

Strata (same names as Step 3 eval): all / conflict / rare10 / flip / tert0-2
    conflict = (outcomes[:,0] > 0).any() over candidates
    flip     = arm's argmax pick != B0's pick, computed per arm
    plus extra "b0_wrong" (B0 top1 final < best - 0.05)
Paired scene bootstrap 95% CI vs B0.

    python scripts/experience/eval_memory.py \
        --query_cache $E/b3cache_navtest --bank_cache $E/b3cache_navtrain \
        --labels_dir $E/navtest_labels_full --latents_dir $E/latents_navtest \
        --tokens /tmp/navtest_all.txt \
        --ref_labels_dir $E/navtrain_labels_full \
        --ref_latents_dir $E/latents_navtrain \
        --ref_tokens $E/split_navtrain/train_tokens.txt \
        --run $E/mem_runs/mem_s0 --out_json $E/mem_runs/eval_navtest_s0.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from train_ewm import LatentDataset, build_index  # noqa: E402
from eval_ewm import KNN_Q, knn_mean_dist, scene_desc  # noqa: E402
from train_memory import (CacheDataset, build_bank,  # noqa: E402
                          eval_selection)
from navsim.agents.drive_jepa_perception_based.experience.ewm_memory import (  # noqa: E402
    KeyNet, MemoryDelta, MemTokenEnc)

p = argparse.ArgumentParser()
p.add_argument("--query_cache", required=True)
p.add_argument("--bank_cache", required=True)
p.add_argument("--labels_dir", required=True)
p.add_argument("--latents_dir", required=True)
p.add_argument("--tokens", required=True)
p.add_argument("--ref_labels_dir", required=True)
p.add_argument("--ref_latents_dir", required=True)
p.add_argument("--ref_tokens", required=True)
p.add_argument("--run", required=True, help="memory run dir (model_best.pt)")
p.add_argument("--arms",
               default="b0,lret,appearance,random,shuffle,scale25,scale50")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out_json", default=None)
args = p.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(Path(args.run) / "model_best.pt", map_location="cpu",
                  weights_only=False)
ck_args = ckpt.get("args", {})
no_latent = bool(ck_args.get("no_latent", False))
readout = ck_args.get("readout", "resid")
keynet = KeyNet().to(device); keynet.load_state_dict(ckpt["keynet"])
memenc = MemTokenEnc(no_latent=no_latent).to(device)
memenc.load_state_dict(ckpt["memenc"])
delta = MemoryDelta(no_latent=no_latent, readout=readout).to(device)
delta.load_state_dict(ckpt["delta"])
keynet.eval(); memenc.eval(); delta.eval()
for m in (keynet, memenc, delta):
    for q in m.parameters():
        q.requires_grad_(False)
print(f"[eval] ckpt ep{ckpt['epoch']} val_final={ckpt['val_final']:.4f}",
      flush=True)

# ---- query set ---------------------------------------------------------------
tokens = [l.strip() for l in open(args.tokens) if l.strip()]
q_ds = CacheDataset(tokens, Path(args.query_cache))
print(f"[eval] query scenes={len(q_ds)}", flush=True)

bank_tokens = [f.stem for f in Path(args.bank_cache).rglob("*.npz")]
bank_ds = CacheDataset(bank_tokens, Path(args.bank_cache))
bank = build_bank(bank_ds, keynet, device)
print(f"[eval] bank scenes={len(bank.logs)}", flush=True)

# ---- tertile/rare10 split (same protocol as eval_ewm) -------------------------
q_lab = LatentDataset(tokens, build_index(Path(args.latents_dir)),
                      Path(args.labels_dir))
ref_lab = LatentDataset([l.strip() for l in open(args.ref_tokens) if l.strip()],
                        build_index(Path(args.ref_latents_dir)),
                        Path(args.ref_labels_dir))
ref_desc = np.stack([scene_desc(ref_lab[i]["outcomes"])
                     for i in range(len(ref_lab))])
mu, sd = ref_desc.mean(0), ref_desc.std(0) + 1e-6
ref_desc = (ref_desc - mu) / sd
ref_den = knn_mean_dist(ref_desc, ref_desc, KNN_Q + 1) * (KNN_Q + 1) / KNN_Q
t_e = np.quantile(ref_den, [1 / 3, 2 / 3])
d9 = np.quantile(ref_den, 0.9)
q_desc = np.stack([scene_desc(q_lab[i]["outcomes"])
                   for i in range(len(q_lab))])
q_desc = (q_desc - mu) / sd
q_den = knn_mean_dist(q_desc, ref_desc, KNN_Q)
tert_l = np.digitize(q_den, t_e)
rare10_l = q_den > d9
print("[eval] tertiles done", flush=True)

# align tertiles (LatentDataset token order) to CacheDataset item order
q_order = [p.stem for p in q_ds.items]
lab_pos = {q_lab.items[i][0]: i for i in range(len(q_lab))}
q_lab_idx = np.asarray([lab_pos[t] for t in q_order])
tert = tert_l[q_lab_idx]
rare10 = rare10_l[q_lab_idx]


def boot_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(m.mean()), float(np.percentile(m, 2.5)), \
        float(np.percentile(m, 97.5))


def gather_query_arrays():
    key, zp, a, yh, sub, b0, oc, logs = [], [], [], [], [], [], [], []
    from torch.utils.data import DataLoader
    from train_memory import collate_cache
    for b in DataLoader(q_ds, batch_size=64, shuffle=False, num_workers=4,
                        collate_fn=collate_cache):
        key.append(b["key_src"].numpy()); zp.append(b["z_pool"].numpy())
        a.append(b["a"].numpy()); yh.append(b["yhat"].numpy())
        sub.append(b["sub"].numpy()); b0.append(b["b0"].numpy())
        oc.append(b["outcomes"].numpy()); logs.extend(b["log_names"])
    return (np.concatenate(key), np.concatenate(zp), np.concatenate(a),
            np.concatenate(yh), np.concatenate(sub), np.concatenate(b0),
            np.concatenate(oc), np.asarray(logs))


key, zp, a, yh, sub, b0s, oc, logs = gather_query_arrays()
conf_scene = (oc[:, :, 0] > 0).any(1)
best = sub[..., 5].max(1)
b0_pick = b0s.argmax(1)
f_b0 = sub[..., 5][np.arange(len(sub)), b0_pick]
b0_wrong = f_b0 < best - 0.05


def strat(f, pick):
    m = dict(all=np.ones(len(f), bool), conflict=conf_scene,
             rare10=rare10, flip=pick != b0_pick, b0_wrong=b0_wrong,
             tert0=tert == 0, tert1=tert == 1, tert2=tert == 2)
    return {k: float(f[v].mean()) if v.any() else None
            for k, v in m.items()}, m


# ---- B0 arm first: must reproduce Step-3 navtest numbers ----------------------
out = {}
rep_b0, _ = strat(f_b0, b0_pick)
out["b0"] = rep_b0
print(f"[eval] b0: all={rep_b0['all']:.4f} conflict={rep_b0['conflict']:.4f} "
      f"rare10={rep_b0['rare10']:.4f} (expected ~.9354/.9337/.9025 on navtest)",
      flush=True)
print(f"[eval] b0_wrong scenes={int(b0_wrong.sum())}/{len(b0_wrong)}",
      flush=True)

# ---- learned arms --------------------------------------------------------------
arms = [a for a in args.arms.split(",") if a != "b0"]
dflat_lret = None
for arm in arms:
    if arm == "lret":
        f, pk, _, _, _, dl = eval_selection(bank, keynet, memenc, delta, q_ds,
                                            device, key_mode="lret")
    elif arm == "appearance":
        f, pk, _, _, _, dl = eval_selection(bank, keynet, memenc, delta, q_ds,
                                            device, key_mode="appearance")
    elif arm == "random":
        f, pk, _, _, _, dl = eval_selection(bank, keynet, memenc, delta, q_ds,
                                            device, key_mode="random")
    elif arm == "shuffle":
        f, pk, _, _, _, dl = eval_selection(bank, keynet, memenc, delta, q_ds,
                                            device, key_mode="lret",
                                            shuffle=True)
    elif arm.startswith("scale"):
        f, pk, _, _, _, dl = eval_selection(bank, keynet, memenc, delta, q_ds,
                                            device, key_mode="lret",
                                            scale=int(arm[5:]) / 100,
                                            seed=args.seed)
    else:
        continue
    rep, masks = strat(f, pk)
    m_all, lo_all, hi_all = boot_diff(f, f_b0)
    out[arm] = rep
    out[arm]["vs_b0_all"] = dict(mean=m_all, lo=lo_all, hi=hi_all)
    for s_name in ("conflict", "rare10", "b0_wrong", "flip"):
        mm = masks[s_name]
        m2, lo2, hi2 = boot_diff(f[mm], f_b0[mm])
        out[arm][f"vs_b0_{s_name}"] = dict(mean=m2, lo=lo2, hi=hi2)
    out[arm]["delta_mean"] = float(dl.mean())
    out[arm]["delta_big_frac"] = float((np.abs(dl) > 0.1).mean())
    out[arm]["dstd"] = float(
        (dl - dl.mean(1, keepdims=True)).std(1).mean())
    out[arm]["flip_rate"] = float((pk != b0_pick).mean())
    if arm == "lret":
        dflat_lret = dl
    if arm == "shuffle" and dflat_lret is not None:
        out[arm]["dgap_vs_lret"] = float(np.abs(dl - dflat_lret).mean())
    print(f"[eval] {arm}: all={rep['all']:.4f} ({m_all:+.4f} "
          f"[{lo_all:+.4f},{hi_all:+.4f}]) conflict={rep['conflict']:.4f} "
          f"rare10={rep['rare10']:.4f} flip={rep['flip']} "
          f"|d|>.1:{out[arm]['delta_big_frac']:.3f}", flush=True)

if args.out_json:
    Path(args.out_json).write_text(json.dumps(out, indent=1))
