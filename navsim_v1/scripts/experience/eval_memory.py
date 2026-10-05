#!/usr/bin/env python
"""Step 4 eval: memory-augmented selection on navtest (or val).

Arms:
    b0          native argmax on pdm_score
    lret        learned g retrieval + MemoryDelta (main method)
    appearance  appearance-cosine retrieval + MemoryDelta
    random      random scenes + MemoryDelta
    shuffle     lret retrieval, memory tokens shuffled
    scale{25,50}  lret with 25/50% of bank scenes (subsampled scenes)
    (nolc arm = a separate run dir trained with --lam_c 0)

Scores per scene -> argmax -> chosen candidate's true final, stratified by
all / conflict / rare10 / flip / tert0-2. Paired scene bootstrap 95% CI vs B0.

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
from train_memory import (CacheDataset, build_bank, collate_cache,  # noqa: E402
                          eval_selection)
from navsim.agents.drive_jepa_perception_based.experience.ewm_memory import (  # noqa: E402
    KeyNet, MemoryDelta, MemTokenEnc, score_with_delta, make_memory_tokens)

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
p.add_argument("--arms", default="b0,lret,appearance,random,shuffle,scale25,scale50")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out_json", default=None)
args = p.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(Path(args.run) / "model_best.pt", map_location="cpu",
                  weights_only=False)
keynet = KeyNet().to(device); keynet.load_state_dict(ckpt["keynet"])
memenc = MemTokenEnc().to(device); memenc.load_state_dict(ckpt["memenc"])
delta = MemoryDelta().to(device); delta.load_state_dict(ckpt["delta"])
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

bank_tokens = [f.stem for f in Path(args.bank_cache).glob("*/*.npz")]
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
tert = np.digitize(q_den, t_e)
rare10 = q_den > d9
print("[eval] tertiles done", flush=True)

# query order alignment: CacheDataset items sorted by path (log/token);
# LatentDataset items sorted by token. Align by token index.
q_order = [p.stem for p in q_ds.items]
lab_pos = {q_lab.items[i][0]: i for i in range(len(q_lab))}
q_lab_idx = np.asarray([lab_pos[t] for t in q_order])
tert = tert[q_lab_idx]
rare10 = rare10[q_lab_idx]

# ---- run arms -----------------------------------------------------------------
def boot_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(m.mean()), float(np.percentile(m, 2.5)), \
        float(np.percentile(m, 97.5))


def strat_report(chosen_final, sub, logs, tert, rare10):
    """per-scene metrics in 6 subsets; chosen_final (S,) true final of argmax."""
    conf_scene = (sub[:, :, 0] < 1).any(1)
    b0_pick = None  # filled by caller via flip mask later
    rep = {}
    for name, m in dict(all=np.ones(len(chosen_final), bool),
                        conflict=conf_scene,
                        rare10=rare10,
                        tert0=tert == 0, tert1=tert == 1,
                        tert2=tert == 2).items():
        rep[name] = float(chosen_final[m].mean()) if m.any() else None
    return rep


out = {}
f0, sub, b0s, logs = eval_selection(bank, keynet, memenc, delta, q_ds,
                                    device, key_mode="random")
# b0 arm: chosen by pdm argmax
f_b0 = sub[np.arange(len(sub)), b0s.argmax(1)]
flip_mask = None
results = {}

arms = args.arms.split(",")
for arm in arms:
    if arm == "b0":
        f = f_b0
    elif arm == "lret":
        f, _, _, _ = eval_selection(bank, keynet, memenc, delta, q_ds, device,
                                    key_mode="lret")
    elif arm == "appearance":
        f, _, _, _ = eval_selection(bank, keynet, memenc, delta, q_ds, device,
                                    key_mode="appearance")
    elif arm == "random":
        f = f0
    elif arm == "shuffle":
        f, _, _, _ = eval_selection(bank, keynet, memenc, delta, q_ds, device,
                                    key_mode="lret", shuffle=True)
    elif arm.startswith("scale"):
        f, _, _, _ = eval_selection(bank, keynet, memenc, delta, q_ds, device,
                                    key_mode="lret",
                                    scale=int(arm[5:]) / 100, seed=args.seed)
    else:
        continue
    results[arm] = f
    rep = strat_report(f, sub, logs, tert, rare10)
    out[arm] = rep
    print(f"[eval] {arm}: all={rep['all']:.4f} conflict={rep['conflict']:.4f} "
          f"rare10={rep['rare10']:.4f}", flush=True)

# flip: scenes where b0 picked wrong (>0.05 below best)
best = sub[..., 5].max(1)
f_b0_final = f_b0
flip = sub[np.arange(len(sub)), b0s.argmax(1)] < best - 0.05
for arm, f in results.items():
    out[arm]["flip"] = float(f[flip].mean()) if flip.any() else None
    m, lo, hi = boot_diff(f, f_b0_final)
    out[arm]["vs_b0"] = dict(mean=m, lo=lo, hi=hi)
    for sub_name, msk in dict(all=np.ones(len(f), bool),
                              rare10=rare10, flip=flip,
                              conflict=(sub[:, :, 0] < 1).any(1)).items():
        m2, lo2, hi2 = boot_diff(f[msk], f_b0_final[msk])
        out[arm][f"vs_b0_{sub_name}"] = dict(mean=m2, lo=lo2, hi=hi2)
    print(f"[eval] {arm} vs b0: all {m:+.4f} [{lo:+.4f},{hi:+.4f}] "
          f"flip={out[arm]['flip']}", flush=True)

print(f"[eval] flip scenes={int(flip.sum())}/{len(flip)}", flush=True)
if args.out_json:
    Path(args.out_json).write_text(json.dumps(out, indent=1))
