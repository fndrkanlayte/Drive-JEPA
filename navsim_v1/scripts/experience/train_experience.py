#!/usr/bin/env python3
"""Stage 3: train the offline experience module (small MLP) on Phase-A data.

Per-candidate input is ``proposal_feature`` (256) + ``ego_status`` (11) from the
export npz; multi-task BCE targets are ``nc_unsafe = subscores[:,0] < 1`` and
``ttc_bad = subscores[:,3] < 1`` from the label npz. Scenes are split BY
log_name into memory / query_train / query_val (same ``split_logs_by_name``
used by oracle_knn_check, same default ratios/seed; the helper asserts no log
overlap).

Variants (``--variants``), all with identical encoder capacity:
  noexp         head on latent only
  random        head on [latent, mean/var of K random other-log memory labels]
  retrieval     head on [latent, softmax-weighted mean/var of top-K neighbour
                labels, mean neighbour latent] (cosine top-K, <=4 neighbours per
                memory scene, same-log excluded; memory re-encoded each epoch)
  retrieval_int retrieval + L_int aux head predicting the no-att main-vehicle
                descriptor (conflict BCE + itype CE + masked smooth-L1 on
                dt_enter/pet/min_dist). Training-side labels only.
  shuffle       eval-time control: the retrieval models evaluated with globally
                shuffled memory labels.

Each non-shuffle variant trains with ``--seeds`` (default 3 seeds); reported
predictions are the per-seed mean. Metrics on query_val: Brier / AUPRC /
top-1-risk with scene-level bootstrap 95% CIs, on all rows and the conflict
subset (has main vehicle & conflict & |dt_enter| < --dt_enter_thresh s).

If ``--navtest_labels_dir``/``--navtest_export_dir`` are given, per-candidate
risk (sigmoid of both logits) is predicted for those scenes with the memory
bank restricted to the navtrain-labelled rows (navtest is never in memory) and
written to ``<out_dir>/navtest_risk/<variant>.npz`` (tokens, risk (S,32,2),
mean over seeds).

Usage:
  python scripts/experience/train_experience.py \
      --labels_dir $EXP/experience/navtrain_labels_3k \
      --export_dir $EXP/experience/navtrain_export_3k \
      --navtest_labels_dir $EXP/experience/navtest_labels_smoke \
      --navtest_export_dir $EXP/experience/navtest_export_smoke \
      --out_dir $EXP/experience/train_experience_out
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))

from navsim.agents.drive_jepa_perception_based.experience.descriptors import (  # noqa: E402
    DESCRIPTOR_FIELD_INDEX as FI,
)
from navsim.agents.drive_jepa_perception_based.experience.knn import (  # noqa: E402
    eval_with_ci,
)
from navsim.agents.drive_jepa_perception_based.experience.model import (  # noqa: E402
    ExperienceModel,
    encode_rows,
)
from navsim.agents.drive_jepa_perception_based.experience.retrieval import (  # noqa: E402
    exp_dim_for,
    random_features,
    retrieval_features,
)
from navsim.agents.drive_jepa_perception_based.experience.records import (  # noqa: E402
    fmt_ci,
    load_npz,
    save_npz,
    split_logs_by_name,
)

SUB_NC, SUB_TTC = 0, 3
TARGET_NAMES = ["nc_unsafe", "ttc_bad"]
VARIANTS = ["noexp", "random", "retrieval", "retrieval_int", "shuffle"]
BASE = {"shuffle": "retrieval"}  # eval-only variants -> trained variant


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--export_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--navtest_labels_dir", default=None)
    p.add_argument("--navtest_export_dir", default=None)
    p.add_argument("--ratios", type=float, nargs=3, default=[0.6, 0.2, 0.2])
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--max_per_scene", type=int, default=4)
    p.add_argument("--aux_weight", type=float, default=0.5)
    p.add_argument("--num_boot", type=int, default=1000)
    p.add_argument("--dt_enter_thresh", type=float, default=2.0)
    p.add_argument("--device",
                   default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    return p.parse_args()


def load_rows(labels_dir: Path, export_dir: Path,
              dt_enter_thresh: float = 2.0) -> Dict[str, np.ndarray]:
    """Join label + export npz into per-candidate row arrays."""
    X, Y, AUX, AUX_MASK = [], [], [], []
    scene_id, log_name, in_subset = [], [], []
    tokens, logs_sc = [], []
    sid = 0
    for lp in sorted(labels_dir.glob("*.npz")):
        ep = export_dir / lp.name
        if not ep.is_file():
            continue
        lab, exp = load_npz(lp), load_npz(ep)
        K = lab["subscores"].shape[0]
        pf = np.asarray(exp["proposal_feature"], dtype=np.float32)  # (K,256)
        es = np.asarray(exp["ego_status"], dtype=np.float32).reshape(-1)  # (11,)
        x = np.concatenate([pf, np.repeat(es[None], K, axis=0)], axis=1)
        y = np.stack([
            (lab["subscores"][:, SUB_NC] < 1.0),
            (lab["subscores"][:, SUB_TTC] < 1.0),
        ], axis=1).astype(np.float32)
        # aux targets from the no-attribution main vehicle (training-side only)
        md = lab.get("main_desc_noatt")
        if md is None:
            raise ValueError(
                f"{lp.name} lacks main_desc_noatt -- re-run label_candidates"
            )
        aux = np.stack([
            md[:, FI["conflict"]],
            md[:, FI["itype"]],
            md[:, FI["dt_enter"]],
            md[:, FI["pet"]],
            md[:, FI["min_dist"]],
        ], axis=1).astype(np.float32)  # (K, 5)
        aux_mask = np.isfinite(aux)
        aux[~aux_mask] = 0.0
        has_main = np.isfinite(md[:, FI["conflict"]])
        conf = md[:, FI["conflict"]] == 1.0
        dte = np.abs(md[:, FI["dt_enter"]])
        subset = has_main & conf & (dte < dt_enter_thresh)
        X.append(x); Y.append(y); AUX.append(aux); AUX_MASK.append(aux_mask)
        scene_id.append(np.full(K, sid)); log_name += [str(lab["log_name"].item())] * K
        in_subset.append(subset)
        tokens.append(str(lab["token"].item()))
        logs_sc.append(str(lab["log_name"].item()))
        sid += 1
    return dict(
        x=np.concatenate(X), y=np.concatenate(Y),
        aux=np.concatenate(AUX), aux_mask=np.concatenate(AUX_MASK),
        scene_id=np.concatenate(scene_id), log=np.asarray(log_name),
        subset=np.concatenate(in_subset),
        tokens=np.asarray(tokens), scene_log=np.asarray(logs_sc),
        n_scenes=sid,
    )


def build_exp(variant: str, q_lat, q_scene, q_log, mem, args, rng,
              shuffle_labels: bool = False) -> np.ndarray:
    """Experience-feature block for a set of query rows."""
    if variant == "noexp":
        return np.zeros((len(q_lat), 0), dtype=np.float32)
    if variant == "random":
        return random_features(len(q_lat), q_log, mem["y"], mem["log"],
                               topk=args.topk, rng=rng)
    return retrieval_features(
        q_lat, q_scene, q_log, mem["latent"], mem["y"], mem["scene_id"],
        mem["log"], topk=args.topk, max_per_scene=args.max_per_scene,
        rng=rng, shuffle_labels=shuffle_labels,
    )


def aux_loss_fn(aux_out, aux, aux_mask, torch):
    """L_int: conflict BCE + itype CE + masked smooth-L1 on dt_enter/pet/min_dist."""
    import torch.nn.functional as TF
    loss = torch.zeros((), device=aux_out.device)
    m = aux_mask[:, 0]  # conflict/itype valid iff the conflict field exists
    if m.any():
        loss = loss + TF.binary_cross_entropy_with_logits(
            aux_out[m, 0], aux[m, 0], reduction="mean")
        loss = loss + TF.cross_entropy(
            aux_out[m, 1:5], aux[m, 1].long().clamp(0, 3), reduction="mean")
    m3 = aux_mask[:, 2:5]
    if m3.any():
        loss = loss + (TF.smooth_l1_loss(
            aux_out[:, 5:8], aux[:, 2:5], reduction="none") * m3).sum() / m3.sum()
    return loss


def predict(model, x_t, exp_np, device, batch=8192):
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x_t), batch):
            e = torch.from_numpy(exp_np[i:i + batch]).to(device)
            logits, _, _ = model(x_t[i:i + batch].to(device), e)
            outs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outs)


def train_one(variant: str, seed: int, rows: Dict[str, np.ndarray],
              mem_rows: Dict[str, np.ndarray], args):
    """Train one (variant, seed). Returns (model, val_preds, val_preds_shuffle)."""
    import torch
    import torch.nn.functional as TF

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = args.device

    tr = rows["group"] == "query_train"
    va = rows["group"] == "query_val"
    x_all = torch.from_numpy(rows["x"]).float()
    y_all = torch.from_numpy(rows["y"]).float()
    aux_all = torch.from_numpy(rows["aux"]).float()
    auxm_all = torch.from_numpy(rows["aux_mask"]).bool()

    latent = 64
    use_int = variant == "retrieval_int"
    model = ExperienceModel(x_all.shape[1], latent + exp_dim_for(variant, latent),
                            use_int=use_int).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    mem_x = torch.from_numpy(mem_rows["x"]).float()
    mem = {"y": mem_rows["y"], "scene_id": mem_rows["scene_id"],
           "log": mem_rows["log"], "latent": None}
    tr_idx = np.flatnonzero(tr)
    va_idx = np.flatnonzero(va)
    q_idx = np.concatenate([tr_idx, va_idx])  # experience feats only for queries

    for epoch in range(args.epochs):
        # re-encode memory + queries with current weights; exp feats are
        # detached inputs for this epoch (encoder still learns via latent)
        mem["latent"] = encode_rows(model, mem_x, device=device)
        q_lat = encode_rows(model, x_all[q_idx], device=device)
        exp = build_exp(variant, q_lat, rows["scene_id"][q_idx],
                        rows["log"][q_idx], mem, args, rng)
        exp_tr = torch.from_numpy(exp[: len(tr_idx)])
        model.train()
        perm = rng.permutation(len(tr_idx))
        for i0 in range(0, len(tr_idx), args.batch_size):
            pos = perm[i0:i0 + args.batch_size]
            bi = tr_idx[pos]
            xb = x_all[bi].to(device)
            eb = exp_tr[pos].to(device)
            logits, z, aux_out = model(xb, eb)
            loss = TF.binary_cross_entropy_with_logits(logits, y_all[bi].to(device))
            if use_int and aux_out is not None:
                loss = loss + args.aux_weight * aux_loss_fn(
                    aux_out, aux_all[bi].to(device), auxm_all[bi].to(device), torch)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # final predictions on query rows with the trained encoder
    mem["latent"] = encode_rows(model, mem_x, device=device)
    q_lat = encode_rows(model, x_all[q_idx], device=device)
    exp = build_exp(variant, q_lat, rows["scene_id"][q_idx], rows["log"][q_idx],
                    mem, args, np.random.default_rng(seed + 777))
    pred = predict(model, x_all[q_idx], exp, device)
    out = {"model": model, "val": pred[len(tr_idx):]}
    if variant == "retrieval":
        exp_sh = build_exp(variant, q_lat, rows["scene_id"][q_idx],
                           rows["log"][q_idx], mem, args,
                           np.random.default_rng(seed + 778), shuffle_labels=True)
        out["val_shuffle"] = predict(model, x_all[q_idx], exp_sh, device)[len(tr_idx):]
    return out


def predict_navtest(model, variant: str, nt: Dict[str, np.ndarray],
                    mem_rows: Dict[str, np.ndarray], args,
                    seed: int) -> np.ndarray:
    """Risk (N,2) for navtest rows; memory bank = navtrain labelled rows."""
    import torch
    device = args.device
    rng = np.random.default_rng(seed + 9000)
    mem_x = torch.from_numpy(mem_rows["x"]).float()
    nt_x = torch.from_numpy(nt["x"]).float()
    mem = {"y": mem_rows["y"], "scene_id": mem_rows["scene_id"],
           "log": mem_rows["log"],
           "latent": encode_rows(model, mem_x, device=device)}
    q_lat = encode_rows(model, nt_x, device=device)
    exp = build_exp(variant, q_lat, nt["scene_id"], nt["log"], mem, args, rng,
                    shuffle_labels=(variant == "shuffle"))
    return predict(model, nt_x, exp, device)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = load_rows(Path(args.labels_dir), Path(args.export_dir),
                     args.dt_enter_thresh)
    n_scenes = int(rows["n_scenes"])
    print(f"[exp] {n_scenes} scenes, {len(rows['y'])} candidates")

    logs = sorted(set(rows["log"].tolist()))
    groups = split_logs_by_name(logs, ratios=args.ratios, seed=args.split_seed)
    group_of = {l: g for g, ls in groups.items() for l in ls}
    rows["group"] = np.array([group_of[l] for l in rows["log"]])
    print(f"[exp] logs: memory={len(groups['memory'])} "
          f"query_train={len(groups['query_train'])} "
          f"query_val={len(groups['query_val'])}")
    for ti, tname in enumerate(TARGET_NAMES):
        for sp in ("memory", "query_train", "query_val"):
            m = rows["group"] == sp
            print(f"[exp] pos-rate {tname} {sp}: {rows['y'][m, ti].mean():.4f}")

    mem_mask = rows["group"] == "memory"
    mem_rows = {k: rows[k][mem_mask] for k in ("x", "y", "scene_id", "log")}

    val_mask = rows["group"] == "query_val"
    val_y = rows["y"][val_mask]
    val_sid = rows["scene_id"][val_mask]
    val_sub = rows["subset"][val_mask]

    # ---- train -------------------------------------------------------------
    # variant -> list of per-seed {"model", "val", "val_shuffle"?}
    runs: Dict[str, List[dict]] = {}
    for variant in args.variants:
        real = BASE.get(variant, variant)
        if variant == "shuffle":
            continue  # eval-only; reuses retrieval models
        runs[variant] = []
        for seed in args.seeds:
            print(f"[exp] training {variant} seed={seed}", flush=True)
            t1 = time.time()
            runs[variant].append(train_one(real, seed, rows, mem_rows, args))
            print(f"[exp]   done in {time.time()-t1:.0f}s", flush=True)

    # ---- evaluate on query_val ---------------------------------------------
    table_rows: List[List[str]] = []
    results: Dict[str, dict] = {}
    for variant in args.variants:
        real = BASE.get(variant, variant)
        if real not in runs:
            continue
        key = "val_shuffle" if variant == "shuffle" else "val"
        per_seed = [r[key] for r in runs[real] if key in r]
        if not per_seed:
            continue
        pred_va = np.mean(per_seed, axis=0)
        for ti, tname in enumerate(TARGET_NAMES):
            for sub_name, sub_mask in (("all", None), ("conflict", val_sub)):
                sel = np.ones(len(val_y), bool) if sub_mask is None else sub_mask
                if sel.sum() == 0:
                    res = {"brier": (np.nan, np.nan, np.nan),
                           "auprc": (np.nan, np.nan, np.nan),
                           "top1_risk": (np.nan, np.nan, np.nan)}
                else:
                    res = eval_with_ci(pred_va[sel, ti], val_y[sel, ti],
                                       val_sid[sel], args.num_boot,
                                       args.split_seed)
                results[f"{variant}|{tname}|{sub_name}"] = res
                table_rows.append([variant, tname, sub_name,
                                   fmt_ci(*res["brier"]), fmt_ci(*res["auprc"]),
                                   fmt_ci(*res["top1_risk"])])

    # save per-scene query_val risk for rerank lambda tuning
    qva_risk_dir = out_dir / "query_val_risk"
    val_scene_ids = np.unique(val_sid)
    val_scene_tokens = rows["tokens"][val_scene_ids]
    for variant in args.variants:
        real = BASE.get(variant, variant)
        if real not in runs:
            continue
        key = "val_shuffle" if variant == "shuffle" else "val"
        per_seed = np.stack([r[key] for r in runs[real] if key in r])
        save_npz(
            qva_risk_dir / f"{variant}.npz",
            tokens=val_scene_tokens,
            risk=per_seed.mean(0).reshape(len(val_scene_ids), -1, 2).astype(np.float32),
        )

    md = ["# Experience-module results (Stage 3)", ""]
    md.append(
        f"Scenes: {n_scenes} | logs: memory={len(groups['memory'])}, "
        f"query_train={len(groups['query_train'])}, "
        f"query_val={len(groups['query_val'])} | split_seed={args.split_seed} "
        f"| seeds={args.seeds} | topk={args.topk} "
        f"max_per_scene={args.max_per_scene} | epochs={args.epochs}"
    )
    md.append("")
    md.append("| variant | target | subset | Brier | AUPRC | top1-risk |")
    md.append("|---|---|---|---|---|---|")
    for r in table_rows:
        md.append("| " + " | ".join(r) + " |")
    (out_dir / "train_experience_results.md").write_text("\n".join(md) + "\n")
    with open(out_dir / "train_experience_results.json", "w") as f:
        json.dump({k: {m: list(v) for m, v in res.items()}
                   for k, res in results.items()}, f, indent=2)

    # ---- navtest risk export (memory = navtrain rows only) ------------------
    if args.navtest_labels_dir and args.navtest_export_dir:
        print("[exp] predicting navtest risks", flush=True)
        nt = load_rows(Path(args.navtest_labels_dir), Path(args.navtest_export_dir),
                       args.dt_enter_thresh)
        risk_dir = out_dir / "navtest_risk"
        for variant, seed_runs in runs.items():
            per_seed = np.stack([
                predict_navtest(r["model"], variant, nt, mem_rows, args, s)
                for s, r in zip(args.seeds, seed_runs)
            ])
            save_npz(
                risk_dir / f"{variant}.npz",
                tokens=nt["tokens"],
                log_names=nt["scene_log"],
                risk=per_seed.mean(0).astype(np.float32),      # (S,K,2)
                risk_per_seed=per_seed.astype(np.float32),     # (nseed,S,K,2)
            )
    print(f"[exp] done in {time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
