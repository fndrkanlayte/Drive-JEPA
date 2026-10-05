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
  noexp_int     noexp + the L_int aux head (matched-capacity control for
                retrieval_int -- is the gain from retrieval or the aux loss?)
  pred_desc_retrieval     deployable retrieval: a descriptor head trained on
                query_train predicts the no-att main-vehicle TIMING descriptor
                from the latent; top-K retrieval runs in descriptor space
                (query d_hat vs memory TRUE desc); head on [latent, weighted
                neighbour label mean/var, mean(d_hat - d) on continuous cols,
                mean neighbour similarity]
  pred_desc_retrieval_pp  same but the memory side also uses its own
                PREDICTED descriptors (pred-vs-pred)

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
    DESCRIPTOR_FIELDS,
    TIMING_FIELDS,
    descriptor_feature_names,
    descriptor_feature_vector,
)
from navsim.agents.drive_jepa_perception_based.experience.knn import (  # noqa: E402
    eval_with_ci,
    standardize_apply,
    standardize_fit,
)
from navsim.agents.drive_jepa_perception_based.experience.model import (  # noqa: E402
    ExperienceModel,
    desc_rows,
    encode_rows,
)
from navsim.agents.drive_jepa_perception_based.experience.retrieval import (  # noqa: E402
    desc_retrieval_features,
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

SUB_NC, SUB_TTC, SUB_FINAL = 0, 3, 5
TARGET_NAMES = ["nc_unsafe", "ttc_bad"]
VARIANTS = ["noexp", "noexp_int", "random", "retrieval", "retrieval_int",
            "shuffle", "pred_desc_retrieval", "pred_desc_retrieval_pp"]
# continuous descriptor cols used for the mean(d_hat - d_i) feature
DESC_CONT_NAMES = ["dt_enter", "pet", "rel_x", "rel_y", "rel_heading",
                   "speed", "ego_speed"]
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
    p.add_argument("--allowed_logs_file", default=None,
                   help="json list of log names; restrict all rows to these "
                        "logs BEFORE the memory/query split (Q7 held-out "
                        "protocol: train on half A only)")
    p.add_argument("--drop_c", action="store_true",
                   help="Q10 held-out long-tail: drop every C candidate "
                        "(noatt main desc: conflict=1, |dt_enter|<2s, "
                        "itype in {CROSSING,ONCOMING}) from ALL rows before "
                        "the memory/query split; scenes are kept")
    p.add_argument("--task", choices=["risk", "residual"], default="risk",
                   help="risk: BCE on nc_unsafe/ttc_bad. residual: Huber "
                        "regression on y_res = labelled final - pdm_score "
                        "(single output; neighbour 'labels' become y_res).")
    p.add_argument("--save_models", action="store_true",
                   help="save per-variant+seed state_dicts + meta to "
                        "<out_dir>/models/ (for post-hoc eval, e.g. "
                        "memory-size ablation).")
    p.add_argument("--device",
                   default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    return p.parse_args()


def c_mask(rows: Dict[str, np.ndarray]) -> np.ndarray:
    """Boolean (N,) -- candidate's noatt main desc is a C interaction:
    conflict=1 AND |dt_enter|<2s AND itype in {CROSSING, ONCOMING}.
    Reads aux columns [conflict, itype, dt_enter] with aux_mask.
    """
    aux, m = rows["aux"], rows["aux_mask"]
    return (
        m[:, 0] & (aux[:, 0] > 0.5)
        & m[:, 1] & np.isin(aux[:, 1].astype(np.int64), [2, 3])
        & m[:, 2] & (np.abs(aux[:, 2]) < 2.0)
    )


def load_rows(labels_dir: Path, export_dir: Path,
              dt_enter_thresh: float = 2.0) -> Dict[str, np.ndarray]:
    """Join label + export npz into per-candidate row arrays."""
    X, Y, AUX, AUX_MASK, DESC, PDM, FIN = [], [], [], [], [], [], []
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
        ego_speed = float(lab["ego_speed"].item()) if "ego_speed" in lab else np.nan
        desc = np.stack([
            descriptor_feature_vector(
                {f: md[k, i] for i, f in enumerate(DESCRIPTOR_FIELDS)},
                ego_speed=ego_speed, fields=TIMING_FIELDS)
            for k in range(K)
        ])  # (K, D_desc) raw, NaNs preserved
        X.append(x); Y.append(y); AUX.append(aux); AUX_MASK.append(aux_mask)
        DESC.append(desc)
        PDM.append(np.asarray(exp["pdm_score"], dtype=np.float32))
        FIN.append(lab["subscores"][:, 5].astype(np.float32))
        scene_id.append(np.full(K, sid)); log_name += [str(lab["log_name"].item())] * K
        in_subset.append(subset)
        tokens.append(str(lab["token"].item()))
        logs_sc.append(str(lab["log_name"].item()))
        sid += 1
    return dict(
        x=np.concatenate(X), y=np.concatenate(Y),
        aux=np.concatenate(AUX), aux_mask=np.concatenate(AUX_MASK),
        desc=np.concatenate(DESC),
        pdm=np.concatenate(PDM), final=np.concatenate(FIN),
        scene_id=np.concatenate(scene_id), log=np.asarray(log_name),
        subset=np.concatenate(in_subset),
        tokens=np.asarray(tokens), scene_log=np.asarray(logs_sc),
        n_scenes=sid,
    )


def build_exp(variant: str, q_lat, q_scene, q_log, mem, args, rng,
              shuffle_labels: bool = False) -> np.ndarray:
    """Experience-feature block for a set of query rows."""
    if variant in ("noexp", "noexp_int"):
        return np.zeros((len(q_lat), 0), dtype=np.float32)
    if variant == "random":
        return random_features(len(q_lat), q_log, mem["y"], mem["log"],
                               topk=args.topk, rng=rng)
    return retrieval_features(
        q_lat, q_scene, q_log, mem["latent"], mem["y"], mem["scene_id"],
        mem["log"], topk=args.topk, max_per_scene=args.max_per_scene,
        rng=rng, shuffle_labels=shuffle_labels, device=args.device,
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


def predict(model, x_t, exp_np, device, batch=8192, sigmoid=True):
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x_t), batch):
            e = torch.from_numpy(exp_np[i:i + batch]).to(device)
            logits, _, _, _ = model(x_t[i:i + batch].to(device), e)
            outs.append(
                (torch.sigmoid(logits) if sigmoid else logits).cpu().numpy())
    return np.concatenate(outs)


def conf_rows(model, x_t, device, conf_col: int = 0, zm_c: float = 0.0,
              zs_c: float = 1.0, batch: int = 8192) -> Optional[np.ndarray]:
    """Per-candidate conflict score for gating (Q2).

    int_head -> sigmoid(conflict logit); desc_head -> predicted desc_z in the
    conflict col mapped back to raw units (approx probability). None if the
    model has neither head.
    """
    import torch
    if model.int_head is None and model.desc_head is None:
        return None
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x_t), batch):
            z = model.encoder(x_t[i:i + batch].to(device))
            if model.int_head is not None:
                outs.append(torch.sigmoid(
                    model.int_head(z)[:, 0]).cpu().numpy())
            else:
                d = model.desc_head(z)[:, conf_col].cpu().numpy()
                outs.append(d * zs_c + zm_c)
    return np.concatenate(outs)


def train_pred_desc(variant: str, seed: int, rows: Dict[str, np.ndarray],
                    mem_rows: Dict[str, np.ndarray], args):
    """Two-phase pred_desc training (see module docstring).

    Phase 1: encoder + DescHead predict the standardized TRUE no-att timing
             descriptor on query_train rows (masked MSE).
    Phase 2: d_hat for queries + memory desc (TRUE or predicted for the _pp
             variant) -> static descriptor-space retrieval features.
    Phase 3: encoder frozen; risk head trained on [latent, exp] with BCE.
    """
    import torch
    import torch.nn.functional as TF

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = args.device
    tr = rows["group"] == "query_train"
    tr_idx = np.flatnonzero(tr)
    va_idx = np.flatnonzero(rows["group"] == "query_val")
    q_idx = np.concatenate([tr_idx, va_idx])

    x_all = torch.from_numpy(rows["x"]).float()
    y_all = torch.from_numpy(rows["y"]).float()
    desc_z = torch.from_numpy(rows["desc_z"]).float()
    desc_valid = torch.from_numpy(rows["desc_valid"].astype(np.float32))
    D_desc = rows["desc_z"].shape[1]
    latent = 64
    cont_cols = rows["cont_cols"]
    exp_dim = exp_dim_for(variant, latent, n_cont=len(cont_cols))
    n_targets = rows["y"].shape[1]
    model = ExperienceModel(x_all.shape[1], latent + exp_dim,
                            desc_dim=D_desc, n_targets=n_targets).to(device)
    residual = args.task == "residual"

    # phase 1: descriptor prediction
    opt = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.desc_head.parameters()),
        lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(len(tr_idx))
        for i0 in range(0, len(tr_idx), args.batch_size):
            bi = tr_idx[perm[i0:i0 + args.batch_size]]
            z = model.encoder(x_all[bi].to(device))
            d_hat = model.desc_head(z)
            m = desc_valid[bi].to(device)
            loss = (TF.mse_loss(d_hat, desc_z[bi].to(device), reduction="none")
                    * m).sum() / m.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # phase 2: static descriptor-space retrieval features
    use_pred_mem = variant == "pred_desc_retrieval_pp"
    d_hat_q = desc_rows(model, x_all[q_idx], device=device)
    if use_pred_mem:
        mem_desc = desc_rows(model, torch.from_numpy(mem_rows["x"]).float(),
                             device=device)
        mem_valid = np.ones_like(mem_desc, dtype=bool)
    else:
        mem_desc = mem_rows["desc_z"]
        mem_valid = mem_rows["desc_valid"]
    exp = desc_retrieval_features(
        d_hat_q, rows["scene_id"][q_idx], rows["log"][q_idx],
        mem_desc, mem_rows["y"], mem_rows["scene_id"], mem_rows["log"],
        cont_cols, mem_valid, topk=args.topk,
        max_per_scene=args.max_per_scene, device=args.device)

    # phase 3: freeze encoder, train risk head on [latent, exp]
    for p_ in model.encoder.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.Adam(model.head.parameters(), lr=args.lr)
    exp_tr = torch.from_numpy(exp[: len(tr_idx)]).float()
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(len(tr_idx))
        for i0 in range(0, len(tr_idx), args.batch_size):
            pos = perm[i0:i0 + args.batch_size]
            bi = tr_idx[pos]
            logits, _, _, _ = model(x_all[bi].to(device), exp_tr[pos].to(device))
            if residual:
                loss = TF.huber_loss(logits[:, 0], y_all[bi, 0].to(device))
            else:
                loss = TF.binary_cross_entropy_with_logits(
                    logits, y_all[bi].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

    pred = predict(model, x_all[q_idx], exp, device, sigmoid=not residual)
    return {"model": model, "val": pred[len(tr_idx):],
            "use_pred_mem": use_pred_mem}


def train_one(variant: str, seed: int, rows: Dict[str, np.ndarray],
              mem_rows: Dict[str, np.ndarray], args):
    """Train one (variant, seed). Returns (model, val_preds, val_preds_shuffle)."""
    import torch
    import torch.nn.functional as TF

    if variant.startswith("pred_desc"):
        return train_pred_desc(variant, seed, rows, mem_rows, args)

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
    use_int = variant in ("retrieval_int", "noexp_int")
    n_targets = rows["y"].shape[1]
    residual = args.task == "residual"
    model = ExperienceModel(x_all.shape[1], latent + exp_dim_for(variant, latent),
                            use_int=use_int, n_targets=n_targets).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    mem_x = torch.from_numpy(mem_rows["x"]).float()
    mem = {"y": mem_rows["y"], "scene_id": mem_rows["scene_id"],
           "log": mem_rows["log"], "latent": None}
    tr_idx = np.flatnonzero(tr)
    va_idx = np.flatnonzero(va)
    q_idx = np.concatenate([tr_idx, va_idx])  # experience feats only for queries
    needs_latent = exp_dim_for(variant, 64) > 4  # retrieval variants only

    # noexp/random exp feats don't depend on the encoder -- build once
    exp_static = None
    if not needs_latent:
        exp_static = build_exp(variant, np.zeros((len(q_idx), 1), np.float32),
                               rows["scene_id"][q_idx], rows["log"][q_idx],
                               mem, args, rng)

    for epoch in range(args.epochs):
        # re-encode memory + queries with current weights; exp feats are
        # detached inputs for this epoch (encoder still learns via latent)
        if needs_latent:
            mem["latent"] = encode_rows(model, mem_x, device=device)
            q_lat = encode_rows(model, x_all[q_idx], device=device)
            exp = build_exp(variant, q_lat, rows["scene_id"][q_idx],
                            rows["log"][q_idx], mem, args, rng)
        else:
            exp = exp_static
        exp_tr = torch.from_numpy(exp[: len(tr_idx)])
        model.train()
        perm = rng.permutation(len(tr_idx))
        for i0 in range(0, len(tr_idx), args.batch_size):
            pos = perm[i0:i0 + args.batch_size]
            bi = tr_idx[pos]
            xb = x_all[bi].to(device)
            eb = exp_tr[pos].to(device)
            logits, z, aux_out, _ = model(xb, eb)
            if residual:
                loss = TF.huber_loss(logits[:, 0], y_all[bi, 0].to(device))
            else:
                loss = TF.binary_cross_entropy_with_logits(
                    logits, y_all[bi].to(device))
            if use_int and aux_out is not None:
                loss = loss + args.aux_weight * aux_loss_fn(
                    aux_out, aux_all[bi].to(device), auxm_all[bi].to(device), torch)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # final predictions on query rows with the trained encoder
    if needs_latent:
        mem["latent"] = encode_rows(model, mem_x, device=device)
        q_lat = encode_rows(model, x_all[q_idx], device=device)
        exp = build_exp(variant, q_lat, rows["scene_id"][q_idx],
                        rows["log"][q_idx], mem, args,
                        np.random.default_rng(seed + 777))
    else:
        exp = exp_static
    pred = predict(model, x_all[q_idx], exp, device, sigmoid=not residual)
    out = {"model": model, "val": pred[len(tr_idx):]}
    if variant == "retrieval":
        exp_sh = build_exp(variant, q_lat, rows["scene_id"][q_idx],
                           rows["log"][q_idx], mem, args,
                           np.random.default_rng(seed + 778), shuffle_labels=True)
        out["val_shuffle"] = predict(
            model, x_all[q_idx], exp_sh, device,
            sigmoid=not residual)[len(tr_idx):]
    return out


def predict_navtest(model, variant: str, nt: Dict[str, np.ndarray],
                    mem_rows: Dict[str, np.ndarray], args,
                    seed: int) -> np.ndarray:
    """Risk/residual (N,C) for navtest rows; memory bank = navtrain rows."""
    import torch
    device = args.device
    rng = np.random.default_rng(seed + 9000)
    residual = args.task == "residual"
    mem_x = torch.from_numpy(mem_rows["x"]).float()
    nt_x = torch.from_numpy(nt["x"]).float()
    if variant.startswith("pred_desc"):
        d_hat_nt = desc_rows(model, nt_x, device=device)
        if variant == "pred_desc_retrieval_pp":
            mem_desc = desc_rows(model, mem_x, device=device)
            mem_valid = np.ones_like(mem_desc, dtype=bool)
        else:
            mem_desc = mem_rows["desc_z"]
            mem_valid = mem_rows["desc_valid"]
        exp = desc_retrieval_features(
            d_hat_nt, nt["scene_id"], nt["log"], mem_desc, mem_rows["y"],
            mem_rows["scene_id"], mem_rows["log"], nt["cont_cols"],
            mem_valid, topk=args.topk, max_per_scene=args.max_per_scene,
            device=device)
        return predict(model, nt_x, exp, device, sigmoid=not residual)
    mem = {"y": mem_rows["y"], "scene_id": mem_rows["scene_id"],
           "log": mem_rows["log"],
           "latent": encode_rows(model, mem_x, device=device)}
    q_lat = encode_rows(model, nt_x, device=device)
    exp = build_exp(variant, q_lat, nt["scene_id"], nt["log"], mem, args, rng,
                    shuffle_labels=(variant == "shuffle"))
    return predict(model, nt_x, exp, device, sigmoid=not residual)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = load_rows(Path(args.labels_dir), Path(args.export_dir),
                     args.dt_enter_thresh)
    n_scenes = int(rows["n_scenes"])
    if args.allowed_logs_file:
        import json as _json
        allowed = set(_json.load(open(args.allowed_logs_file)))
        m = np.array([l in allowed for l in rows["log"]])
        rows = {k: (v[m] if isinstance(v, np.ndarray)
                    and v.ndim >= 1 and v.shape[0] == m.shape[0]
                    else v) for k, v in rows.items()}
        n_scenes = int(len(np.unique(rows["scene_id"])))
        print(f"[exp] restricted to {len(allowed)} logs from "
              f"{args.allowed_logs_file}")
    if args.drop_c:
        c = c_mask(rows)
        rows = {k: (v[~c] if isinstance(v, np.ndarray)
                    and v.ndim >= 1 and v.shape[0] == c.shape[0]
                    else v) for k, v in rows.items()}
        n_scenes = int(len(np.unique(rows["scene_id"])))
        print(f"[exp] dropped {int(c.sum())} C candidates "
              f"(kept {n_scenes} scenes)")


    print(f"[exp] {n_scenes} scenes, {len(rows['y'])} candidates")
    residual = args.task == "residual"
    if residual:
        # Q1: y_res = labelled final - model pdm_score (Huber regression)
        rows["y"] = (rows["final"] - rows["pdm"])[:, None].astype(np.float32)
        print(f"[exp] residual target: mean={rows['y'].mean():+.4f} "
              f"std={rows['y'].std():.4f}")
    target_names = ["y_res"] if residual else TARGET_NAMES

    logs = sorted(set(rows["log"].tolist()))
    groups = split_logs_by_name(logs, ratios=args.ratios, seed=args.split_seed)
    group_of = {l: g for g, ls in groups.items() for l in ls}
    rows["group"] = np.array([group_of[l] for l in rows["log"]])
    print(f"[exp] logs: memory={len(groups['memory'])} "
          f"query_train={len(groups['query_train'])} "
          f"query_val={len(groups['query_val'])}")
    for ti, tname in enumerate(target_names):
        for sp in ("memory", "query_train", "query_val"):
            m = rows["group"] == sp
            print(f"[exp] {'mean' if residual else 'pos-rate'} "
                  f"{tname} {sp}: {rows['y'][m, ti].mean():.4f}")

    # descriptor standardization on query_train (pred_desc variants)
    qtr_mask0 = np.array([group_of[l] for l in rows["log"]]) == "query_train"
    zm, zs = standardize_fit(rows["desc"][qtr_mask0])
    rows["desc_valid"] = np.isfinite(rows["desc"])
    rows["desc_z"] = np.nan_to_num(
        (rows["desc"] - zm) / zs, nan=0.0).astype(np.float32)
    names = descriptor_feature_names(TIMING_FIELDS)
    rows["cont_cols"] = np.array(
        [names.index(n) for n in DESC_CONT_NAMES], dtype=np.int64)

    mem_mask = rows["group"] == "memory"
    mem_rows = {k: rows[k][mem_mask] for k in
                ("x", "y", "scene_id", "log", "desc_z", "desc_valid")}

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
            if args.save_models:
                import torch
                mdir = out_dir / "models"
                mdir.mkdir(exist_ok=True)
                torch.save(
                    {"state_dict": runs[variant][-1]["model"].state_dict(),
                     "meta": {"variant": variant, "seed": seed,
                              "task": args.task,
                              "feat_in": rows["x"].shape[1],
                              "n_targets": rows["y"].shape[1],
                              "use_int": variant in ("retrieval_int",
                                                     "noexp_int"),
                              "desc_dim": (rows["desc_z"].shape[1]
                                           if variant.startswith("pred_desc")
                                           else 0),
                              "n_cont": len(rows["cont_cols"]),
                              "latent": 64,
                              "zm": np.asarray(zm).tolist(),
                              "zs": np.asarray(zs).tolist()}},
                    mdir / f"{variant}_seed{seed}.pt")

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
        if residual:
            for sub_name, sub_mask in (("all", None), ("conflict", val_sub)):
                sel = np.ones(len(val_y), bool) if sub_mask is None else sub_mask
                if sel.sum() == 0:
                    continue
                pv, tv = pred_va[sel, 0], val_y[sel, 0]
                mae = float(np.abs(pv - tv).mean())
                corr = float(np.corrcoef(pv, tv)[0, 1]) if pv.std() > 0 else 0.0
                res = {"mae": mae, "pearson": corr}
                results[f"{variant}|y_res|{sub_name}"] = res
                table_rows.append([variant, "y_res", sub_name,
                                   f"{mae:.4f}", f"{corr:.4f}", "-"])
            continue
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
        C = rows["y"].shape[1]
        import torch
        conf_per_seed = [
            conf_rows(r["model"],
                      torch.from_numpy(
                          rows["x"][np.flatnonzero(val_mask)]).float(),
                      args.device, conf_col=0,
                      zm_c=float(zm[0]), zs_c=float(zs[0]))
            for r in runs[real]]
        conf_arr = (None if all(c is None for c in conf_per_seed)
                    else np.stack([
                        c if c is not None else np.full(
                            len(conf_per_seed[0]), np.nan)
                        for c in conf_per_seed]).mean(0).reshape(
                            len(val_scene_ids), -1).astype(np.float32))
        kw = {} if conf_arr is None else {"conf": conf_arr}
        save_npz(
            qva_risk_dir / f"{variant}.npz",
            tokens=val_scene_tokens,
            risk=per_seed.mean(0).reshape(len(val_scene_ids), -1, C).astype(
                np.float32),
            **kw,
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
    if residual:
        md.append("| variant | target | subset | MAE | pearson | - |")
    else:
        md.append("| variant | target | subset | Brier | AUPRC | top1-risk |")
    md.append("|---|---|---|---|---|---|")
    for r in table_rows:
        md.append("| " + " | ".join(r) + " |")
    (out_dir / "train_experience_results.md").write_text("\n".join(md) + "\n")
    with open(out_dir / "train_experience_results.json", "w") as f:
        json.dump({k: {m: (np.asarray(v).tolist() if np.isscalar(v)
                          else list(v)) for m, v in res.items()}
                   for k, res in results.items()}, f, indent=2)

    # ---- navtest risk export (memory = navtrain rows only) ------------------
    if args.navtest_labels_dir and args.navtest_export_dir:
        print("[exp] predicting navtest risks", flush=True)
        nt = load_rows(Path(args.navtest_labels_dir), Path(args.navtest_export_dir),
                       args.dt_enter_thresh)
        nt["desc_valid"] = np.isfinite(nt["desc"])
        nt["desc_z"] = np.nan_to_num(
            (nt["desc"] - zm) / zs, nan=0.0).astype(np.float32)
        nt["cont_cols"] = rows["cont_cols"]
        risk_dir = out_dir / "navtest_risk"
        n_nt = len(nt["tokens"])
        C = rows["y"].shape[1]
        import torch
        nt_x = torch.from_numpy(nt["x"]).float()
        for variant, seed_runs in runs.items():
            per_seed = np.stack([  # (nseed, S*K, C) -> (nseed, S, K, C)
                predict_navtest(r["model"], variant, nt, mem_rows, args, s)
                .reshape(n_nt, -1, C)
                for s, r in zip(args.seeds, seed_runs)
            ])
            conf_per_seed = [
                conf_rows(r["model"], nt_x, args.device, conf_col=0,
                          zm_c=float(zm[0]), zs_c=float(zs[0]))
                for r in seed_runs]
            conf_arr = (None if all(c is None for c in conf_per_seed)
                        else np.stack([
                            c if c is not None else np.full(
                                len(conf_per_seed[0]), np.nan)
                            for c in conf_per_seed]).mean(0).reshape(
                                n_nt, -1).astype(np.float32))
            kw = {} if conf_arr is None else {"conf": conf_arr}
            save_npz(
                risk_dir / f"{variant}.npz",
                tokens=nt["tokens"],
                log_names=nt["scene_log"],
                risk=per_seed.mean(0).astype(np.float32),      # (S,K,C)
                risk_per_seed=per_seed.astype(np.float32),     # (nseed,S,K,C)
                **kw,
            )
    print(f"[exp] done in {time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
