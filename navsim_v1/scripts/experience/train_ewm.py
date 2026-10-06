#!/usr/bin/env python
"""Step 3 trainer: minimal EWM-JEPA (B2) and direct-regression baseline (B1).

Data (per token):
    latents npz:  image_feature (512,256) f16, proposal_feature (32,256) f16,
                  proposals (32,8,3) f32, pdm_score (32,), selected_idx
    labels npz:   subscores (32,6) [NC,DAC,EP,TTC,Comfort,final],
                  main_desc_noatt (32,F) — TIMING_FIELDS subset is used for o_k

Outcome feature o_k (18,) = [TIMING_FIELDS(12), subscores[k,:5], final].

    python scripts/experience/train_ewm.py \
        --latents_dir $E/latents_navtrain --labels_dir $E/navtrain_labels \
        --train_tokens $E/split_navtrain/train_tokens.txt \
        --val_tokens   $E/split_navtrain/val_tokens.txt \
        --model b2 --seed 0 --epochs 15 --out_dir $E/ewm_runs/b2_s0
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))

from navsim.agents.drive_jepa_perception_based.experience.descriptors import (  # noqa: E402
    DESCRIPTOR_FIELD_INDEX,
    TIMING_FIELDS,
)
from navsim.agents.drive_jepa_perception_based.experience.ewm import (  # noqa: E402
    EWMJEPA,
    N_SUB,
    vicreg_var_cov,
)
from navsim.agents.drive_jepa_perception_based.experience.ewm_structured import (  # noqa: E402
    B1Aux,
    EWMStructured,
    agent_targets,
    b1aux_loss,
    b3_loss,
    xs_loss,
)
from navsim.agents.drive_jepa_perception_based.experience.records import load_npz  # noqa: E402

TIMING_IDX = np.array([DESCRIPTOR_FIELD_INDEX[f] for f in TIMING_FIELDS])

# fixed per-field scales (nan->0 before scaling); keeps timing magnitudes ~O(1)
TIMING_SCALE = {
    "conflict": 1.0, "dt_enter": 5.0, "pet": 5.0,
    "k_in_censored": 1.0, "k_out_censored": 1.0,
    "j_in_censored": 1.0, "j_out_censored": 1.0, "multi_entry": 1.0,
    "rel_x": 50.0, "rel_y": 20.0, "rel_heading": math.pi, "speed": 15.0,
}


def outcome_features(labels: dict) -> np.ndarray:
    """(K,18) = [timing fields scaled, subscores[:5], final]."""
    md = np.asarray(labels["main_desc_noatt"], dtype=np.float32)   # (K,F)
    timing = md[:, TIMING_IDX]
    timing = np.nan_to_num(timing, nan=0.0)
    scales = np.array([TIMING_SCALE[f] for f in TIMING_FIELDS], dtype=np.float32)
    timing = timing / scales
    sub = np.asarray(labels["subscores"], dtype=np.float32)        # (K,6)
    return np.concatenate([timing, sub[:, :5], sub[:, 5:6]], axis=-1)


def load_bank_npz(path: str) -> dict:
    """bank_ours.npz -> {token: (trajs (N,8,3), subs (N,6))}."""
    z = np.load(path, allow_pickle=True)
    toks, counts = z["tokens"], z["counts"]
    trajs, subs = z["trajs"], z["subs"]
    bank, off = {}, 0
    for t, c in zip(toks, counts):
        bank[str(t)] = (trajs[off:off + c], subs[off:off + c])
        off += c
    return bank


class LatentDataset(Dataset):
    def __init__(self, tokens, lat_index: dict, labels_dir: Path,
                 structured: bool = False, future_map: dict = None,
                 bank: dict = None, bank_per_scene: int = 0):
        self.structured = structured
        self.future_map = future_map
        self.bank = bank
        self.bank_per_scene = bank_per_scene
        self.lat_index = lat_index
        self.items = []
        labels_dir = Path(labels_dir)
        for t in tokens:
            lp = labels_dir / f"{t}.npz"
            if t in lat_index and lp.exists():
                self.items.append((t, lat_index[t], lp))
        self.items.sort()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        token, lat_path, lab_path = self.items[i]
        lat = load_npz(lat_path)
        lab = load_npz(lab_path)
        extra = {}
        if self.bank is not None and self.bank_per_scene > 0:
            bt, bs = self.bank.get(token, (None, None))
            bk_traj = np.zeros((self.bank_per_scene, 8, 3), np.float32)
            bk_sub = np.zeros((self.bank_per_scene, 6), np.float32)
            if bt is not None and len(bt) > 0:
                n = len(bt)
                unsafe = np.where((bs[:, 0] < 0.5) | (bs[:, 1] < 0.5))[0]
                n_half = self.bank_per_scene // 2
                pick_u = np.random.choice(unsafe, min(n_half, len(unsafe)),
                                          replace=False) if len(unsafe) else np.array([], int)
                rest = np.setdiff1d(np.arange(n), pick_u)
                need = self.bank_per_scene - len(pick_u)
                pick_r = np.random.choice(rest, min(need, len(rest)),
                                          replace=False) if len(rest) else np.array([], int)
                pick = np.concatenate([pick_u, pick_r])
                if len(pick) < self.bank_per_scene:        # bank smaller than BK
                    pad = np.random.choice(pick, self.bank_per_scene - len(pick),
                                           replace=True)
                    pick = np.concatenate([pick, pad])
                bk_traj = np.asarray(bt[pick], np.float32)
                bk_sub = np.asarray(bs[pick], np.float32)
            extra.update(bank_trajs=bk_traj, bank_labels=bk_sub)
        if self.structured:
            vals, valid, slot = agent_targets(lab["descriptors"], lab["vehicle_mask"])
            extra.update(agent_vals=vals, agent_valid=valid, agent_slot=slot)
            if "trajectory" in lat:
                extra.update(expert_traj=np.asarray(lat["trajectory"], dtype=np.float32),
                             has_traj=np.float32(1.0))
            else:
                extra.update(expert_traj=np.zeros((8, 3), np.float32), has_traj=np.float32(0.0))
        if self.future_map is not None:
            ft = self.future_map.get(token, {}).get("future_token")
            fp = self.lat_index.get(ft) if ft else None
            if fp is not None:
                extra.update(image_future=np.asarray(load_npz(fp)["image_feature"], dtype=np.float32),
                             has_future=np.float32(1.0))
            else:
                extra.update(image_future=np.zeros_like(np.asarray(lat["image_feature"], dtype=np.float32)),
                             has_future=np.float32(0.0))
        return dict(
            **extra,
            token=token,
            log_name=str(lat["log_name"]),
            image_feature=np.asarray(lat["image_feature"], dtype=np.float32),
            proposal_feature=np.asarray(lat["proposal_feature"], dtype=np.float32),
            proposals=np.asarray(lat["proposals"], dtype=np.float32),
            pdm_score=np.asarray(lat["pdm_score"], dtype=np.float32),
            outcomes=outcome_features(lab),
            labels=np.asarray(lab["subscores"], dtype=np.float32),
        )


def build_index(latents_dir: Path) -> dict:
    idx = {}
    for log_dir in sorted(Path(latents_dir).iterdir()):
        if log_dir.is_dir():
            for f in log_dir.glob("*.npz"):
                idx[f.stem] = f
    return idx


OPTIONAL_KEYS = ["agent_vals", "agent_valid", "agent_slot", "expert_traj", "has_traj",
                 "image_future", "has_future", "bank_trajs", "bank_labels"]


def collate(batch):
    extra = {k: torch.from_numpy(np.stack([b[k] for b in batch]))
             for k in OPTIONAL_KEYS if k in batch[0]}
    return dict(
        **extra,
        tokens=[b["token"] for b in batch],
        log_names=[b["log_name"] for b in batch],
        image_feature=torch.from_numpy(np.stack([b["image_feature"] for b in batch])),
        proposal_feature=torch.from_numpy(np.stack([b["proposal_feature"] for b in batch])),
        proposals=torch.from_numpy(np.stack([b["proposals"] for b in batch])),
        pdm_score=torch.from_numpy(np.stack([b["pdm_score"] for b in batch])),
        outcomes=torch.from_numpy(np.stack([b["outcomes"] for b in batch])),
        labels=torch.from_numpy(np.stack([b["labels"] for b in batch])),
    )


def auprc(scores: np.ndarray, y: np.ndarray) -> float:
    """Area under precision-recall curve (sklearn-free)."""
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(-scores)
    y = y[order].astype(np.float64)
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / tp[-1]
    # step-wise AP
    return float(np.sum(prec * np.diff(np.concatenate([[0.0], rec]))))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def evaluate(model, loader, device, direct: bool) -> dict:
    model.eval()
    scores, unsafe_nc, unsafe_ttc, finals_pred, finals_true = [], [], [], [], []
    native_top1_unsafe, model_top1_unsafe = [], []
    for b in loader:
        img = b["image_feature"].to(device, non_blocking=True)
        pf = b["proposal_feature"].to(device, non_blocking=True)
        tr = b["proposals"].to(device, non_blocking=True)
        labels = b["labels"].numpy()                      # (B,K,6)
        out = model(img, pf, tr)
        logits = out["logits"] if "logits" in out else out["readout"]
        probs = torch.sigmoid(logits).float().cpu().numpy()  # (B,K,6)
        final = probs[..., 5]
        sel = final.argmax(1)
        nat = b["pdm_score"].numpy().argmax(1)
        B, K = final.shape
        for i in range(B):
            scores.append(probs[i])
            unsafe_nc.append((labels[i, :, 0] < 1.0).astype(np.float32))
            unsafe_ttc.append((labels[i, :, 3] < 1.0).astype(np.float32))
            finals_true.append(labels[i, :, 5])
            finals_pred.append(final[i])
            model_top1_unsafe.append(labels[i, sel[i], 0] < 1.0)
            native_top1_unsafe.append(labels[i, nat[i], 0] < 1.0)
    s = np.concatenate(scores)          # (N,6)
    unc = np.concatenate(unsafe_nc)
    utt = np.concatenate(unsafe_ttc)
    ft = np.concatenate(finals_true)
    fp = np.concatenate(finals_pred)
    return dict(
        nc_auprc=auprc(1.0 - s[:, 0], unc),
        ttc_auprc=auprc(1.0 - s[:, 3], utt),
        model_top1_unsafe_rate=float(np.mean(model_top1_unsafe)),
        native_top1_unsafe_rate=float(np.mean(native_top1_unsafe)),
        spearman_final=spearman(fp, ft),
    )


def build_model(name: str, n_layers: int, use_future: bool = False,
                dropout: float = 0.0, no_pfeat: bool = False,
                traj_jitter: float = 0.0):
    if name in ("b1", "b2"):
        return EWMJEPA(n_layers=n_layers, direct=(name == "b1"), dropout=dropout,
                       no_pfeat=no_pfeat, traj_jitter=traj_jitter)
    if name == "b1aux":
        return B1Aux(n_layers=n_layers, dropout=dropout, no_pfeat=no_pfeat,
                     traj_jitter=traj_jitter)
    if name == "b3":
        return EWMStructured(n_layers=n_layers, use_future=use_future,
                             dropout=dropout, no_pfeat=no_pfeat,
                             traj_jitter=traj_jitter)
    raise ValueError(name)


@torch.no_grad()
def collect_yego(model, loader, device):
    """-> ego-slot y_hat (N,L), true finals (N,), per-candidate log names (N,)."""
    ye, fin, lg = [], [], []
    for b in loader:
        out = model(b["image_feature"].to(device),
                    b["proposal_feature"].to(device),
                    b["proposals"].to(device))
        ye.append(out["y_hat"][:, :, 0].float().cpu().numpy())
        fin.append(b["labels"][..., 5].numpy())
        for i, ln in enumerate(b["log_names"]):
            lg.extend([ln] * b["labels"].shape[1])
    return (np.concatenate(ye).reshape(-1, ye[0].shape[-1]),
            np.concatenate(fin).reshape(-1), np.asarray(lg))


def knn_val_mae(q_y, q_fin, q_logs, b_y, b_fin, b_logs, k: int = 16,
                t: float = 0.1, device="cpu") -> float:
    """Cross-log top-k cosine kNN of ego-slot y_hat -> weighted final MAE.

    weights = softmax(cos / t) over the k neighbours (t=0.1). Logs are mapped
    to integer ids once and the same-log exclusion is a vectorised mask, so
    the (chunk, bank) sim matrix stays small and there is no per-row python.
    """
    qn = q_y / (np.linalg.norm(q_y, axis=1, keepdims=True) + 1e-8)
    bn = b_y / (np.linalg.norm(b_y, axis=1, keepdims=True) + 1e-8)
    bt = torch.from_numpy(bn).to(device)
    bf = torch.from_numpy(b_fin).float().to(device)
    uniq = np.unique(np.concatenate([np.asarray(q_logs), np.asarray(b_logs)]))
    lid = {v: i for i, v in enumerate(uniq)}
    ql = torch.from_numpy(np.array([lid[v] for v in q_logs])).to(device)
    bl = torch.from_numpy(np.array([lid[v] for v in b_logs])).to(device)
    maes = []
    CH = 1024
    for s in range(0, len(qn), CH):
        qc = torch.from_numpy(qn[s:s + CH]).to(device)
        sim = qc @ bt.T                                     # (c, NB)
        same = ql[s:s + CH, None] == bl[None, :]
        sim = sim.masked_fill(same, -1e9)
        tk = sim.topk(k, dim=-1)                            # (c, k)
        w = torch.softmax(tk.values / t, dim=-1)
        fhat = (w * bf[tk.indices]).sum(-1).cpu().numpy()
        maes.append(np.abs(fhat - q_fin[s:s + CH]))
    return float(np.concatenate(maes).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--latents_dir", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--train_tokens", required=True)
    p.add_argument("--val_tokens", required=True)
    p.add_argument("--model", choices=["b1", "b2", "b1aux", "b3"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--ema_m", type=float, default=0.996)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_train_tokens", type=int, default=None)
    p.add_argument("--future_map", default=None,
                   help="b3 only: build_future_map.py JSON for the z_{t+H} loss")
    p.add_argument("--lam_rel", type=float, default=0.5)
    p.add_argument("--lam_fut", type=float, default=0.5)
    p.add_argument("--lam_aux", type=float, default=0.5)
    p.add_argument("--no_pfeat", action="store_true",
                   help="ActionEncoder drops proposal_feature (learned const) "
                        "so arbitrary trajectories can be embedded")
    p.add_argument("--bank_npz", default=None,
                   help="bank_ours.npz (T0): extra per-scene candidate trajs; "
                        "requires --no_pfeat, train split only")
    p.add_argument("--bank_per_scene", type=int, default=16)
    p.add_argument("--lam_xs", type=float, default=0.0,
                   help="cross-scene soft InfoNCE on ego-slot y_hat")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--traj_jitter", type=float, default=0.0,
                   help="sigma (m) of train-time xy jitter on candidate trajs")
    p.add_argument("--select_metric", choices=["none", "knn_val"], default="none",
                   help="knn_val: save model_best.pt by cross-log kNN finalMAE")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    if args.bank_npz and not args.no_pfeat:
        raise SystemExit("--bank_npz requires --no_pfeat")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tokens = [l.strip() for l in open(args.train_tokens) if l.strip()]
    val_tokens = [l.strip() for l in open(args.val_tokens) if l.strip()]
    if args.max_train_tokens:
        train_tokens = train_tokens[: args.max_train_tokens]

    print(f"[train] indexing latents in {args.latents_dir}", flush=True)
    lat_index = build_index(Path(args.latents_dir))
    labels_dir = Path(args.labels_dir)
    structured = args.model in ("b1aux", "b3")
    fmap = None
    if args.model == "b3" and args.future_map:
        fmap = json.load(open(args.future_map))["map"]
    bank = load_bank_npz(args.bank_npz) if args.bank_npz else None
    train_ds = LatentDataset(train_tokens, lat_index, labels_dir, structured, fmap,
                             bank=bank, bank_per_scene=args.bank_per_scene)
    val_ds = LatentDataset(val_tokens, lat_index, labels_dir, structured, None)
    if bank is not None:
        n_hit = sum(1 for t, _, _ in train_ds.items if t in bank)
        print(f"[train] bank coverage: {n_hit}/{len(train_ds)} train scenes", flush=True)
    if fmap is not None:
        n_fut = sum(1 for t, _, _ in train_ds.items
                    if fmap.get(t, {}).get("future_token") in lat_index)
        print(f"[train] future-latent coverage: {n_fut}/{len(train_ds)}", flush=True)
    print(f"[train] train={len(train_ds)} val={len(val_ds)} (labels-joined)", flush=True)
    assert len(train_ds) > 0 and len(val_ds) > 0

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=True)

    mem_loader = None
    if args.select_metric == "knn_val" and args.model == "b3":
        mem_rng = np.random.default_rng(0)
        mem_tokens = mem_rng.choice(
            train_tokens, size=min(4000, len(train_tokens)), replace=False)
        mem_ds = LatentDataset(list(mem_tokens), lat_index, labels_dir,
                               structured, None)
        mem_loader = DataLoader(mem_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                collate_fn=collate, pin_memory=True)
        print(f"[train] knn_val memory subset: {len(mem_ds)} scenes", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.n_layers, use_future=fmap is not None,
                        dropout=args.dropout, no_pfeat=args.no_pfeat,
                        traj_jitter=args.traj_jitter).to(device)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = []
    best_sel = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses = {}
        nb = 0
        for b in train_loader:
            img = b["image_feature"].to(device, non_blocking=True)
            pf = b["proposal_feature"].to(device, non_blocking=True)
            tr = b["proposals"].to(device, non_blocking=True)
            o = b["outcomes"].to(device, non_blocking=True)     # (B,K,18)
            labels = b["labels"].to(device, non_blocking=True).clamp(0, 1)
            bd = {k: b[k].to(device, non_blocking=True) for k in OPTIONAL_KEYS if k in b}
            K0 = tr.shape[1]
            bank_k = 0
            if "bank_trajs" in bd:
                bk_t = bd.pop("bank_trajs")
                bk_l = bd.pop("bank_labels").clamp(0, 1)
                tr = torch.cat([tr, bk_t], dim=1)
                labels = torch.cat([labels, bk_l], dim=1)
                bank_k = bk_t.shape[1]
                B, M = bd["agent_slot"].shape[0], bd["agent_slot"].shape[2]
                f_agent = bd["agent_vals"].shape[-1]
                bd["agent_vals"] = torch.cat(
                    [bd["agent_vals"], torch.zeros(B, bank_k, M, f_agent, device=device)], 1)
                bd["agent_valid"] = torch.cat(
                    [bd["agent_valid"], torch.zeros(B, bank_k, M, f_agent, device=device)], 1)
                bd["agent_slot"] = torch.cat(
                    [bd["agent_slot"], torch.zeros(B, bank_k, M, device=device)], 1)
            bd["labels"] = labels

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(img, pf, tr)
                if args.model == "b1aux":
                    loss, parts = b1aux_loss(out, bd, lam_aux=args.lam_aux)
                elif args.model == "b3":
                    loss, parts = b3_loss(model, out, bd, lam_rel=args.lam_rel,
                                          lam_fut=args.lam_fut, bank_k=bank_k)
                    if args.lam_xs > 0:
                        inv = np.unique(np.asarray(b["log_names"]),
                                        return_inverse=True)[1]
                        lid = torch.as_tensor(
                            np.repeat(inv[:, None], out["y_hat"].shape[1], axis=1),
                            device=device)
                        xs = xs_loss(out["y_hat"][:, :, 0], labels, lid)
                        loss = loss + args.lam_xs * xs
                        parts["xs"] = float(xs.detach())
                elif args.model == "b1":
                    loss = F.binary_cross_entropy_with_logits(out["logits"], labels)
                    parts = {"bce": loss.item()}
                else:
                    y = model.outcome(o)
                    with torch.no_grad():
                        y_t = model.outcome_ema(o)
                    l2 = F.mse_loss(out["y_hat"], y_t)
                    bce_hat = F.binary_cross_entropy_with_logits(out["readout"], labels)
                    bce_y = F.binary_cross_entropy_with_logits(model.readout(y), labels)
                    vic = vicreg_var_cov(y)
                    loss = l2 + bce_hat + 0.5 * bce_y + vic
                    parts = {"l2": l2.item(), "bce_hat": bce_hat.item(),
                             "bce_y": bce_y.item(), "vic": vic.item()}
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if args.model in ("b2", "b3"):
                model.update_ema(args.ema_m)
            for k, v in parts.items():
                losses[k] = losses.get(k, 0.0) + v
            nb += 1
        metrics = evaluate(model, val_loader, device, direct=(args.model in ("b1", "b1aux")))
        row = {"epoch": epoch, "secs": round(time.time() - t0, 1),
               **{k: round(v / nb, 4) for k, v in losses.items()}, **metrics}
        if args.select_metric == "knn_val" and args.model == "b3":
            ty, tf, tl = collect_yego(model, mem_loader, device)
            vy, vf, vl = collect_yego(model, val_loader, device)
            mae = knn_val_mae(vy, vf, vl, ty, tf, tl, device=device)
            row["knn_val"] = round(mae, 5)
            if mae < best_sel:
                best_sel = mae
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "knn_val": mae,
                            "args": {**vars(args), "use_future": fmap is not None}},
                           out_dir / "model_best.pt")
        history.append(row)
        print(f"[train] {row}", flush=True)

    torch.save({"model": model.state_dict(), "args": {**vars(args), "use_future": fmap is not None}},
               out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    print(f"[train] saved {out_dir}/model.pt")


if __name__ == "__main__":
    main()
