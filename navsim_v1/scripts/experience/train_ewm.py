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


class LatentDataset(Dataset):
    def __init__(self, tokens, lat_index: dict, labels_dir: Path,
                 structured: bool = False, future_map: dict = None):
        self.structured = structured
        self.future_map = future_map
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
                 "image_future", "has_future"]


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


def build_model(name: str, n_layers: int, use_future: bool = False):
    if name in ("b1", "b2"):
        return EWMJEPA(n_layers=n_layers, direct=(name == "b1"))
    if name == "b1aux":
        return B1Aux(n_layers=n_layers)
    if name == "b3":
        return EWMStructured(n_layers=n_layers, use_future=use_future)
    raise ValueError(name)


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
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

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
    train_ds = LatentDataset(train_tokens, lat_index, labels_dir, structured, fmap)
    val_ds = LatentDataset(val_tokens, lat_index, labels_dir, structured, None)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.n_layers, use_future=fmap is not None).to(device)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = []
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
            bd["labels"] = labels

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(img, pf, tr)
                if args.model == "b1aux":
                    loss, parts = b1aux_loss(out, bd, lam_aux=args.lam_aux)
                elif args.model == "b3":
                    loss, parts = b3_loss(model, out, bd, lam_rel=args.lam_rel,
                                          lam_fut=args.lam_fut)
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
        history.append(row)
        print(f"[train] {row}", flush=True)

    torch.save({"model": model.state_dict(), "args": {**vars(args), "use_future": fmap is not None}},
               out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    print(f"[train] saved {out_dir}/model.pt")


if __name__ == "__main__":
    main()
