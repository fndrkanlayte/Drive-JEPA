"""Step 3 variants with per-vehicle interaction targets.

b1aux: B1 trunk + 6-logit BCE + direct auxiliary regression of the per-vehicle
       interaction targets (same targets b3 encodes) -> fair "no latent" control.
b3:    action-conditioned, vehicle-factorised outcome-latent predictor.
       For candidate k and slot j in {ego, veh_1..veh_M}:
           y_hat_kj = P(a_k + e_j, z_t)            (cross-attn to scene tokens)
           target   = sg(EMA E_y(o_kj))             (JEPA target)
       + readout h(y_hat_k) -> 6 logits, slot-existence head,
       + within-scene relational loss (candidate-candidate latent geometry),
       + optional factual future-latent loss on the expert action:
           P_f(z_t, tau*) ~ sg(EMA SceneEncoder(image_feature_{t+H})).

Targets are labels (they may use future / attribution information); inputs
remain image_feature, proposal_feature and the candidate poses only.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .descriptors import DESCRIPTOR_FIELD_INDEX
from .ewm import (
    D_LATENT,
    D_SCENE,
    N_SUB,
    ActionEncoder,
    PredictorLayer,
    SceneEncoder,
    vicreg_var_cov,
)

AGENT_TARGET_SCALE = {
    "conflict": 1.0, "t_k_in": 4.0, "t_k_out": 4.0, "t_j_in": 4.0, "t_j_out": 4.0,
    "dt_enter": 5.0, "overlap": 4.0, "pet": 5.0,
    "k_in_censored": 1.0, "k_out_censored": 1.0,
    "j_in_censored": 1.0, "j_out_censored": 1.0, "multi_entry": 1.0,
    "min_dist": 20.0, "att_collision": 1.0, "att_ttc": 1.0,
    "rel_x": 50.0, "rel_y": 20.0, "rel_heading": float(np.pi), "speed": 15.0,
}
AGENT_TARGET_FIELDS = list(AGENT_TARGET_SCALE)
AGENT_BINARY_FIELDS = ["conflict", "k_in_censored", "k_out_censored",
                       "j_in_censored", "j_out_censored", "multi_entry",
                       "att_collision", "att_ttc"]
AGENT_TARGET_IDX = np.array([DESCRIPTOR_FIELD_INDEX[f] for f in AGENT_TARGET_FIELDS])
BINARY_MASK = torch.tensor([f in AGENT_BINARY_FIELDS for f in AGENT_TARGET_FIELDS])
N_AGENT_F = len(AGENT_TARGET_FIELDS)


def agent_targets(descriptors: np.ndarray, vehicle_mask: np.ndarray):
    """descriptors (K,M,F_all), vehicle_mask (K,M) -> values (K,M,F), valid (K,M,F), slot (K,M)."""
    d = np.asarray(descriptors, dtype=np.float32)[..., AGENT_TARGET_IDX]
    slot = np.asarray(vehicle_mask, dtype=bool)
    valid = ~np.isnan(d) & slot[..., None]
    scale = np.array([AGENT_TARGET_SCALE[f] for f in AGENT_TARGET_FIELDS], dtype=np.float32)
    vals = np.nan_to_num(d, nan=0.0) / scale
    vals = np.clip(vals, -5.0, 5.0)
    return vals.astype(np.float32), valid.astype(np.float32), slot.astype(np.float32)


def agent_aux_loss(pred: torch.Tensor, vals: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Masked Huber on continuous fields + BCE on binary fields. pred/vals/valid (B,K,M,F)."""
    bmask = BINARY_MASK.to(pred.device).view(1, 1, 1, -1).float()
    hub = F.smooth_l1_loss(pred, vals, reduction="none")
    bce = F.binary_cross_entropy_with_logits(pred, vals.clamp(0, 1), reduction="none")
    per = bce * bmask + hub * (1 - bmask)
    return (per * valid).sum() / valid.sum().clamp_min(1.0)


class _Trunk(nn.Module):
    def __init__(self, n_layers: int, d_model: int = D_SCENE,
                 dropout: float = 0.0, no_pfeat: bool = False,
                 traj_jitter: float = 0.0) -> None:
        super().__init__()
        self.scene = SceneEncoder(d_model)
        self.action = ActionEncoder(d_model, no_pfeat=no_pfeat,
                                    traj_jitter=traj_jitter)
        self.layers = nn.ModuleList(PredictorLayer(d_model, dropout=dropout)
                                    for _ in range(n_layers))

    def forward_trunk(self, image_feature, proposal_feature, trajectories):
        z = self.scene(image_feature)
        a = self.action(proposal_feature, trajectories)
        for layer in self.layers:
            a = layer(a, z)
        return a, z


class B1Aux(_Trunk):
    """Direct regression + per-vehicle auxiliary regression (no latent)."""

    def __init__(self, n_layers: int = 3, n_slots: int = 4, d_model: int = D_SCENE,
                 dropout: float = 0.0, no_pfeat: bool = False,
                 traj_jitter: float = 0.0) -> None:
        super().__init__(n_layers, d_model, dropout=dropout, no_pfeat=no_pfeat,
                         traj_jitter=traj_jitter)
        self.n_slots = n_slots
        self.head = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, N_SUB))
        self.aux = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(),
                                 nn.Linear(256, n_slots * (N_AGENT_F + 1)))

    def forward(self, image_feature, proposal_feature, trajectories) -> Dict[str, torch.Tensor]:
        a, _ = self.forward_trunk(image_feature, proposal_feature, trajectories)
        aux = self.aux(a).view(*a.shape[:2], self.n_slots, N_AGENT_F + 1)
        return {"logits": self.head(a), "agent_pred": aux[..., :N_AGENT_F],
                "slot_logit": aux[..., N_AGENT_F]}


FUT_GRID = (16, 32)   # Drive-JEPA image_feature token grid
FUT_POOL = (4, 4)     # -> 4x8 = 32 target tokens


def future_target(image_future: torch.Tensor) -> torch.Tensor:
    """Frozen JEPA feature of o_{t+H}, avg-pooled to 32 tokens and layer-normed (no params).

    Using the frozen encoder output (as in V-JEPA 2-AC) instead of an EMA of our own trainable
    SceneEncoder removes the collapse/drift path of a moving target."""
    B, N, D = image_future.shape
    g = image_future.float().transpose(1, 2).reshape(B, D, *FUT_GRID)
    g = F.avg_pool2d(g, FUT_POOL).flatten(2).transpose(1, 2)            # (B,32,D)
    return F.layer_norm(g, (D,))


class FuturePredictor(nn.Module):
    """P_f(z_t, tau*) -> z_hat_{t+H} (B, 32, D) in the frozen JEPA feature space."""

    def __init__(self, d_model: int = D_SCENE, n_layers: int = 2) -> None:
        super().__init__()
        n_tok = (FUT_GRID[0] // FUT_POOL[0]) * (FUT_GRID[1] // FUT_POOL[1])
        self.queries = nn.Parameter(torch.randn(n_tok, d_model) * 0.02)
        self.layers = nn.ModuleList(PredictorLayer(d_model) for _ in range(n_layers))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, z: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q = self.queries.unsqueeze(0) + act.unsqueeze(1)
        mem = torch.cat([z, act.unsqueeze(1)], dim=1)
        for layer in self.layers:
            q = layer(q, mem)
        return self.out(q)


class EWMStructured(_Trunk):
    """b3: vehicle-factorised outcome-latent world model."""

    def __init__(self, n_layers: int = 3, n_slots: int = 4, d_model: int = D_SCENE,
                 d_latent: int = D_LATENT, use_future: bool = True,
                 dropout: float = 0.0, no_pfeat: bool = False,
                 traj_jitter: float = 0.0) -> None:
        super().__init__(n_layers, d_model, dropout=dropout, no_pfeat=no_pfeat,
                         traj_jitter=traj_jitter)
        self.n_slots = n_slots
        self.d_latent = d_latent
        self.slot_emb = nn.Parameter(torch.randn(n_slots + 1, d_model) * 0.02)  # 0=ego
        self.slot_layer = PredictorLayer(d_model)
        self.y_head = nn.Linear(d_model, d_latent)
        self.readout = nn.Sequential(nn.Linear((n_slots + 1) * d_latent, 256), nn.GELU(),
                                     nn.Linear(256, N_SUB))
        self.exist = nn.Linear(d_latent, 1)
        self.enc_agent = nn.Sequential(nn.Linear(2 * N_AGENT_F + 1, 128), nn.GELU(),
                                       nn.Linear(128, d_latent))
        self.enc_ego = nn.Sequential(nn.Linear(N_SUB, 128), nn.GELU(), nn.Linear(128, d_latent))
        self.enc_agent_ema = _frozen_copy(self.enc_agent)
        self.enc_ego_ema = _frozen_copy(self.enc_ego)
        self.use_future = use_future
        if use_future:
            self.expert_token = nn.Parameter(torch.zeros(d_model))
            self.fut = FuturePredictor(d_model)

    @torch.no_grad()
    def update_ema(self, m: float = 0.996) -> None:
        pairs = [(self.enc_agent_ema, self.enc_agent), (self.enc_ego_ema, self.enc_ego)]
        for tgt, src in pairs:
            for pe, pl in zip(tgt.parameters(), src.parameters()):
                pe.mul_(m).add_(pl, alpha=1.0 - m)

    def encode_outcome(self, vals, valid, slot, labels, ema: bool):
        ea, ee = (self.enc_agent_ema, self.enc_ego_ema) if ema else (self.enc_agent, self.enc_ego)
        y_ag = ea(torch.cat([vals * valid, valid, slot.unsqueeze(-1)], dim=-1))  # (B,K,M,L)
        y_ego = ee(labels).unsqueeze(2)                                           # (B,K,1,L)
        return torch.cat([y_ego, y_ag], dim=2)                                    # (B,K,M+1,L)

    def predict(self, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, K, D = a.shape
        S = self.n_slots + 1
        q = (a.unsqueeze(2) + self.slot_emb.view(1, 1, S, D)).reshape(B, K * S, D)
        q = self.slot_layer(q, z)
        return self.y_head(q).view(B, K, S, self.d_latent)

    def read(self, y: torch.Tensor) -> torch.Tensor:
        return self.readout(y.flatten(-2))

    def forward(self, image_feature, proposal_feature, trajectories) -> Dict[str, torch.Tensor]:
        a, z = self.forward_trunk(image_feature, proposal_feature, trajectories)
        y_hat = self.predict(a, z)
        return {"y_hat": y_hat, "readout": self.read(y_hat), "z": z,
                "slot_logit": self.exist(y_hat[:, :, 1:]).squeeze(-1)}

    def future_loss(self, z, expert_traj, image_future, has_future):
        act = self.action.traj_mlp(expert_traj.flatten(-2))
        act = self.action.proj(torch.cat([self.expert_token.expand(act.shape[0], -1), act], -1))
        z_hat = self.fut(z, act)
        with torch.no_grad():
            z_tgt = future_target(image_future)
        per = F.mse_loss(z_hat, z_tgt, reduction="none").mean((1, 2))
        return (per * has_future).sum() / has_future.sum().clamp_min(1.0)


def _frozen_copy(m: nn.Module) -> nn.Module:
    import copy
    c = copy.deepcopy(m)
    for p in c.parameters():
        p.requires_grad_(False)
    return c


def relational_loss(y_hat: torch.Tensor, y_tgt: torch.Tensor) -> torch.Tensor:
    """Match within-scene candidate-candidate cosine geometry. (B,K,S,L)."""
    a = F.normalize(y_hat.flatten(2), dim=-1)
    b = F.normalize(y_tgt.flatten(2), dim=-1)
    return F.mse_loss(a @ a.transpose(1, 2), b @ b.transpose(1, 2))


def relational_loss_ego(y_ego_hat: torch.Tensor, y_ego_tgt: torch.Tensor) -> torch.Tensor:
    """Ego-slot-only relational geometry over all candidates. (B,K,L)."""
    a = F.normalize(y_ego_hat, dim=-1)
    b = F.normalize(y_ego_tgt, dim=-1)
    return F.mse_loss(a @ a.transpose(1, 2), b @ b.transpose(1, 2))


def xs_loss(y_ego: torch.Tensor, subs: torch.Tensor, log_ids: torch.Tensor,
            n_sample: int = 512, tau: float = 0.1, t_s: float = 0.1) -> torch.Tensor:
    """Cross-scene soft InfoNCE on the ego-slot outcome latent.

    y_ego (B,K,L) candidate latents, subs (B,K,6) true subscores,
    log_ids (B,K) integer log id per candidate. Prediction: softmax over
    cos(y_a, y_b)/tau restricted to different-log b. Target: softmax over
    exp(-||s_a - s_b||_1 / T_s) on the same restriction. Loss = KL(target||pred)
    averaged over rows whose target support is non-empty.
    """
    B, K, L = y_ego.shape
    y = F.normalize(y_ego.reshape(-1, L).float(), dim=-1)
    s = subs.reshape(-1, subs.shape[-1]).float()
    lg = log_ids.reshape(-1)
    n = y.shape[0]
    if n > n_sample:
        idx = torch.randperm(n, device=y.device)[:n_sample]
        y, s, lg = y[idx], s[idx], lg[idx]
        n = n_sample
    sim = y @ y.T / tau                                             # (n,n)
    d = (s[:, None, :] - s[None, :, :]).abs().sum(-1)               # L1 on subscores
    allowed = (lg[:, None] != lg[None, :]) & ~torch.eye(n, dtype=torch.bool, device=y.device)
    w = torch.exp(-d / t_s) * allowed
    has = w.sum(-1) > 0
    if not bool(has.any()):
        return y.new_zeros(())
    sim = sim.masked_fill(~allowed, -1e9)
    logp = F.log_softmax(sim, dim=-1)
    tgt = w[has] / w[has].sum(-1, keepdim=True)
    lp = logp[has]
    kl = (tgt * (torch.log(tgt.clamp_min(1e-12)) - lp)).sum(-1)
    return kl.mean()


def b3_loss(model: EWMStructured, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
            lam_rel: float = 0.5, lam_fut: float = 0.5, lam_exist: float = 0.2,
            bank_k: int = 0):
    """b3 loss. With ``bank_k`` > 0, the last bank_k candidates are bank
    trajectories: ego-slot outcome target + readout BCE apply to them, agent
    slots / existence are masked out, and the relational loss is ego-slot-only
    over all candidates."""
    vals, valid, slot = batch["agent_vals"], batch["agent_valid"], batch["agent_slot"]
    labels = batch["labels"]
    y_on = model.encode_outcome(vals, valid, slot, labels, ema=False)
    with torch.no_grad():
        y_t = model.encode_outcome(vals, valid, slot, labels, ema=True)
    w = torch.cat([torch.ones_like(slot[..., :1]), slot], dim=-1)            # (B,K,S)
    if bank_k:
        w[:, -bank_k:, 1:] = 0.0                                             # bank rows: ego only
    l2 = ((out["y_hat"] - y_t).pow(2).mean(-1) * w).sum() / w.sum().clamp_min(1.0)
    bce_hat = F.binary_cross_entropy_with_logits(out["readout"], labels)
    k0 = out["y_hat"].shape[1] - bank_k
    # bank rows have no agent-slot supervision: the through-encoder BCE (which
    # reads all slots) is restricted to B0 rows.
    bce_y = F.binary_cross_entropy_with_logits(
        model.read(y_on[:, :k0]), labels[:, :k0])
    vic = vicreg_var_cov(y_on[w.bool()])
    if bank_k:
        rel = relational_loss_ego(out["y_hat"][:, :, 0], y_t[:, :, 0])
        exist = F.binary_cross_entropy_with_logits(
            out["slot_logit"][:, :k0], slot[:, :k0])
    else:
        rel = relational_loss(out["y_hat"], y_t)
        exist = F.binary_cross_entropy_with_logits(out["slot_logit"], slot)
    loss = l2 + bce_hat + 0.5 * bce_y + vic + lam_rel * rel + lam_exist * exist
    parts = {"l2": l2, "bce_hat": bce_hat, "bce_y": bce_y, "vic": vic, "rel": rel, "exist": exist}
    if model.use_future and "image_future" in batch:
        fut = model.future_loss(out["z"], batch["expert_traj"], batch["image_future"],
                                batch["has_future"] * batch["has_traj"])
        loss = loss + lam_fut * fut
        parts["fut"] = fut
    return loss, {k: float(v.detach()) for k, v in parts.items()}


def b1aux_loss(out, batch, lam_aux: float = 0.5, lam_exist: float = 0.2):
    bce = F.binary_cross_entropy_with_logits(out["logits"], batch["labels"])
    aux = agent_aux_loss(out["agent_pred"], batch["agent_vals"], batch["agent_valid"])
    exist = F.binary_cross_entropy_with_logits(out["slot_logit"], batch["agent_slot"])
    loss = bce + lam_aux * aux + lam_exist * exist
    return loss, {"bce": float(bce.detach()), "aux": float(aux.detach()), "exist": float(exist.detach())}
