"""Minimal EWM-JEPA (Step 3, no memory): predict interaction-outcome latents.

Per Devin Bot Step-3 spec:

    Scene encoder:  8 learnable queries attention-pool image_feature
                    (B,512,256) -> z (B,8,256). Queries 6 and 7 are tagged
                    'map' and 'agent' (MAP_QUERY_IDX / AGENT_QUERY_IDX) for
                    later decoupled retrieval — same weights for now.
    Action encoder: concat(proposal_feature_k, MLP(flattened traj 8x3)) -> a_k.
                    For the model's own 32 proposals the cached proposal_feature
                    IS the score_external mode='last' output (bit-exact, see
                    test_score_external), so no refiner re-run is needed.
    Predictor P:    L transformer layers; a_k tokens self-attend then
                    cross-attend z -> y_hat_k in R^64.
    Target enc E_y: MLP([TIMING_FIELDS(12), 5 subscores, final]) -> y_k in R^64.
                    An EMA copy (m=0.996) produces the regression targets.
    Readout h:      R^64 -> 6 logits (NC, DAC, TTC, EP, C, final).

Baselines:
    B0 = native pdm_score (no model — handled in the eval script).
    B1 = same encoders + trunk, direct 6-logit regression, BCE only.
    B2 = EWM loss = L2(y_hat, sg(EMA(E_y(o)))) + BCE(h(y_hat))
                 + 0.5*BCE(h(E_y(o))) + VICReg(var+cov on E_y outputs).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

D_SCENE = 256          # image_feature / proposal_feature channel width
D_LATENT = 64          # outcome latent width
N_QUERIES = 8          # scene queries
MAP_QUERY_IDX = 6      # tagged 'map' query (for later use)
AGENT_QUERY_IDX = 7    # tagged 'agent' query (for later use)
N_SUB = 6              # [NC, DAC, TTC, EP, Comfort, final] readout logits


class SceneEncoder(nn.Module):
    """Attention-pool the frozen JEPA image grid with 8 learnable queries."""

    def __init__(self, d_model: int = D_SCENE, n_queries: int = N_QUERIES,
                 n_heads: int = 8) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, image_feature: torch.Tensor) -> torch.Tensor:
        # image_feature: (B, 512, D)
        B = image_feature.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        z, _ = self.attn(q, image_feature, image_feature)
        return self.norm(z)  # (B, 8, D)


class ActionEncoder(nn.Module):
    """Per-candidate action embedding from cached proposal_feature + raw traj."""

    def __init__(self, d_model: int = D_SCENE, d_traj_hidden: int = 128) -> None:
        super().__init__()
        self.traj_mlp = nn.Sequential(
            nn.Linear(8 * 3, d_traj_hidden), nn.ReLU(), nn.Linear(d_traj_hidden, d_traj_hidden),
        )
        self.proj = nn.Linear(d_model + d_traj_hidden, d_model)

    def forward(self, proposal_feature: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
        # proposal_feature: (B, K, D); trajectories: (B, K, 8, 3)
        t = self.traj_mlp(trajectories.flatten(-2))
        return self.proj(torch.cat([proposal_feature, t], dim=-1))  # (B, K, D)


class PredictorLayer(nn.Module):
    """Self-attn over the K action tokens + cross-attn to the scene z."""

    def __init__(self, d_model: int = D_SCENE, n_heads: int = 8,
                 d_ff: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h, _ = self.self_attn(a, a, a)
        a = self.norm1(a + self.drop(h))
        h, _ = self.cross_attn(a, z, z)
        a = self.norm2(a + self.drop(h))
        a = self.norm3(a + self.drop(self.ff(a)))
        return a


class OutcomeEncoder(nn.Module):
    """E_y: outcome features -> y in R^64. Input dim = 12 timing + 5 sub + 1 final."""

    def __init__(self, in_dim: int = 18, d_latent: int = D_LATENT,
                 hidden: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, d_latent),
        )

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.mlp(o)


class Readout(nn.Module):
    """h: R^64 -> 6 logits (NC, DAC, TTC, EP, Comfort, final)."""

    def __init__(self, d_latent: int = D_LATENT, hidden: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.GELU(), nn.Linear(hidden, N_SUB),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.mlp(y)


class EWMJEPA(nn.Module):
    """B2 EWM model (and B1 when ``direct=True``): shared encoders + trunk."""

    def __init__(self, n_layers: int = 3, d_model: int = D_SCENE,
                 d_latent: int = D_LATENT, direct: bool = False) -> None:
        super().__init__()
        self.direct = direct
        self.scene = SceneEncoder(d_model)
        self.action = ActionEncoder(d_model)
        self.layers = nn.ModuleList(
            PredictorLayer(d_model) for _ in range(n_layers)
        )
        if direct:
            self.head = nn.Sequential(
                nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, N_SUB),
            )
        else:
            self.y_head = nn.Linear(d_model, d_latent)
            self.readout = Readout(d_latent)
        self.outcome = OutcomeEncoder()          # E_y
        self.outcome_ema = OutcomeEncoder()      # EMA copy
        for p in self.outcome_ema.parameters():
            p.requires_grad_(False)
        self.outcome_ema.load_state_dict(self.outcome.state_dict())

    @torch.no_grad()
    def update_ema(self, m: float = 0.996) -> None:
        for pe, pl in zip(self.outcome_ema.parameters(), self.outcome.parameters()):
            pe.mul_(m).add_(pl, alpha=1.0 - m)
        for be, bl in zip(self.outcome_ema.buffers(), self.outcome.buffers()):
            be.copy_(bl)

    def forward_trunk(self, image_feature: torch.Tensor,
                      proposal_feature: torch.Tensor,
                      trajectories: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.scene(image_feature)                    # (B, 8, D)
        a = self.action(proposal_feature, trajectories)  # (B, K, D)
        for layer in self.layers:
            a = layer(a, z)
        return a, z

    def forward(self, image_feature: torch.Tensor, proposal_feature: torch.Tensor,
                trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
        a, z = self.forward_trunk(image_feature, proposal_feature, trajectories)
        if self.direct:
            return {"logits": self.head(a)}
        y_hat = self.y_head(a)                    # (B, K, 64)
        return {"y_hat": y_hat, "readout": self.readout(y_hat)}


def vicreg_var_cov(y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """VICReg variance+invariance-free covariance penalty over E_y outputs.

    y: (N, D). var: hinge on per-dim std; cov: off-diagonal covariance.
    """
    y = y.flatten(0, -2)                         # (N, D)
    std = torch.sqrt(y.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(gamma - std))
    y = y - y.mean(dim=0)
    cov = (y.T @ y) / (y.shape[0] - 1)
    d = cov.shape[0]
    cov_loss = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return var_loss + cov_loss / d
