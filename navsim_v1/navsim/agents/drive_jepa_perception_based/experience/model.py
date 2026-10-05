"""Offline experience module for Phase A (Stage 3) -- torch models.

Small MLP encoder over per-candidate inputs (proposal_feature + ego_status)
with an optional retrieval branch over a frozen "memory" split. The retrieval
feature builders live in experience/retrieval.py (numpy-only).

Variants (same encoder + head capacity, heads differ only by input width):
  noexp          -- head on latent only
  random         -- head on [latent, label mean/var of K random other-log
                    memory candidates]
  retrieval      -- head on [latent, softmax-weighted mean/var of cosine
                    top-K neighbour labels, mean neighbour latent]
  retrieval_int  -- retrieval + L_int aux head on the latent predicting the
                    no-att main-vehicle descriptor (training-side labels only)
  shuffle        -- eval-time control: retrieval model, shuffled memory labels
  pred_desc_*    -- deployable retrieval: a descriptor head predicts the
                    no-att main-vehicle timing descriptor from the latent;
                    retrieval then runs in descriptor space
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

N_TARGETS = 2          # [nc_unsafe, ttc_bad]
N_AUX_OUT = 1 + 4 + 3  # conflict logit, itype logits (4), reg (dt_enter, pet, min_dist)


class ExperienceEncoder(nn.Module):
    """MLP input -> latent (feat_in -> hidden -> latent)."""

    def __init__(self, in_dim: int, hidden: int = 128, latent: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, latent),
            nn.LayerNorm(latent),
        )
        self.latent_dim = latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RiskHead(nn.Module):
    """MLP head on a variable-width feature vector -> n_targets logits."""

    def __init__(self, in_dim: int, hidden: int = 64, n_targets: int = N_TARGETS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InteractionHead(nn.Module):
    """Auxiliary L_int head: latent -> [conflict(1), itype(4), reg(3)].

    Predicts the no-attribution main-vehicle interaction descriptor.
    TRAINING-SIDE auxiliary only -- never an input at deployment.
    """

    def __init__(self, latent: int = 64, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Linear(hidden, N_AUX_OUT),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DescHead(nn.Module):
    """Descriptor predictor: latent -> timing-descriptor vector (D dims).

    Deployable counterpart of the GT descriptors: memory/query descriptors
    used at retrieval time come from THIS head for pred-vs-pred variants.
    """

    def __init__(self, latent: int = 64, hidden: int = 64, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ExperienceModel(nn.Module):
    """Encoder + risk head + optional interaction / descriptor heads."""

    def __init__(self, feat_in_dim: int, head_in_dim: int, use_int: bool = False,
                 desc_dim: int = 0, hidden: int = 128, latent: int = 64,
                 head_hidden: int = 64, n_targets: int = N_TARGETS):
        super().__init__()
        self.encoder = ExperienceEncoder(feat_in_dim, hidden, latent)
        self.head = RiskHead(head_in_dim, head_hidden, n_targets)
        self.int_head = InteractionHead(latent, head_hidden) if use_int else None
        self.desc_head = DescHead(latent, head_hidden, desc_dim) if desc_dim else None

    def forward(
        self, x: torch.Tensor, exp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """x: (B, feat_in_dim) raw inputs; exp: (B, exp_dim) experience features.

        :return: (logits (B,2), latent (B,64), aux (B,8) or None,
                  desc (B,D) or None)
        """
        z = self.encoder(x)
        logits = self.head(torch.cat([z, exp], dim=-1))
        aux = self.int_head(z) if self.int_head is not None else None
        desc = self.desc_head(z) if self.desc_head is not None else None
        return logits, z, aux, desc


def encode_rows(model: ExperienceModel, x: torch.Tensor, batch: int = 4096,
                device: str = "cpu") -> np.ndarray:
    """Encode rows in chunks -> (N, latent_dim) float32 numpy."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            outs.append(model.encoder(x[i:i + batch].to(device)).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def desc_rows(model: ExperienceModel, x: torch.Tensor, batch: int = 4096,
              device: str = "cpu") -> np.ndarray:
    """Predicted descriptor vectors in chunks -> (N, desc_dim) float32."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            z = model.encoder(x[i:i + batch].to(device))
            outs.append(model.desc_head(z).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)
