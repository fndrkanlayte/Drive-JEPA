"""Step 4 (CODE SKETCH ONLY -- not run until Step 3 gate passes).

Memory-augmented EWM-JEPA variants:

- B2+in-context : retrieved memory tokens are cross-attended by the predictor
                  (hierarchical retrieval: map slot first, then agent slot).
- B2+L-TTT      : per-scene LoRA (r=8) on predictor+readout, 3 AdamW steps
                  lr 2e-4 on retrieved neighbours, M0-style cosine trigger,
                  re-initialised and discarded per scene.
- B2+R-TTT      : same procedure with k RANDOM train scenes (control).

Memory entry per candidate j of scene i (train logs only, never same log):
    m_ij = (z_map_i, z_agent_i, a_ij, y_ij, is_failure_ij)
where z_map/z_agent are the tagged scene queries (MAP_QUERY_IDX /
AGENT_QUERY_IDX), a_ij the action embedding, y_ij the E_y outcome latent,
is_failure = final < threshold. Both failures AND successes are kept.

Hierarchical retrieval for query scene q:
    1) rank train scenes by cosine(q.z_map, m.z_map)   -> keep top-K1
    2) re-rank those by cosine(q.z_agent, m.z_agent)   -> keep top-K2
    3) take their candidate entries -> memory token set M_q
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ewm import EWMJEPA, D_LATENT, MAP_QUERY_IDX, AGENT_QUERY_IDX

D_SCENE = 256


class MemoryBank:
    """Holds per-candidate memory entries from TRAIN scenes only.

    entries: dict with stacked tensors
      z_map   (M, D_SCENE)   z_agent (M, D_SCENE)
      a       (M, D_SCENE)   y       (M, D_LATENT)
      final   (M,)           log_id  (M,) int
    """

    def __init__(self):
        self.buf = {k: [] for k in
                    ("z_map", "z_agent", "a", "y", "final", "log_id")}

    def add_scene(self, z, a, y, final, log_id):
        """z (8,D_SCENE), a (K,D_SCENE), y (K,D_LATENT), final (K,)."""
        k = a.shape[0]
        self.buf["z_map"].append(z[MAP_QUERY_IDX].expand(k, -1))
        self.buf["z_agent"].append(z[AGENT_QUERY_IDX].expand(k, -1))
        self.buf["a"].append(a)
        self.buf["y"].append(y)
        self.buf["final"].append(final)
        self.buf["log_id"].append(torch.full((k,), log_id))

    def finalize(self, device):
        self.e = {k: torch.cat(v).to(device) for k, v in self.buf.items()}

    def retrieve(self, z_map, z_agent, log_id, k1=32, k2=8):
        """Hierarchical: map-slot scene match -> agent-slot re-rank.
        Returns memory tokens (k2_candidates,) indices into self.e,
        excluding same log."""
        cand = torch.nonzero(self.e["log_id"] != log_id).squeeze(-1)
        sim_map = F.cosine_similarity(
            z_map.unsqueeze(0), self.e["z_map"][cand], dim=-1)
        top1 = cand[sim_map.topk(min(k1, len(cand))).indices]
        sim_ag = F.cosine_similarity(
            z_agent.unsqueeze(0), self.e["z_agent"][top1], dim=-1)
        return top1[sim_ag.topk(min(k2, len(top1))).indices]


class InContextPredictor(nn.Module):
    """B2 predictor + cross-attention over retrieved memory tokens.

    Memory token i for candidate k is built as [a_i | y_i] projected to
    D_SCENE; the action query a_k cross-attends scene z AND memory M_q.
    """

    def __init__(self, ewm: EWMJEPA):
        super().__init__()
        self.mem_proj = nn.Linear(D_SCENE + D_LATENT, D_SCENE)
        self.mem_attn = nn.MultiheadAttention(D_SCENE, 4, batch_first=True)
        self.mem_norm = nn.LayerNorm(D_SCENE)

    def forward(self, a, mem_a, mem_y):
        """a (K,D_SCENE) queries; mem_a/mem_y (M,*). -> a' (K,D_SCENE)."""
        m = self.mem_proj(torch.cat([mem_a, mem_y], dim=-1)).unsqueeze(0)
        q = a.unsqueeze(0)
        ctx, _ = self.mem_attn(q, m, m)
        return self.mem_norm(q + ctx).squeeze(0)


class LoRA(nn.Module):
    """LoRA wrapper for Linear layers (r=8)."""

    def __init__(self, linear: nn.Linear, r=8):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(linear.in_features, r, bias=False)
        self.B = nn.Linear(r, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.linear(x) + self.B(self.A(x))


def lora_ttt_step(ewm: EWMJEPA, neigh_labels, neigh_a, neigh_z,
                  steps=3, lr=2e-4):
    """B2+L-TTT sketch: attach LoRA to predictor self-attn + readout,
    run `steps` AdamW BCE steps on the retrieved neighbours, score the
    query, then DISCARD the LoRA (weights re-init per scene).
    Not implemented for running -- signature only."""
    raise NotImplementedError("Step 4 sketch -- run only after Step 3 gate")
