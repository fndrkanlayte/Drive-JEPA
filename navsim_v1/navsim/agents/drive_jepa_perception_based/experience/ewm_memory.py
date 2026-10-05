"""Step 4 (spec v0): outcome-labelled episodic memory on top of frozen B0+b3.

Frozen pieces (no grads):
    - Drive-JEPA latents: image_feature, proposal_feature, proposals, pdm_score
    - b3 (EWMStructured) full s*: forward -> y_hat (B,32,S,64), forward_trunk
      -> (a, z); encode_outcome(..., ema=True) -> y_t (B,32,S,64)

Memory bank M (offline, navtrain):
    per scene i:  key_src_i = [mean-pool(image_feature), mean-pool(proposal_feature)]  (512,)
                  z_pool_i  (256,), log
    per cand j:   y_hat_ego (64,), y_t (5,64), sub (6,), b0 (), traj (8,3)
Retrieval always excludes same-log entries.

Retrieval key g (H3): 2-layer MLP key_src -> 128, L2-normed.
Hierarchical retrieval: appearance part (first 256 dims) -> top-3k scenes,
then cosine on g keys -> top-k (k=8) scenes.

Delta correction: score_k = logit(B0_k) + Delta_k; Delta=0 without memory.

o(x) (for L_ret target, action-aligned approximation):
    k-means K=64 over navtrain proposals flattened (8 poses x,y -> 16 dims)
    gives proposal_vocab_64.npy. o(x)_c = true 6 subscores of the candidate
    nearest to centre c (masked when distance > 75th percentile of all
    nearest distances). d(o_a,o_b) = mean L2 over rows valid on both sides.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_SCENE = 256          # image/proposal feature width
D_KEY = 512            # key_src width = 2*D_SCENE
D_LAT = 64             # b3 latent width
N_SLOTS = 5            # b3 slot count (ego + 4 agent slots)
N_SUB = 6              # subscores [NC,DAC,EP,TTC,C,final]
K_RETR = 8             # retrieved neighbour scenes
TOPK_STAGE1 = 3000     # stage-1 appearance shortlist


# --------------------------------------------------------------------------
# learned pieces
# --------------------------------------------------------------------------

class KeyNet(nn.Module):
    """g: key_src (512) -> 128, L2-normalised output."""

    def __init__(self, in_dim: int = D_KEY, hid: int = 256, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU(),
                                 nn.Linear(hid, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1)


class TrajEnc(nn.Module):
    """Enc(tau): flattened trajectory 8x3=24 -> 64."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(24, 128), nn.GELU(),
                                 nn.Linear(128, out_dim))

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        # traj (..., 8, 3)
        return self.mlp(traj.flatten(-2))


class MemTokenEnc(nn.Module):
    """m_ij = MLP([z_pool(256), Enc(tau)(64), yhat flat(320), y_t flat(320),
    s(6), b0(1)]) -> 256.  yhat = b3's prediction for that memory candidate,
    y_t = actual encoded outcome: the 'predicted vs actual' contrast.
    no_latent=True drops BOTH latent blocks (appearance+traj+scores only)."""

    def __init__(self, d: int = D_SCENE, no_latent: bool = False):
        super().__init__()
        self.no_latent = no_latent
        self.traj = TrajEnc()
        in_dim = D_SCENE + 64 + N_SUB + 1 + (0 if no_latent else 2 * N_SLOTS * D_LAT)
        self.proj = nn.Sequential(nn.Linear(in_dim, d),
                                  nn.GELU(), nn.Linear(d, d))

    def forward(self, z_pool: torch.Tensor, traj: torch.Tensor,
                y_t: torch.Tensor, yhat: torch.Tensor, s: torch.Tensor,
                b0: torch.Tensor) -> torch.Tensor:
        """z_pool (...,256); traj (...,8,3); y_t (...,5,64);
        yhat (...,5,64); s (...,6); b0 (...,)"""
        t = self.traj(traj)
        parts = [z_pool, t]
        if not self.no_latent:
            parts += [yhat.flatten(-2), y_t.flatten(-2)]
        parts += [s, b0.unsqueeze(-1)]
        return self.proj(torch.cat(parts, dim=-1))


class MemoryDelta(nn.Module):
    """h_k = [a_k(256), y_hat_k flat(320)] -> q_proj; 2 XAttn layers over
    k*32 memory tokens -> scalar Delta_k (zero-init output head).
    no_latent=True queries with a_k alone (yhat ignored).
    readout="resid": h accumulates residual ctx (baseline residual scorer).
    readout="mem":   layer-1 ctx1 = attn(h, mem, mem); layer-2 queries
                     norm(h+ctx1) but Delta = out(ctx2) — memory content only."""

    def __init__(self, d: int = D_SCENE, n_heads: int = 4,
                 no_latent: bool = False, readout: str = "resid"):
        super().__init__()
        self.no_latent = no_latent
        self.readout = readout
        self.q_proj = nn.Linear(D_SCENE + (0 if no_latent else N_SLOTS * D_LAT),
                                d)
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(d, n_heads, batch_first=True)
            for _ in range(2))
        self.norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(2))
        self.out = nn.Linear(d, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, a: torch.Tensor, yhat_flat: torch.Tensor,
                mem: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """a (B,K,256); yhat_flat (B,K,320); mem (B,M,256);
        pad_mask (B,M) bool, True = ignore that memory token.
        Rows whose memory is fully masked get Delta = 0."""
        if self.no_latent:
            h = self.q_proj(a)
        else:
            h = self.q_proj(torch.cat([a, yhat_flat], dim=-1))
        fully = None
        if pad_mask is not None:
            pad_mask = pad_mask.clone()
            fully = pad_mask.all(1)
            pad_mask[fully, 0] = False        # keep one token to avoid NaN
        if self.readout == "mem":
            # Delta reads ONLY attention-retrieved memory content;
            # h enters the second pass only as the attention query
            ctx1, _ = self.attn[0](h, mem, mem,
                                   key_padding_mask=pad_mask)
            q2 = self.norm[0](h + ctx1)
            ctx2, _ = self.attn[1](q2, mem, mem,
                                   key_padding_mask=pad_mask)
            d = self.out(ctx2).squeeze(-1)
        else:
            for attn, norm in zip(self.attn, self.norm):
                ctx, _ = attn(h, mem, mem, key_padding_mask=pad_mask)
                h = norm(h + ctx)
            d = self.out(h).squeeze(-1)
        if fully is not None:
            d = d * (~fully).float().unsqueeze(-1)
        return d


def score_with_delta(b0: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """score = logit(B0) + Delta. b0 in (0,1)."""
    b0 = b0.clamp(1e-6, 1 - 1e-6)
    return torch.logit(b0) + delta


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------

def listwise_ce(scores: torch.Tensor, final: torch.Tensor,
                temp: float = 0.05, weight: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
    """CE of softmax(scores) vs soft target softmax(true_final/temp)."""
    tgt = F.softmax(final / temp, dim=-1)
    ll = -(tgt * F.log_softmax(scores, dim=-1)).sum(-1)
    if weight is not None:
        ll = ll * weight
    return ll.mean()


def pairwise_hinge(scores: torch.Tensor, final: torch.Tensor,
                   gap: float = 0.2, margin: float = 0.1,
                   weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Hinge on same-scene pairs with |Delta final| > gap."""
    B, K = final.shape
    loss = scores.new_zeros(())
    cnt = 0
    for b in range(B):
        f, s = final[b], scores[b]
        hi = f[None, :] - f[:, None] > gap          # (K,K) better row = i
        if not hi.any():
            continue
        diffs = s[:, None] - s[None, :]             # want s_i > s_j when f_i>f_j
        l = F.relu(margin - diffs[hi]).mean()
        loss = loss + l * (1.0 if weight is None else weight[b])
        cnt += 1
    return loss / max(cnt, 1)


def retrieval_kl(g: KeyNet, key_src: torch.Tensor, o: torch.Tensor,
                 o_mask: torch.Tensor, log_id: torch.Tensor,
                 t: float = 0.1, T: float = 1.0) -> torch.Tensor:
    """L_ret = KL(softmax(w_ab) || softmax(cos(g_a,g_b)/t)) with in-batch
    negatives, cross-log pairs only. o (B,C,6); o_mask (B,C) bool."""
    B = key_src.shape[0]
    keys = g(key_src)                                   # (B,128)
    # target weights w_ab = exp(-d(o_a,o_b)/T) on jointly-valid rows
    both = o_mask[:, None, :] & o_mask[None, :, :]      # (B,B,C)
    d = ((o[:, None, :, :] - o[None, :, :, :]) ** 2).sum(-1)
    n_common = both.sum(-1).float()                     # (B,B)
    d = (d * both).sum(-1) / n_common.clamp_min(1.0)
    valid = (n_common >= 8) & (log_id[:, None] != log_id[None, :])
    w = torch.exp(-d / T)
    w = w * valid
    # pred distribution
    cos = keys @ keys.T / t
    cos = cos.masked_fill(~valid | torch.eye(B, dtype=torch.bool,
                                             device=cos.device), -1e9)
    tgt = w / w.sum(1, keepdim=True).clamp_min(1e-9)
    logp = F.log_softmax(cos, dim=-1)
    keep = w.sum(1) > 0
    if keep.sum() == 0:
        return keys.new_zeros(())
    return F.kl_div(logp[keep], tgt[keep], reduction="batchmean")


# --------------------------------------------------------------------------
# numpy-side memory bank + retrieval
# --------------------------------------------------------------------------

class MemoryBank:
    """Scene-level memory bank over navtrain scenes.

    Arrays:
      key_src (S,512)   z_pool (S,256)   log (S,) str
      yt_flat (S,K,320) sub (S,K,6)      b0 (S,K,)   traj (S,K,8,3)
    """

    def __init__(self, key_src, z_pool, logs, yt_flat, sub, b0, traj,
                 g_keys=None, yhat_flat=None):
        self.key_src = key_src.astype(np.float32)
        self.z_pool = z_pool.astype(np.float32)
        self.logs = np.asarray(logs)
        self.yt_flat = yt_flat.astype(np.float32)
        self.sub = sub.astype(np.float32)
        self.b0 = b0.astype(np.float32)
        self.traj = traj.astype(np.float32)
        self.g_keys = g_keys.astype(np.float32) if g_keys is not None else None
        # (S,K,320) b3 predictions for each bank candidate; falls back to
        # zeros when absent (e.g. a bank built without them)
        self.yhat_flat = (yhat_flat.astype(np.float32)
                          if yhat_flat is not None
                          else np.zeros((len(logs), self.sub.shape[1],
                                         N_SLOTS * D_LAT), np.float32))
        self.app = self.key_src[:, :D_SCENE]            # appearance part
        self.app_n = self.app / (np.linalg.norm(self.app, axis=1,
                                                keepdims=True) + 1e-8)

    def stage1(self, q_key_src: np.ndarray) -> np.ndarray:
        """appearance cosine -> top-TOPK_STAGE1 scene indices (S_q,3k)."""
        q = q_key_src[:, :D_SCENE]
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        sim = qn @ self.app_n.T
        k = min(TOPK_STAGE1, sim.shape[1])
        return np.argpartition(-sim, k - 1, axis=1)[:, :k]

    def stage2(self, q_g: np.ndarray, cand_idx: np.ndarray,
               q_logs: np.ndarray, k: int = K_RETR) -> np.ndarray:
        """cosine on g keys within stage-1 shortlist -> (S_q,k), same-log out."""
        assert self.g_keys is not None
        out = np.full((len(cand_idx), k), -1, np.int64)
        bk = self.g_keys / (np.linalg.norm(self.g_keys, axis=1,
                                           keepdims=True) + 1e-8)
        for i in range(len(cand_idx)):
            cand = cand_idx[i]
            cand = cand[self.logs[cand] != q_logs[i]]
            if len(cand) == 0:
                continue
            sim = bk[cand] @ q_g[i]
            top = cand[np.argsort(-sim)[:k]]
            out[i, : len(top)] = top
        return out

    def retrieve_appearance(self, q_key_src: np.ndarray,
                            q_logs: np.ndarray, k: int = K_RETR) -> np.ndarray:
        """control key (a): plain appearance cosine, no learned g."""
        cand = self.stage1(q_key_src)
        out = np.full((len(cand), k), -1, np.int64)
        qn = q_key_src[:, :D_SCENE] / (
            np.linalg.norm(q_key_src[:, :D_SCENE], axis=1,
                           keepdims=True) + 1e-8)
        sim = qn @ self.app_n.T
        for i in range(len(cand)):
            row = sim[i].copy()
            row[self.logs == q_logs[i]] = -np.inf
            top = np.argsort(-row)
            top = top[np.isfinite(row[top])][:k]
            out[i, : len(top)] = top
        return out

    def retrieve_random(self, q_logs: np.ndarray, k: int = K_RETR,
                        seed: int = 0) -> np.ndarray:
        """control key (b): random scenes, same-log excluded."""
        rng = np.random.default_rng(seed)
        out = np.full((len(q_logs), k), -1, np.int64)
        for i in range(len(q_logs)):
            pool = np.where(self.logs != q_logs[i])[0]
            if len(pool) == 0:
                continue
            n = min(k, len(pool))
            out[i, :n] = rng.choice(pool, n, replace=False)
        return out

    def gather(self, scene_idx: np.ndarray) -> Dict[str, np.ndarray]:
        """(B,k) scene idx -> stacked memory arrays (B,k,K,...); -1 -> zeros."""
        B, kk = scene_idx.shape
        ok = scene_idx >= 0
        idx = np.clip(scene_idx, 0, len(self.logs) - 1)
        g = dict(
            z_pool=self.z_pool[idx],                    # (B,k,256)
            yt_flat=self.yt_flat[idx],                  # (B,k,K,320)
            yhat_flat=self.yhat_flat[idx],              # (B,k,K,320)
            sub=self.sub[idx],                          # (B,k,K,6)
            b0=self.b0[idx],                            # (B,k,K)
            traj=self.traj[idx],                        # (B,k,K,8,3)
        )
        g["valid"] = ok                                 # (B,k)
        return g


def make_memory_tokens(enc: MemTokenEnc, gathered: Dict[str, np.ndarray],
                       device) -> torch.Tensor:
    """(B,k,K,*) gathers -> (B, k*K, 256) memory tokens."""
    B, k = gathered["z_pool"].shape[:2]
    K = gathered["sub"].shape[2]
    z = torch.from_numpy(gathered["z_pool"]).to(device)
    z = z[:, :, None, :].expand(B, k, K, D_SCENE)
    t = torch.from_numpy(gathered["traj"]).to(device)
    yt = torch.from_numpy(gathered["yt_flat"]).to(device)
    yt = yt.reshape(B, k, K, N_SLOTS, D_LAT)
    yh = torch.from_numpy(gathered["yhat_flat"]).to(device)
    yh = yh.reshape(B, k, K, N_SLOTS, D_LAT)
    s = torch.from_numpy(gathered["sub"]).to(device)
    b0 = torch.from_numpy(gathered["b0"]).to(device)
    tok = enc(z, t, yt, yh, s, b0)                      # (B,k,K,256)
    return tok.reshape(B, k * K, -1)


# --------------------------------------------------------------------------
# action-aligned o(x) builder (approximation for L_ret targets)
# --------------------------------------------------------------------------

def build_proposal_vocab(proposals: np.ndarray, n_clusters: int = 64,
                         seed: int = 0, iters: int = 50) -> np.ndarray:
    """k-means over flattened proposal poses (x,y) -> (n_clusters,16).
    Proposals are already in ego frame: no centring anywhere."""
    X = np.asarray(proposals[:, :, :2], dtype=np.float32).reshape(
        len(proposals), -1)
    rng = np.random.default_rng(seed)
    ctr = X[rng.choice(len(X), n_clusters, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - ctr[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for c in range(n_clusters):
            if (lab == c).any():
                ctr[c] = X[lab == c].mean(0)
    return ctr


def scene_o(proposals: np.ndarray, sub: np.ndarray,
            vocab: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    """o(x): (C,6) + valid mask (C,) for one scene.

    For each cluster centre c, fill subscores of the scene candidate nearest
    to c; mask the row if that distance exceeds `thresh`."""
    X = proposals[:, :, :2].reshape(len(proposals), -1)
    d = ((X[:, None, :] - vocab[None]) ** 2).sum(-1)    # (K,C)
    nn_idx = d.argmin(0)                                # (C,)
    nn_d = np.sqrt(d[nn_idx, np.arange(len(vocab))])
    o = sub[nn_idx]                                     # (C,6)
    mask = nn_d <= thresh
    return o, mask


def derange(idx2d: np.ndarray, logs: np.ndarray,
            rng: np.random.Generator = None) -> np.ndarray:
    """Assign every query another query's neighbour list: random
    permutation with perm[i] != i and logs[perm[i]] != logs[i] (a plain
    roll would hand each scene its adjacent frames' near-identical
    neighbours and the 'wrong memory' control would be a no-op).

    Rows that cannot find a different-log partner keep their own list.
    """
    if rng is None:
        rng = np.random.default_rng()
    logs = np.asarray(logs)
    n = len(idx2d)
    out = np.asarray(idx2d).copy()
    perm = np.arange(n)
    if n < 2:
        return out
    # candidate permutation: random derangement, retry to satisfy
    # cross-log + no-fixed-point constraints
    for _ in range(100):
        p = rng.permutation(n)
        ok = (p != np.arange(n)) & (logs[p] != logs)
        if ok.all():
            perm = p
            break
    else:
        p = perm
        ok = np.zeros(n, bool)
    # rows still bad: swap with a random different-log partner
    bad = np.where(~((p != np.arange(n)) & (logs[p] != logs)))[0]
    for i in bad:
        cands = np.where((np.arange(n) != i) & (logs != logs[i]))[0]
        cands = cands[cands != p[i]]
        if len(cands):
            j = cands[rng.integers(len(cands))]
            p[i], p[j] = p[j], p[i]
    good = (p != np.arange(n)) & (logs[p] != logs)
    perm[good] = p[good]
    return out[perm]
