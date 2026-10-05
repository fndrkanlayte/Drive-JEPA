"""Experience-feature construction for the Stage-3 module (numpy only).

Given query latents and a memory bank, builds the per-candidate
experience-feature blocks consumed by the risk head:

- retrieval: cosine top-K neighbours (per-scene cap, same-log excluded) ->
  [softmax-weighted label mean (2), weighted label var (2),
  mean neighbour latent (D)]
- random: K random memory candidates from other logs -> [label mean (2),
  label var (2)]

Separated from model.py so the builders are importable/testable without
torch installed.
"""

from typing import Iterator, Optional, Tuple

import numpy as np


def _pick_neighbours(
    q_vec: np.ndarray,
    q_log: np.ndarray,
    mem_vec: np.ndarray,
    mem_scene: np.ndarray,
    mem_log: np.ndarray,
    topk: int,
    max_per_scene: int,
    chunk: int = 512,
    device: Optional[str] = None,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Yield (query_idx, picked_memory_idx, picked_cosine_sims) per row.

    Vectors are cosine-normalized inside. Same-log neighbours are excluded and
    at most ``max_per_scene`` neighbours per memory scene are kept; a wider
    candidate set is fetched first so the cap rarely starves a row.
    """
    ql = q_vec / np.clip(np.linalg.norm(q_vec, axis=1, keepdims=True), 1e-8, None)
    ml = mem_vec / np.clip(np.linalg.norm(mem_vec, axis=1, keepdims=True), 1e-8, None)
    # encode logs as ints once -- string compare in the hot loop dominates
    log_ids = {lg: i for i, lg in
               enumerate(np.unique(np.concatenate([q_log, mem_log])))}
    q_lid = np.array([log_ids[l] for l in q_log])
    mem_lid = np.array([log_ids[l] for l in mem_log])

    n = len(ql)
    fetch = min(len(ml), max(topk * max_per_scene, topk + 8))
    use_torch = device is not None
    if use_torch:
        import torch
        q_t = torch.from_numpy(ql).to(device)
        m_t = torch.from_numpy(ml).to(device)
        mlid_t = torch.from_numpy(mem_lid).to(device)
        qlid_t = torch.from_numpy(q_lid).to(device)
    for i0 in range(0, n, chunk):
        if use_torch:
            s_t = q_t[i0:i0 + chunk] @ m_t.T
            s_t.masked_fill_(
                mlid_t[None, :] == qlid_t[i0:i0 + chunk, None], -np.inf)
            sims, idx = torch.topk(s_t, fetch, dim=1)
            idx, sims = idx.cpu().numpy(), sims.cpu().numpy()
        else:
            s = ql[i0:i0 + chunk] @ ml.T  # (c, M) cosine sims
            s = np.where(mem_lid[None, :] == q_lid[i0:i0 + chunk, None],
                         -np.inf, s)
            idx = np.argpartition(-s, fetch - 1, axis=1)[:, :fetch]
            order = np.argsort(-np.take_along_axis(s, idx, 1), axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
            sims = np.take_along_axis(s, idx, 1)
        for bi in range(idx.shape[0]):
            si = sims[bi]
            picked, picked_s, counts = [], [], {}
            for jj, j in enumerate(idx[bi]):
                if not np.isfinite(si[jj]):
                    break
                sc = mem_scene[j]
                if counts.get(sc, 0) >= max_per_scene:
                    continue
                counts[sc] = counts.get(sc, 0) + 1
                picked.append(j)
                picked_s.append(si[jj])
                if len(picked) == topk:
                    break
            if picked:
                yield i0 + bi, np.asarray(picked), np.asarray(picked_s)


def retrieval_features(
    q_latent: np.ndarray,
    q_scene: np.ndarray,
    q_log: np.ndarray,
    mem_latent: np.ndarray,
    mem_labels: np.ndarray,
    mem_scene: np.ndarray,
    mem_log: np.ndarray,
    topk: int = 16,
    max_per_scene: int = 4,
    chunk: int = 512,
    rng: Optional[np.random.Generator] = None,
    shuffle_labels: bool = False,
    device: Optional[str] = None,
) -> np.ndarray:
    """Cosine top-k retrieval features per query row.

    Returns (N, 2+2+D): [weighted label mean (2), weighted label var (2),
    mean neighbour latent (D)]. Neighbours with the same log as the query are
    excluded; at most ``max_per_scene`` neighbours per memory scene. When
    ``shuffle_labels`` is set, memory labels are globally permuted (the
    shuffle control -- destroys the latent->label correspondence).
    """
    labels = mem_labels.copy()
    if shuffle_labels:
        r = rng if rng is not None else np.random.default_rng(0)
        labels = labels[r.permutation(len(labels))]

    D = mem_latent.shape[1]
    out = np.zeros((len(q_latent), 4 + D), dtype=np.float32)
    ml_norm = mem_latent / np.clip(
        np.linalg.norm(mem_latent, axis=1, keepdims=True), 1e-8, None)
    for i, picked, picked_s in _pick_neighbours(
            q_latent, q_log, mem_latent, mem_scene, mem_log,
            topk, max_per_scene, chunk, device):
        w = np.exp(picked_s - picked_s.max())
        w /= w.sum()
        lab = labels[picked]  # (k, 2)
        mean = (w[:, None] * lab).sum(0)
        var = (w[:, None] * (lab - mean) ** 2).sum(0)
        out[i, :2] = mean
        out[i, 2:4] = var
        out[i, 4:] = ml_norm[picked].mean(0)
    return out


def desc_retrieval_features(
    q_desc: np.ndarray,
    q_scene: np.ndarray,
    q_log: np.ndarray,
    mem_desc: np.ndarray,
    mem_labels: np.ndarray,
    mem_scene: np.ndarray,
    mem_log: np.ndarray,
    cont_cols: np.ndarray,
    mem_desc_valid: np.ndarray,
    topk: int = 16,
    max_per_scene: int = 4,
    chunk: int = 512,
    device: Optional[str] = None,
) -> np.ndarray:
    """Descriptor-space top-k retrieval features (pred_desc variants).

    ``q_desc``/``mem_desc`` are standardized descriptor vectors (predicted
    d_hat for the query side; TRUE or predicted for memory). Returns
    (N, 4+len(cont_cols)+1): [weighted label mean (2), weighted label var (2),
    mean(d_hat_k - d_i) over picked neighbours on the continuous cols masked
    by ``mem_desc_valid``, mean picked cosine similarity (1)].
    """
    C = len(cont_cols)
    out = np.zeros((len(q_desc), 4 + C + 1), dtype=np.float32)
    for i, picked, picked_s in _pick_neighbours(
            q_desc, q_log, mem_desc, mem_scene, mem_log,
            topk, max_per_scene, chunk, device):
        w = np.exp(picked_s - picked_s.max())
        w /= w.sum()
        lab = mem_labels[picked]
        mean = (w[:, None] * lab).sum(0)
        var = (w[:, None] * (lab - mean) ** 2).sum(0)
        out[i, :2] = mean
        out[i, 2:4] = var
        # mean predicted-vs-memory descriptor gap on valid continuous cols
        valid = mem_desc_valid[picked][:, cont_cols]  # (k, C) bool
        diff = q_desc[i, cont_cols][None, :] - mem_desc[picked][:, cont_cols]
        cnt = valid.sum(0)
        out[i, 4:4 + C] = np.where(cnt > 0,
                                   (diff * valid).sum(0) / np.maximum(cnt, 1), 0.0)
        out[i, 4 + C] = picked_s.mean()
    return out


def random_features(
    n: int,
    q_log: np.ndarray,
    mem_labels: np.ndarray,
    mem_log: np.ndarray,
    topk: int = 16,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Label mean/var (4) of ``topk`` random memory candidates from other logs."""
    r = rng if rng is not None else np.random.default_rng(0)
    # cache the exclusion pool once per distinct query log (O(|logs| * M),
    # not O(N * M) rescans of the string array)
    pools = {lg: np.flatnonzero(mem_log != lg) for lg in np.unique(q_log)}
    out = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        pool = pools[q_log[i]]
        if len(pool) == 0:
            continue
        idx = r.choice(pool, size=min(topk, len(pool)), replace=False)
        lab = mem_labels[idx]
        out[i, :2] = lab.mean(0)
        out[i, 2:] = lab.var(0)
    return out


def exp_dim_for(variant: str, latent: int, n_cont: int = 0) -> int:
    """Width of the experience-feature block appended to the latent."""
    if variant in ("noexp", "noexp_int"):
        return 0
    if variant == "random":
        return 4
    if variant.startswith("pred_desc"):
        return 4 + n_cont + 1  # label mean/var + mean desc gap + mean sim
    return 4 + latent  # retrieval / retrieval_int / shuffle
