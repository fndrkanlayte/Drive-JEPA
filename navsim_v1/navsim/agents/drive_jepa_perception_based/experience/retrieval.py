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

from typing import Optional

import numpy as np


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
) -> np.ndarray:
    """Cosine top-k retrieval features per query row.

    Returns (N, 2+2+D): [weighted label mean (2), weighted label var (2),
    mean neighbour latent (D)]. Neighbours with the same log as the query are
    excluded; at most ``max_per_scene`` neighbours per memory scene. When
    ``shuffle_labels`` is set, memory labels are globally permuted (the
    shuffle control -- destroys the latent->label correspondence).
    """
    ql = q_latent / np.clip(np.linalg.norm(q_latent, axis=1, keepdims=True), 1e-8, None)
    ml = mem_latent / np.clip(np.linalg.norm(mem_latent, axis=1, keepdims=True), 1e-8, None)
    labels = mem_labels.copy()
    if shuffle_labels:
        r = rng if rng is not None else np.random.default_rng(0)
        labels = labels[r.permutation(len(labels))]

    D = ml.shape[1]
    n = len(ql)
    out = np.zeros((n, 4 + D), dtype=np.float32)
    # fetch a wider candidate set, then enforce the per-scene cap
    fetch = min(len(ml), max(topk * max_per_scene, topk + 8))
    for i0 in range(0, n, chunk):
        sims = ql[i0:i0 + chunk] @ ml.T  # (c, M)
        for bi in range(sims.shape[0]):
            i = i0 + bi
            s = sims[bi]
            s = np.where(mem_log == q_log[i], -np.inf, s)
            idx = np.argpartition(-s, fetch - 1)[:fetch]
            idx = idx[np.argsort(-s[idx])]
            picked, counts = [], {}
            for j in idx:
                if not np.isfinite(s[j]):
                    break
                sc = mem_scene[j]
                if counts.get(sc, 0) >= max_per_scene:
                    continue
                counts[sc] = counts.get(sc, 0) + 1
                picked.append(j)
                if len(picked) == topk:
                    break
            if not picked:
                continue
            picked = np.asarray(picked)
            w = np.exp(s[picked] - s[picked].max())
            w /= w.sum()
            lab = labels[picked]  # (k, 2)
            mean = (w[:, None] * lab).sum(0)
            var = (w[:, None] * (lab - mean) ** 2).sum(0)
            out[i, :2] = mean
            out[i, 2:4] = var
            out[i, 4:] = ml[picked].mean(0)
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


def exp_dim_for(variant: str, latent: int) -> int:
    """Width of the experience-feature block appended to the latent."""
    if variant == "noexp":
        return 0
    if variant == "random":
        return 4
    return 4 + latent  # retrieval / retrieval_int / shuffle
