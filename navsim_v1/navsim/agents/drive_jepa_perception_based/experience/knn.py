"""Dependency-light kNN / metric helpers for oracle_knn_check.py (A5).

numpy + sklearn only. All functions operate on plain float32/float64 arrays;
NaNs are mean-imputed *after* standardization (i.e. mapped to 0.0).
"""

from typing import Dict, Optional, Tuple

import numpy as np


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) per column over finite entries; std floored at eps."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize columns and map NaN -> 0 (post-standardization mean impute)."""
    z = (np.asarray(x, dtype=np.float64) - mean) / std
    return np.nan_to_num(z, nan=0.0)


def knn_predict(
    pool_x: np.ndarray,
    pool_y: np.ndarray,
    query_x: np.ndarray,
    k: int,
    pool_block: Optional[np.ndarray] = None,
    query_block: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    chunk_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """kNN regression: predict mean/var of neighbour labels per query row.

    Vectorized in chunks of `chunk_size` queries: each chunk forms a
    (chunk, N) distance matrix, same-block pool rows are masked with +inf,
    and per-row neighbour sets are extracted with argpartition + argsort.

    :param pool_x: (N, D) standardized pool features.
    :param pool_y: (N,) pool labels.
    :param query_x: (M, D) standardized query features.
    :param k: number of neighbours (capped at available pool rows).
    :param pool_block/query_block: optional group ids (e.g. log_name); pool rows
        sharing the query's group are excluded as neighbours (leakage guard).
    :param rng: if given, neighbours are sampled uniformly at random instead of
        by distance ("random memory" control; kept per-row since it is cheap).
    :return: (mean_labels (M,), var_labels (M,))
    """
    pool_x = np.asarray(pool_x, dtype=np.float64)
    query_x = np.asarray(query_x, dtype=np.float64)
    pool_y = np.asarray(pool_y, dtype=np.float64)
    m, n = query_x.shape[0], pool_x.shape[0]
    means = np.full(m, np.nan, dtype=np.float64)
    variances = np.full(m, np.nan, dtype=np.float64)

    if rng is not None:  # random-memory control: per-row sampling
        for i in range(m):
            valid = np.ones(n, dtype=bool)
            if pool_block is not None and query_block is not None:
                valid &= pool_block != query_block[i]
            idx = np.flatnonzero(valid)
            if len(idx) == 0:
                continue
            sel = rng.choice(idx, size=min(k, len(idx)), replace=False)
            labels = pool_y[sel]
            means[i] = labels.mean()
            variances[i] = labels.var()
        return means, variances

    pool_sq = (pool_x**2).sum(axis=1)  # (N,)
    k_take = min(k, n)
    for s in range(0, m, chunk_size):
        q = query_x[s : s + chunk_size]
        # squared L2 via dot products: no (c, N, D) intermediate, just (c, N)
        d = np.sqrt(
            np.clip((q**2).sum(axis=1)[:, None] + pool_sq[None, :] - 2.0 * q @ pool_x.T,
                    0.0, None)
        )
        if pool_block is not None and query_block is not None:
            d[pool_block[None, :] == query_block[s : s + q.shape[0], None]] = np.inf
        # candidate neighbours: k_take smallest distances per row
        part = np.argpartition(d, k_take - 1, axis=1)[:, :k_take]
        dpart = np.take_along_axis(d, part, axis=1)
        valid = np.isfinite(dpart)
        order = np.argsort(np.where(valid, dpart, np.inf), axis=1)
        sorted_labels = np.take_along_axis(pool_y[part], order, axis=1)
        sorted_valid = np.take_along_axis(valid, order, axis=1)
        counts = sorted_valid.sum(axis=1)
        k_per = np.minimum(k, counts)
        take = sorted_valid & (np.arange(k_take)[None, :] < k_per[:, None])
        denom = np.maximum(k_per, 1)
        row_mean = (sorted_labels * take).sum(axis=1) / denom
        row_var = np.clip((sorted_labels**2 * take).sum(axis=1) / denom - row_mean**2, 0.0, None)
        has = counts > 0
        sl = slice(s, s + q.shape[0])
        means[sl] = np.where(has, row_mean, np.nan)
        variances[sl] = np.where(has, row_var, np.nan)
    return means, variances


def top1_risk_hits(pred_risk: np.ndarray, y_true: np.ndarray, scene_ids: np.ndarray) -> np.ndarray:
    """Per scene: 1.0 if the argmax-risk predicted candidate is truly unsafe.

    :param pred_risk: (N,) predicted risk score per candidate row.
    :param y_true: (N,) binary unsafe labels.
    :param scene_ids: (N,) integer scene id per row.
    :return: (num_scenes,) float array ordered by sorted unique scene id.
    """
    scenes = np.unique(scene_ids)
    hits = np.zeros(len(scenes), dtype=np.float64)
    for i, s in enumerate(scenes):
        rows = np.flatnonzero(scene_ids == s)
        top = rows[np.argmax(pred_risk[rows])]
        hits[i] = float(y_true[top])
    return hits


def brier_score(pred: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of probability predictions (finite rows only)."""
    mask = np.isfinite(pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean((pred[mask] - y[mask]) ** 2))


def auprc(pred: np.ndarray, y: np.ndarray) -> float:
    """Average precision (AUPRC); NaN when undefined."""
    from sklearn.metrics import average_precision_score

    mask = np.isfinite(pred)
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return np.nan
    return float(average_precision_score(y[mask], pred[mask]))


def safe_logreg(x_train: np.ndarray, y_train: np.ndarray):
    """Train a logistic regression; returns None if only one class present."""
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y_train)) < 2:
        return None
    clf = LogisticRegression(max_iter=2000)
    clf.fit(x_train, y_train)
    return clf


def predict_proba_or_prior(clf, x: np.ndarray, prior: float) -> np.ndarray:
    """Predict P(y=1); fall back to the training prior if clf is None."""
    if clf is None:
        return np.full(x.shape[0], prior, dtype=np.float64)
    return clf.predict_proba(x)[:, 1]


def metric_bundle(pred: np.ndarray, y: np.ndarray, scene_ids: np.ndarray) -> Dict[str, float]:
    """Brier + AUPRC + top-1-risk hit rate for a prediction vector."""
    return {
        "brier": brier_score(pred, y),
        "auprc": auprc(pred, y),
        "top1_risk": float(np.mean(top1_risk_hits(pred, y, scene_ids))),
    }
