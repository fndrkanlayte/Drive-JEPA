"""Shared, dependency-light helpers for the Phase-A experience tooling.

- record IO (npz) used by export_candidates.py / label_candidates.py
- log-level scene splitting used by oracle_knn_check.py
- scene-level bootstrap CIs used by headroom.py / oracle_knn_check.py

Pure numpy only (pandas avoided so the helpers stay testable on a bare env).
"""

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Record IO
# --------------------------------------------------------------------------- #

def save_npz(path: Path, **arrays) -> None:
    """Atomically save a compressed npz (write-then-rename for resume safety)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load an npz into a plain dict of ndarrays (str arrays stay object/str)."""
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def npz_scalar(arr: np.ndarray):
    """Extract a python scalar from a 0-d/1-elem npz array."""
    return arr.item() if hasattr(arr, "item") else arr


# --------------------------------------------------------------------------- #
# Log-level splitting (oracle check)
# --------------------------------------------------------------------------- #

def split_logs_by_name(
    log_names: Sequence[str],
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> Dict[str, List[str]]:
    """Split unique log names into memory / query_train / query_val groups.

    Guaranteed: disjoint groups covering every log; deterministic given seed.
    """
    logs = sorted(set(str(l) for l in log_names))
    r = np.asarray(ratios, dtype=np.float64)
    assert r.ndim == 1 and len(r) == 3 and (r > 0).all(), "ratios must be 3 positive numbers"
    r = r / r.sum()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(logs))
    n = len(logs)
    n_mem = int(round(n * r[0]))
    n_qtr = int(round(n * r[1]))
    # query_val gets the remainder so coverage is exact
    idx_mem = order[:n_mem]
    idx_qtr = order[n_mem:n_mem + n_qtr]
    idx_qva = order[n_mem + n_qtr:]

    out = {
        "memory": [logs[i] for i in sorted(idx_mem)],
        "query_train": [logs[i] for i in sorted(idx_qtr)],
        "query_val": [logs[i] for i in sorted(idx_qva)],
    }
    # hard guarantee of the task spec
    s_mem, s_qtr, s_qva = set(out["memory"]), set(out["query_train"]), set(out["query_val"])
    assert not (s_mem & s_qtr) and not (s_mem & s_qva) and not (s_qtr & s_qva)
    assert s_mem | s_qtr | s_qva == set(logs)
    return out


# --------------------------------------------------------------------------- #
# Bootstrap CIs (scene-level)
# --------------------------------------------------------------------------- #

def bootstrap_scene_ci(
    scene_values: np.ndarray,
    num_boot: int = 1000,
    seed: int = 0,
    agg=np.nanmean,
) -> Tuple[float, float, float]:
    """Scene-level bootstrap 95% CI of ``agg`` over per-scene values.

    :param scene_values: (num_scenes, ...) array; ``agg`` is applied to the
        resampled array with ``axis=0`` semantics of numpy (we call agg on the
        resampled 1-D or N-D scene array).
    :return: (point_estimate, lo, hi)
    """
    scene_values = np.asarray(scene_values)
    n = len(scene_values)
    point = float(agg(scene_values))
    if n == 0:
        return point, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(num_boot, dtype=np.float64)
    for b in range(num_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[b] = agg(scene_values[idx])
    lo, hi = np.nanpercentile(boot_stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def fmt_ci(point: float, lo: float, hi: float, digits: int = 4) -> str:
    """'0.123 [0.100, 0.145]' or '0.123' if CI undefined."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #

def iter_export_records(export_dir: Path, tokens: Iterable[str] = None) -> List[Path]:
    """List exported scene records (per-token .npz) in an export directory."""
    export_dir = Path(export_dir)
    if tokens is None:
        return sorted(export_dir.glob("*.npz"))
    return [export_dir / f"{t}.npz" for t in tokens if (export_dir / f"{t}.npz").is_file()]


def ade_to_human(proposals: np.ndarray, human_traj: np.ndarray) -> np.ndarray:
    """Mean L2 between candidate xy and human GT xy over the poses.

    :param proposals: (K, T, >=2) candidate poses in ego frame.
    :param human_traj: (T, >=2) human GT poses in ego frame.
    :return: (K,) mean Euclidean distance over xy.
    """
    proposals = np.asarray(proposals, dtype=np.float64)
    human_traj = np.asarray(human_traj, dtype=np.float64)
    return np.linalg.norm(proposals[..., :2] - human_traj[None, :, :2], axis=-1).mean(axis=-1)
