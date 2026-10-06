"""Unit tests for knn_rerank.py (kNN-outcome reranking)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]
                     / "scripts" / "experience"))

from knn_rerank import (  # noqa: E402
    cand_feats, derange_fhat, knn_fhat, metrics, rerank_scores, topk_idx)


def _toy(S=4, K=8):
    rng = np.random.default_rng(0)
    return dict(
        yhat=rng.normal(size=(S, K, 5, 64)).astype(np.float32),
        traj=rng.normal(size=(S, K, 8, 3)).astype(np.float32),
        z_pool=rng.normal(size=(S, 256)).astype(np.float32),
        sub=rng.random((S, K, 6)).astype(np.float32),
        b0=rng.random((S, K)).astype(np.float32),
        logs=np.array([f"log{i}" for i in range(S)]))


def test_cand_feats_shapes():
    d = _toy()
    assert cand_feats(d["yhat"], d["traj"], d["z_pool"],
                      "yhat_ego").shape == (4, 8, 64)
    assert cand_feats(d["yhat"], d["traj"], d["z_pool"],
                      "yhat_mean").shape == (4, 8, 64)
    assert cand_feats(d["yhat"], d["traj"], d["z_pool"],
                      "traj_z").shape == (4, 8, 280)
    assert np.allclose(cand_feats(d["yhat"], None, None, "yhat_ego"),
                       d["yhat"][:, :, 0, :])


def test_knn_fhat_excludes_same_log():
    # bank: 2 logs x 4 cands; query log0 must never see log0 neighbours
    bf = np.eye(8, 4, dtype=np.float32)          # (8,4) distinct feats
    b_final = np.array([1., 1., 1., 1., 0., 0., 0., 0.], np.float32)
    b_log = np.array(["logA"] * 4 + ["logB"] * 4)
    qf = np.array([[[1, 0, 0, 0]]], np.float32)  # matches bank row 0 (logA)
    fh = knn_fhat(qf, bf, b_final, b_log, np.array(["logA"]),
                  n=4, t=0.05, device="cpu")
    # all logA excluded -> neighbours all have final 0
    assert fh[0, 0] < 0.05


def test_knn_fhat_distance_weighting():
    bf = np.array([[0.], [1.], [10.]], np.float32)
    b_final = np.array([1., 0.5, 0.], np.float32)
    b_log = np.array(["L1", "L1", "L1"])
    qf = np.array([[[0.05]]], np.float32)
    fh_hot = knn_fhat(qf, bf, b_final, b_log, np.array(["Q"]),
                      n=3, t=0.01, device="cpu")
    fh_cold = knn_fhat(qf, bf, b_final, b_log, np.array(["Q"]),
                       n=3, t=1e6, device="cpu")
    # hot temperature -> nearest dominates; cold -> uniform mean 0.5
    assert fh_hot[0, 0] > 0.9
    assert abs(fh_cold[0, 0] - 0.5) < 0.05


def test_rerank_scores_masks_outside_topk():
    b0 = np.array([[0.9, 0.8, 0.7, 0.6]], np.float32)
    tk = topk_idx(b0, 2)                         # candidates 0,1
    fh = np.array([[0.0, 0.0]], np.float32)
    sc = rerank_scores(b0, fh, tk, beta=4.0)
    assert sc.shape == (1, 4)
    assert sc[0, 2] == -1e9 and sc[0, 3] == -1e9
    assert sc.argmax(1)[0] in (0, 1)


def test_rerank_beta0_is_b0():
    rng = np.random.default_rng(1)
    b0 = rng.random((16, 32)).astype(np.float32)
    tk = topk_idx(b0, 8)
    fh = rng.normal(size=(16, 8)).astype(np.float32)
    sc = rerank_scores(b0, fh, tk, beta=0.0)
    assert (sc.argmax(1) == b0.argmax(1)).all()


def test_derange_fhat_crosses_logs_only():
    fh = np.arange(20, dtype=np.float32).reshape(10, 2)
    logs = np.array(["A", "A", "B", "B", "C", "C", "A", "B", "C", "D"])
    out = derange_fhat(fh, logs, seed=0)
    moved = (out != fh).any(1)
    assert moved.any()
    kept = np.flatnonzero(~moved)
    # every kept row unchanged; every changed row came from a different log
    assert np.all(out[kept] == fh[kept])


def test_metrics_wrong_scene():
    final = np.array([[0.5, 0.9]], np.float32)   # b0 pick 0 but cand1 better
    b0 = np.array([[0.9, 0.1]], np.float32)      # b0 picks 0 (wrong)
    m = metrics(final, b0, np.array([1]))        # rerank picks 1: fixed
    assert m["net_fix"] > 0.3 and m["flip"] == 1.0
    m0 = metrics(final, b0, np.array([0]))       # keep b0 pick: nothing
    assert m0["net_fix"] == 0.0 and m0["val"] == 0.5
