"""Synthetic-array tests for experience/knn.py."""

import numpy as np
import pytest

from navsim.agents.drive_jepa_perception_based.experience.knn import (
    auprc,
    brier_score,
    knn_predict,
    metric_bundle,
    standardize_apply,
    standardize_fit,
    top1_risk_hits,
)


class TestStandardize:
    def test_fit_apply(self):
        x = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        m, s = standardize_fit(x)
        z = standardize_apply(x, m, s)
        assert np.allclose(z.mean(axis=0), 0, atol=1e-9)
        assert np.allclose(z.std(axis=0), 1, atol=1e-9)

    def test_nan_imputed_to_zero(self):
        x = np.array([[np.nan, 5.0], [2.0, 7.0]])
        m, s = standardize_fit(x)
        z = standardize_apply(x, m, s)
        assert z[0, 0] == 0.0
        assert np.isfinite(z).all()

    def test_zero_variance_column(self):
        x = np.array([[1.0, 4.0], [1.0, 8.0]])
        m, s = standardize_fit(x)
        assert s[0] == 1.0  # floored, no div-by-zero
        z = standardize_apply(x, m, s)
        assert np.isfinite(z).all()


class TestKnnPredict:
    def test_retrieves_nearest_labels(self):
        pool_x = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
        pool_y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        mean, var = knn_predict(pool_x, pool_y, np.array([[0.05], [10.15]]), k=3)
        assert np.isclose(mean[0], 0.0)
        assert np.isclose(mean[1], 1.0)
        assert var[0] == pytest.approx(0.0)

    def test_block_exclusion(self):
        pool_x = np.array([[0.0], [0.05], [5.0]])
        pool_y = np.array([1.0, 1.0, 0.0])
        pool_block = np.array(["A", "A", "B"], dtype=object)
        mean, _ = knn_predict(
            pool_x, pool_y, np.array([[0.0]]), k=2,
            pool_block=pool_block, query_block=np.array(["A"], dtype=object),
        )
        # same-log pool rows excluded -> only the "B" row remains
        assert mean[0] == pytest.approx(0.0)

    def test_random_neighbours(self):
        rng = np.random.default_rng(0)
        pool_x = np.arange(20, dtype=float).reshape(-1, 1)
        pool_y = (np.arange(20) % 2).astype(float)
        mean, var = knn_predict(pool_x, pool_y, np.array([[0.0]]), k=4, rng=rng)
        assert np.isfinite(mean[0])
        # random neighbours over alternating labels -> var>0 likely, mean in [0,1]
        assert 0.0 <= mean[0] <= 1.0


class TestMetrics:
    def test_top1(self):
        scene = np.array([0, 0, 0, 1, 1])
        y = np.array([0, 0, 1, 0, 0])
        pred = np.array([0.9, 0.5, 0.1, 0.8, 0.2])
        hits = top1_risk_hits(pred, y, scene)
        # scene0: argmax pred = idx0 (safe) -> 0; scene1: idx3 (safe) -> 0
        assert hits.tolist() == [0.0, 0.0]
        pred2 = np.array([0.1, 0.5, 0.9, 0.8, 0.2])
        hits2 = top1_risk_hits(pred2, y, scene)
        assert hits2[0] == 1.0  # argmax is idx2 which is unsafe

    def test_brier_and_auprc(self):
        y = np.array([0, 0, 1, 1], dtype=float)
        pred = np.array([0.1, 0.2, 0.9, 0.8])
        assert brier_score(pred, y) < 0.05
        assert auprc(pred, y) == pytest.approx(1.0)
        assert np.isnan(auprc(pred, np.zeros(4)))

    def test_metric_bundle(self):
        scene = np.array([0, 0, 1, 1])
        y = np.array([0, 1, 0, 1], dtype=float)
        pred = np.array([0.1, 0.9, 0.2, 0.8])
        out = metric_bundle(pred, y, scene)
        assert set(out) == {"brier", "auprc", "top1_risk"}
        assert out["top1_risk"] == pytest.approx(1.0)
