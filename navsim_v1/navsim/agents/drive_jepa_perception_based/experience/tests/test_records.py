"""Unit tests for shared helpers: splitting, bootstrap CI, npz IO, ADE."""

import numpy as np
import pytest

from navsim.agents.drive_jepa_perception_based.experience.records import (
    ade_to_human,
    bootstrap_scene_ci,
    iter_export_records,
    load_npz,
    save_npz,
    split_logs_by_name,
)


class TestSplitLogs:
    def test_disjoint_and_covering(self):
        logs = [f"log_{i}" for i in range(50)]
        out = split_logs_by_name(logs, ratios=(0.6, 0.2, 0.2), seed=3)
        s = {k: set(v) for k, v in out.items()}
        assert not (s["memory"] & s["query_train"])
        assert not (s["memory"] & s["query_val"])
        assert not (s["query_train"] & s["query_val"])
        assert s["memory"] | s["query_train"] | s["query_val"] == set(logs)
        assert len(s["memory"]) == 30
        assert len(s["query_train"]) == 10
        assert len(s["query_val"]) == 10

    def test_deterministic(self):
        logs = [f"log_{i}" for i in range(30)]
        assert split_logs_by_name(logs, seed=1) == split_logs_by_name(logs, seed=1)

    def test_small_set(self):
        out = split_logs_by_name(["a", "b", "c"], seed=0)
        assert sum(len(v) for v in out.values()) == 3


class TestBootstrap:
    def test_ci_contains_point(self):
        vals = np.random.default_rng(0).normal(0.5, 0.1, size=200)
        p, lo, hi = bootstrap_scene_ci(vals, num_boot=200, seed=0)
        assert lo <= p <= hi
        assert p == pytest.approx(0.5, abs=0.02)

    def test_empty(self):
        p, lo, hi = bootstrap_scene_ci(np.array([]), num_boot=10)
        assert np.isnan(lo) and np.isnan(hi)


class TestNpz:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "rec.npz"
        save_npz(
            p,
            token=np.str_("abc"),
            proposals=np.zeros((32, 8, 3), dtype=np.float32),
            proposal_feature=np.ones((32, 256), dtype=np.float16),
            selected_idx=np.int64(7),
        )
        d = load_npz(p)
        assert d["token"].item() == "abc"
        assert d["proposals"].shape == (32, 8, 3)
        assert d["proposal_feature"].dtype == np.float16
        assert d["selected_idx"].item() == 7

    def test_iter_records(self, tmp_path):
        save_npz(tmp_path / "a.npz", x=np.array([1]))
        save_npz(tmp_path / "b.npz", x=np.array([2]))
        assert len(iter_export_records(tmp_path)) == 2
        assert len(iter_export_records(tmp_path, ["a", "missing"])) == 1


class TestAde:
    def test_zero_for_identical(self):
        human = np.random.default_rng(0).normal(size=(8, 3))
        props = np.stack([human, human + 1.0])
        ade = ade_to_human(props, human)
        assert ade[0] == pytest.approx(0.0)
        assert ade[1] == pytest.approx(1.0 * np.sqrt(2.0))
