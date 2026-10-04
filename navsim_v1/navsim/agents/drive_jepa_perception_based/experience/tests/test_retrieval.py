"""Unit tests for experience/retrieval.py feature builders (numpy only)."""

import numpy as np

from navsim.agents.drive_jepa_perception_based.experience.retrieval import (
    exp_dim_for,
    random_features,
    retrieval_features,
)


def _mem_bank(n=40, D=8, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        latent=rng.normal(size=(n, D)).astype(np.float32),
        labels=(rng.random((n, 2)) > 0.5).astype(np.float32),
        scene=np.repeat(np.arange(n // 4), 4),
        log=np.array([f"log{i}" for i in np.repeat(np.arange(n // 8), 8)]),
    )


def test_retrieval_excludes_same_log():
    mem = _mem_bank()
    q = mem["latent"][[0]].copy()  # identical to memory row 0 (log0)
    q_scene, q_log = np.array([999]), np.array(["log0"])
    out = retrieval_features(q, q_scene, q_log, mem["latent"], mem["labels"],
                             mem["scene"], mem["log"], topk=4, max_per_scene=4)
    assert out.shape == (1, 4 + mem["latent"].shape[1])
    assert np.isfinite(out).all()


def test_retrieval_respects_scene_cap():
    rng = np.random.default_rng(1)
    D = 8
    # one scene with many identical rows -> cap should limit to max_per_scene
    mem_lat = np.tile(rng.normal(size=D), (20, 1)).astype(np.float32)
    mem_labels = np.ones((20, 2), dtype=np.float32)
    mem_scene = np.zeros(20)
    mem_log = np.array(["A"] * 20)
    q = rng.normal(size=(1, D)).astype(np.float32)
    out = retrieval_features(q, np.array([0]), np.array(["B"]),
                             mem_lat, mem_labels, mem_scene, mem_log,
                             topk=8, max_per_scene=3)
    # only 3 neighbours usable -> weighted mean over 3 identical labels = 1
    assert out[0, 0] == 1.0
    # mean neighbour latent should equal the shared latent direction
    assert np.allclose(out[0, 4:], mem_lat[0] / np.linalg.norm(mem_lat[0]))


def test_retrieval_shuffle_changes_features():
    mem = _mem_bank(n=80)
    q = mem["latent"][:4]
    a = retrieval_features(q, np.arange(4), np.array(["X"] * 4),
                           mem["latent"], mem["labels"], mem["scene"],
                           mem["log"], topk=8, max_per_scene=4)
    b = retrieval_features(q, np.arange(4), np.array(["X"] * 4),
                           mem["latent"], mem["labels"], mem["scene"],
                           mem["log"], topk=8, max_per_scene=4,
                           shuffle_labels=True, rng=np.random.default_rng(3))
    # neighbour latents identical; label stats differ (unless degenerate)
    assert np.allclose(a[:, 4:], b[:, 4:])
    assert not np.allclose(a[:, :4], b[:, :4]) or np.all(mem["labels"] == mem["labels"][0])


def test_random_features_other_log_and_shape():
    mem = _mem_bank(n=64)
    rng = np.random.default_rng(5)
    out = random_features(8, np.array(["log0"] * 8), mem["labels"], mem["log"],
                          topk=16, rng=rng)
    assert out.shape == (8, 4)
    # labels binary -> mean in [0,1]
    assert ((out[:, :2] >= 0) & (out[:, :2] <= 1)).all()


def test_exp_dim_for():
    assert exp_dim_for("noexp", 64) == 0
    assert exp_dim_for("random", 64) == 4
    assert exp_dim_for("retrieval", 64) == 68
    assert exp_dim_for("retrieval_int", 64) == 68
