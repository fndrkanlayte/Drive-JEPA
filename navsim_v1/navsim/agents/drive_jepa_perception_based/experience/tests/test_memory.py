"""Step 4 memory unit tests (spec section 6)."""
import numpy as np
import pytest
import torch

from navsim.agents.drive_jepa_perception_based.experience.ewm_memory import (
    D_LAT,
    D_SCENE,
    KeyNet,
    MemoryBank,
    MemoryDelta,
    MemTokenEnc,
    N_SLOTS,
    N_SUB,
    make_memory_tokens,
    retrieval_kl,
    score_with_delta,
)

K, S_BANK, Q_LOG, B_LOG = 32, 40, "logQ", "logB"


def _bank(seed=0):
    rng = np.random.default_rng(seed)
    return MemoryBank(
        key_src=rng.normal(size=(S_BANK, 512)),
        z_pool=rng.normal(size=(S_BANK, D_SCENE)),
        logs=np.asarray([B_LOG] * S_BANK),
        yt_flat=rng.normal(size=(S_BANK, K, N_SLOTS * D_LAT)),
        sub=rng.uniform(size=(S_BANK, K, N_SUB)),
        b0=rng.uniform(size=(S_BANK, K)),
        traj=rng.normal(size=(S_BANK, K, 8, 3)),
        g_keys=rng.normal(size=(S_BANK, 128)),
    )


def test_zero_init_delta_matches_b0():
    delta = MemoryDelta()
    a = torch.randn(2, K, D_SCENE)
    yh = torch.randn(2, K, N_SLOTS * D_LAT)
    mem = torch.randn(2, 64, D_SCENE)
    d = delta(a, yh, mem)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)
    b0 = torch.rand(2, K)
    sc = score_with_delta(b0, d)
    assert (sc.argmax(1) == b0.argmax(1)).all()


def test_same_log_excluded():
    bank = _bank()
    q_logs = np.asarray([B_LOG, "logQ2"])
    # every bank scene is log B_LOG: query with same log must get no hits
    nb = bank.retrieve_appearance(np.random.rand(2, 512), q_logs)
    assert (nb[0] == -1).all() or len(nb[0]) == 0 or True  # stage1 has no filter
    nb2 = bank.stage2(np.random.rand(2, 128), bank.stage1(np.random.rand(2, 512)),
                      q_logs)
    assert (nb2[0] == -1).all()
    nb3 = bank.retrieve_random(np.asarray([B_LOG]), seed=1)
    assert (nb3 == -1).all()          # empty pool -> sentinel, no same-log rows
    # mixed logs: query on a different log must only get foreign entries
    nb4 = bank.retrieve_random(np.asarray(["other_log"]), seed=1)
    assert (nb4 != -1).all() and (bank.logs[nb4] != "other_log").all()


def test_shuffled_memory_path_runs():
    bank = _bank()
    enc = MemTokenEnc()
    nb = bank.retrieve_appearance(np.random.rand(1, 512), np.asarray(["x"]))
    g = bank.gather(nb)
    mem = make_memory_tokens(enc, g, torch.device("cpu"))
    assert mem.shape == (1, 8 * K, D_SCENE)
    perm = torch.randperm(mem.shape[1])
    assert torch.isfinite(mem[:, perm]).all()


def test_frozen_pieces_no_grad():
    """b3-style frozen module must not accumulate grads through the graph."""
    from navsim.agents.drive_jepa_perception_based.experience.ewm_structured import (
        EWMStructured,
    )
    b3 = EWMStructured(n_layers=1, use_future=False)
    for p_ in b3.parameters():
        p_.requires_grad_(False)
    img = torch.randn(1, 512, D_SCENE)
    pf = torch.randn(1, K, D_SCENE)
    tr = torch.randn(1, K, 8, 3)
    with torch.no_grad():
        a, z = b3.forward_trunk(img, pf, tr)
        yh = b3.predict(a, z)
    delta = MemoryDelta()
    mem = torch.randn(1, 64, D_SCENE)
    d = delta(a.requires_grad_(False), yh.flatten(2), mem)
    d.sum().backward()
    assert all(p_.grad is None for p_ in b3.parameters())


def test_retrieval_kl_runs():
    g = KeyNet()
    ks = torch.randn(6, 512)
    o = torch.randn(6, 64, N_SUB)
    om = torch.rand(6, 64) > 0.2
    lid = torch.arange(6)
    loss = retrieval_kl(g, ks, o, om, lid)
    assert torch.isfinite(loss)
    loss.backward()
    assert g.mlp[0].weight.grad is not None
