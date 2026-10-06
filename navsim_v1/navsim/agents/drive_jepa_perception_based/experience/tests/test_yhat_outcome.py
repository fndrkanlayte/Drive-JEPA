"""Tests for the y_hat training upgrades (no_pfeat / bank / xs / jitter)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from navsim.agents.drive_jepa_perception_based.experience.ewm import (  # noqa: E402
    ActionEncoder, EWMJEPA,
)
from navsim.agents.drive_jepa_perception_based.experience.ewm_structured import (  # noqa: E402
    EWMStructured, b3_loss, xs_loss,
)

D, L = 256, 64


def test_no_pfeat_ignores_proposal_feature():
    torch.manual_seed(0)
    enc = ActionEncoder(no_pfeat=True)
    tr = torch.randn(2, 5, 8, 3)
    pf1 = torch.randn(2, 5, D)
    pf2 = torch.randn(2, 5, D) * 100
    enc.eval()
    assert torch.equal(enc(pf1, tr), enc(pf2, tr))


def test_traj_jitter_train_only():
    enc = ActionEncoder(traj_jitter=0.2)
    tr = torch.randn(2, 5, 8, 3)
    pf = torch.randn(2, 5, D)
    enc.eval()
    assert torch.equal(enc(pf, tr), enc(pf, tr.clone()))
    enc.train()
    torch.manual_seed(0)
    a1 = enc(pf, tr)
    torch.manual_seed(0)
    a2 = enc(pf, tr)
    assert torch.equal(a1, a2)          # deterministic under same seed
    assert not torch.equal(a1, enc(pf, tr))


def test_xs_loss_ignores_same_log():
    torch.manual_seed(0)
    y = torch.randn(2, 4, L)
    s = torch.rand(2, 4, 6)
    # all candidates same log -> no valid rows -> zero loss
    l_same = torch.zeros(2, 4, dtype=torch.long)
    assert xs_loss(y, s, l_same).item() == 0.0
    # different logs -> positive KL
    l_diff = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]])
    assert xs_loss(y, s, l_diff).item() > 0


def test_xs_loss_low_when_geometry_matches():
    # y identical to subs (padded): nearest neighbours share subscore vector
    s = torch.tensor([[[1., 1, 1, 1, 1, 1]] * 2 + [[0., 0, 0, 0, 0, 0]] * 2,
                      [[1., 1, 1, 1, 1, 1]] * 2 + [[0., 0, 0, 0, 0, 0]] * 2])
    y = torch.cat([s[:, :, :6], torch.zeros(2, 4, L - 6)], dim=-1) + 1e-4 * torch.randn(2, 4, L)
    l_diff = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]])
    good = xs_loss(y, s, l_diff).item()
    y_bad = y.flip(1)                  # scramble candidate order within scenes
    bad = xs_loss(y_bad, s, l_diff).item()
    assert good < bad


def _mk_batch(B=2, K0=4, bank_k=2, M=4, F_agent=20):
    K = K0 + bank_k
    batch = dict(
        agent_vals=torch.zeros(B, K, M, F_agent),
        agent_valid=torch.zeros(B, K, M, F_agent),
        agent_slot=torch.zeros(B, K, M),
        labels=torch.rand(B, K, 6),
    )
    # give B0 rows some agent supervision
    batch["agent_vals"][:, :K0] = torch.rand(B, K0, M, F_agent)
    batch["agent_valid"][:, :K0] = torch.rand(B, K0, M, F_agent)
    batch["agent_slot"][:, :K0] = (torch.rand(B, K0, M) > 0.3).float()
    return batch


def test_b3_loss_bank_masks_agent_slots():
    torch.manual_seed(0)
    B, K0, BK = 2, 4, 2
    model = EWMStructured(n_layers=1, use_future=False)
    img = torch.randn(B, 512, D)
    tr = torch.randn(B, K0 + BK, 8, 3)
    out = model(img, torch.randn(B, K0 + BK, D), tr)
    batch = _mk_batch(B, K0, BK)
    l1, _ = b3_loss(model, out, batch, bank_k=BK)
    # perturb bank agent targets: loss must not change (masked)
    b2 = {k: v.clone() for k, v in batch.items()}
    b2["agent_vals"][:, K0:] = torch.randn(B, BK, 4, 20) * 10
    b2["agent_valid"][:, K0:] = torch.ones(B, BK, 4, 20)
    b2["agent_slot"][:, K0:] = torch.ones(B, BK, 4)
    l2, _ = b3_loss(model, out, b2, bank_k=BK)
    assert abs(l1.item() - l2.item()) < 1e-5


def test_b3_loss_bank_relational_ego_only():
    torch.manual_seed(0)
    B, K0, BK = 2, 4, 2
    model = EWMStructured(n_layers=1, use_future=False)
    img = torch.randn(B, 512, D)
    tr = torch.randn(B, K0 + BK, 8, 3)
    out = model(img, torch.randn(B, K0 + BK, D), tr)
    batch = _mk_batch(B, K0, BK)
    l1, parts1 = b3_loss(model, out, batch, bank_k=BK)
    # corrupt bank agent-slot y_hat: l2/rel must not change
    out2 = {k: v.clone() if torch.is_tensor(v) else v for k, v in out.items()}
    out2["y_hat"] = out["y_hat"].clone()
    out2["y_hat"][:, K0:, 1:] = 5.0
    l2, parts2 = b3_loss(model, out2, batch, bank_k=BK)
    assert abs(parts1["rel"] - parts2["rel"]) < 1e-5
    assert abs(parts1["l2"] - parts2["l2"]) < 1e-5
    # same corruption must change loss under bank_k=0 semantics? (rel uses all
    # slots, so yes if it were treated as B0 rows)
    l3, parts3 = b3_loss(model, out2, batch, bank_k=0)
    assert abs(parts3["rel"] - parts2["rel"]) > 1e-5


def test_bank_npz_loading(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "scripts" / "experience"))
    from train_ewm import load_bank_npz
    toks = np.array(["a", "b"])
    trajs = np.random.rand(5, 8, 3).astype(np.float32)
    subs = np.random.rand(5, 6).astype(np.float32)
    np.savez(tmp_path / "b.npz", tokens=toks, counts=np.array([2, 3]),
             trajs=trajs, subs=subs)
    bank = load_bank_npz(str(tmp_path / "b.npz"))
    assert set(bank) == {"a", "b"}
    assert bank["a"][0].shape == (2, 8, 3)
    assert np.allclose(bank["b"][1], subs[2:])


def test_knn_val_mae_same_log_exclusion():
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "scripts" / "experience"))
    from train_ewm import knn_val_mae
    rng = np.random.default_rng(0)
    b_y = rng.normal(size=(20, L)).astype(np.float32)
    b_fin = np.linspace(0, 1, 20).astype(np.float32)
    b_logs = np.array(["L"] * 10 + ["M"] * 10)
    q_y = b_y[:1].copy()
    q_fin = np.array([b_fin[0]])
    mae_diff_log = knn_val_mae(q_y, q_fin, np.array(["L"]), b_y, b_fin, b_logs, k=4)
    # nearest same-log neighbour excluded -> estimate biased away from true 0
    assert mae_diff_log > 0.0


def test_build_model_no_pfeat():
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "scripts" / "experience"))
    from train_ewm import build_model
    m = build_model("b3", 1, use_future=False, no_pfeat=True, dropout=0.1,
                    traj_jitter=0.2)
    assert isinstance(m, EWMStructured)
    assert m.action.no_pfeat
    img = torch.randn(1, 512, D)
    out = m(img, torch.randn(1, 10, D), torch.randn(1, 10, 8, 3))
    assert out["y_hat"].shape == (1, 10, 5, L)
