import numpy as np
import pytest

torch = pytest.importorskip("torch")

from navsim.agents.drive_jepa_perception_based.experience.descriptors import NUM_DESCRIPTOR_FIELDS
from navsim.agents.drive_jepa_perception_based.experience.ewm_structured import (
    N_AGENT_F, B1Aux, EWMStructured, agent_targets, b1aux_loss, b3_loss)


def _batch(B=2, K=32, M=4):
    rng = np.random.default_rng(0)
    desc = rng.normal(size=(B, K, M, NUM_DESCRIPTOR_FIELDS)).astype(np.float32)
    desc[:, :, -1] = np.nan
    vm = np.ones((B, K, M), bool); vm[:, :, -1] = False
    vals, valid, slot = zip(*[agent_targets(desc[i], vm[i]) for i in range(B)])
    return dict(
        image_feature=torch.randn(B, 512, 256), proposal_feature=torch.randn(B, K, 256),
        proposals=torch.randn(B, K, 8, 3), labels=torch.rand(B, K, 6),
        agent_vals=torch.from_numpy(np.stack(vals)), agent_valid=torch.from_numpy(np.stack(valid)),
        agent_slot=torch.from_numpy(np.stack(slot)), expert_traj=torch.randn(B, 8, 3),
        has_traj=torch.ones(B), image_future=torch.randn(B, 512, 256),
        has_future=torch.tensor([1.0, 0.0]))


def test_agent_targets_masks_nan_and_empty_slots():
    b = _batch()
    assert b["agent_vals"].shape[-1] == N_AGENT_F
    assert float(b["agent_valid"][:, :, -1].sum()) == 0
    assert torch.isfinite(b["agent_vals"]).all()


@pytest.mark.parametrize("use_future", [True, False])
def test_b3_forward_backward_and_ema(use_future):
    b = _batch()
    m = EWMStructured(n_layers=1, use_future=use_future)
    out = m(b["image_feature"], b["proposal_feature"], b["proposals"])
    assert out["readout"].shape == (2, 32, 6) and out["y_hat"].shape == (2, 32, 5, 64)
    loss, parts = b3_loss(m, out, b)
    loss.backward()
    assert ("fut" in parts) == use_future and np.isfinite(loss.item())
    assert all(p.grad is None for p in m.enc_agent_ema.parameters())
    before = next(m.enc_agent_ema.parameters()).clone()
    with torch.no_grad():
        for p in m.enc_agent.parameters():
            p.add_(1.0)
    m.update_ema(0.5)
    assert not torch.allclose(before, next(m.enc_agent_ema.parameters()))


def test_b1aux_forward_backward():
    b = _batch()
    m = B1Aux(n_layers=1)
    out = m(b["image_feature"], b["proposal_feature"], b["proposals"])
    assert out["logits"].shape == (2, 32, 6) and out["agent_pred"].shape == (2, 32, 4, N_AGENT_F)
    loss, _ = b1aux_loss(out, b)
    loss.backward()
    assert np.isfinite(loss.item())
