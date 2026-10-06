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


def test_collect_bank_ragged_log_alignment():
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "scripts" / "experience"))
    from torch.utils.data import DataLoader
    from train_ewm import build_model, collate
    from eval_yhat import collect_bank

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    K = 4

    def item(tok, ln):
        return dict(token=tok, log_name=ln,
                    image_feature=rng.normal(size=(512, D)).astype(np.float32),
                    proposal_feature=rng.normal(size=(K, D)).astype(np.float32),
                    proposals=rng.normal(size=(K, 8, 3)).astype(np.float32),
                    pdm_score=rng.random(K).astype(np.float32),
                    pred_logit=rng.normal(size=(K, 6)).astype(np.float32),
                    outcomes=rng.normal(size=(K, 18)).astype(np.float32),
                    labels=rng.random((K, 6)).astype(np.float32))

    ds = [item("tA", "logA"), item("tB", "logB")]
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
    bank = {"tA": (rng.normal(size=(3, 8, 3)).astype(np.float32),
                   rng.random((3, 6)).astype(np.float32)),
            "tB": (rng.normal(size=(5, 8, 3)).astype(np.float32),
                   rng.random((5, 6)).astype(np.float32))}
    model = build_model("b3", 1, use_future=False, no_pfeat=True).eval()
    y, fin, lg, sc, subs = collect_bank(model, loader, torch.device("cpu"),
                                       bank=bank)

    assert len(lg) == len(y) == 2 * K + 3 + 5
    assert subs.shape == (2 * K + 3 + 5, 6)
    assert list(lg[:K]) == ["logA"] * K
    assert list(lg[K:2 * K]) == ["logB"] * K
    # ragged bank rows must stay aligned to their own scene's log/token
    assert list(lg[2 * K:2 * K + 3]) == ["logA"] * 3
    assert list(lg[2 * K + 3:]) == ["logB"] * 5
    assert list(sc[2 * K:2 * K + 3]) == ["tA"] * 3
    assert list(sc[2 * K + 3:]) == ["tB"] * 5
    assert np.allclose(fin[2 * K:2 * K + 3], bank["tA"][1][:, 5])
    assert np.allclose(fin[2 * K + 3:], bank["tB"][1][:, 5])
    assert np.allclose(subs[2 * K:2 * K + 3], bank["tA"][1][:, :6])
    assert np.allclose(subs[2 * K + 3:], bank["tB"][1][:, :6])


def test_knn_readout_subscores():
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "scripts" / "experience"))
    from eval_yhat import knn_readout
    rng = np.random.default_rng(0)
    # two-bank candidates in another log: one all-safe, one NC=0
    b_y = np.concatenate([np.ones((1, 4)), -np.ones((1, 4))], 0).astype(np.float32)
    b_subs = np.array([[1, 1, .5, 1, 1, .9],
                       [0, 1, .5, 1, 1, .1]], np.float32)
    b_logs = np.array(["logX", "logX"])
    q_y = np.ones((1, 4), np.float32)
    rd = knn_readout(q_y, np.array(["Q"]), b_y, b_subs, b_logs, k=2, t=.01)
    assert rd["fhat"][0] > .85            # weight dominated by the aligned nbr
    assert rd["p_nc"][0] < .1
    assert abs(rd["cos"][0]) < .1         # mean sim of (1,-1) pair
    # all neighbours masked (same log) -> softmax falls back to uniform
    rd2 = knn_readout(q_y, np.array(["logX"]), b_y, b_subs, b_logs, k=2, t=.01)
    assert .45 < rd2["fhat"][0] < .55
    assert .45 < rd2["p_nc"][0] < .55


def test_bank_filter_tokens_excludes_uncovered():
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "scripts" / "experience"))
    from train_ewm import bank_filter_tokens
    bank = {"t1": (np.zeros((2, 8, 3)), np.zeros((2, 6))), "t3": None}
    toks = ["t0", "t1", "t2", "t3"]
    out = bank_filter_tokens(toks, bank)
    # t3 is in bank but maps to None -> also excluded; t0/t2 absent -> excluded
    assert out == ["t1"]


def test_dump_tokens_align_with_loader_rows(tmp_path):
    """Regression: LatentDataset sorts items by token internally, so dump
    tokens must come from ds.items (not the file's original order), else
    every token-keyed lookup (subscores, pred_logit) is misaligned."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "scripts" / "experience"))
    from train_ewm import LatentDataset, build_index
    from eval_yhat import dump_tokens

    rng = np.random.default_rng(0)
    K = 4
    lat_dir = tmp_path / "lat" / "logA"
    lab_dir = tmp_path / "lab"
    lat_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    toks = ["tok_c", "tok_a", "tok_b"]          # deliberately unsorted
    for t in toks:
        np.savez(lat_dir / f"{t}.npz", token=t, log_name="logA",
                 image_feature=rng.normal(size=(512, D)).astype(np.float32),
                 proposal_feature=rng.normal(size=(K, D)).astype(np.float32),
                 proposals=rng.normal(size=(K, 8, 3)).astype(np.float32),
                 pdm_score=rng.random(K).astype(np.float32))
        subs = rng.random((K, 6)).astype(np.float32)
        np.savez(lab_dir / f"{t}.npz", token=t, log_name="logA",
                 subscores=subs,
                 descriptors=rng.normal(size=(K, 4, 21)).astype(np.float32),
                 vehicle_mask=np.ones((K, 4), bool),
                 main_desc_noatt=rng.normal(size=(K, 24)).astype(np.float32))

    ds = LatentDataset(list(toks), build_index(lat_dir.parent), lab_dir)
    toks_out = dump_tokens(ds)
    assert toks_out == sorted(toks)             # loader order, not file order
    for i, t in enumerate(toks_out):
        lab = np.load(lab_dir / f"{t}.npz")
        np.testing.assert_allclose(
            lab["subscores"][:, 5], ds[i]["labels"][:, 5],
            err_msg=f"row {i} token {t} misaligned")


def test_readout_select_rules():
    """Rule a picks argmax s_hat inside pdm top-K; rule b with beta=0 picks
    B0's top1; composed score follows the PDM weighting."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]
                           / "scripts" / "experience"))
    import numpy as np
    from readout_select import pick_scene, s_hat_scores, danger_auc, topk_idx

    rng = np.random.default_rng(1)
    S, K = 8, 32
    pdm = np.sort(rng.random((S, K)), axis=1)
    shat = rng.random((S, K))
    for s in range(S):
        tk = topk_idx(pdm[s], 4)
        want = tk[np.argmax(shat[s, tk])]
        assert pick_scene(pdm[s], shat[s], 4, "a") == want
        # beta=0 z-fusion degenerates to pdm argmax inside top-K
        assert pick_scene(pdm[s], shat[s], 4, "b", 0.0) == tk[0]

    rd = np.zeros((1, 4, 6), np.float32)
    rd[0, :, 0] = rd[0, :, 1] = 1.0          # NC=DAC=1
    rd[0, :, 2] = rd[0, :, 3] = rd[0, :, 4] = 1.0   # EP=TTC=C=1
    comp = s_hat_scores(rd, "composed")
    np.testing.assert_allclose(comp[0], np.ones(4), atol=1e-6)

    # danger_auc: make readout unsafe exactly on the dangerous candidates
    z = {"pdm": np.array([[.9, .8, .7, .6]]),
         "subs": np.array([[[1, 1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1]]]),
         "readout": np.array([[[1., 1., 1., 1., 1., 1.],
                              [0., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1., 1.]]])}
    d = danger_auc(z, k=4)
    assert d["n_danger"] == 1 and abs(d["auc"] - 1.0) < 1e-9
