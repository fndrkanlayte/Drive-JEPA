"""Q11: candidate-level diagnostic on navtrain query_val (810 scenes).

Pre-registered subsets/metrics: see README.md "Q11 candidate-level
diagnostic". Uses existing predictions only (no new training) except a
tiny logistic "parametric_desc" baseline fit on memory rows, and a
true-descriptor kNN oracle — both inference-time predictors over the
navtrain memory bank, never navtest.

Usage:
    python scripts/experience/q11_diag.py \
        --labels_dir .../navtrain_labels_3k \
        --export_dir .../navtrain_export_3k \
        --qval_risk_dir .../train_experience_full5_m/query_val_risk \
        --out_dir .../q11_diag_out
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train_experience as te  # noqa: E402
from navsim.agents.drive_jepa_perception_based.experience.descriptors import (  # noqa: E402
    ITYPE_CROSSING,
    ITYPE_ONCOMING,
    ITYPE_SAME_DIR,
)
from navsim.agents.drive_jepa_perception_based.experience.records import (  # noqa: E402
    load_npz,
)

RISK_NPZ_VARIANTS = [
    "noexp", "noexp_int", "random", "shuffle",
    "retrieval", "retrieval_int", "pred_desc_retrieval_pp",
]
# risk column indices inside the (S,K,2) npz
COL_NC, COL_TTC = 0, 1
SUB_NC, SUB_TTC, SUB_FIN = 0, 3, 5
KNN_Q = 20      # density neighbourhood
KNN_ORACLE = 16  # oracle-kNN label averaging
TURN_RAD = np.deg2rad(30.0)
ITYPES = [(ITYPE_SAME_DIR, "SAME_DIR"), (ITYPE_CROSSING, "CROSSING"),
          (ITYPE_ONCOMING, "ONCOMING")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--export_dir", required=True)
    p.add_argument("--qval_risk_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--dt_enter_thresh", type=float, default=2.0)
    p.add_argument("--num_boot", type=int, default=1000)
    p.add_argument("--device", default="cuda" if
                   __import__("torch").cuda.is_available() else "cpu")
    return p.parse_args()


def wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def zscore(a: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    z = (a - m) / np.where(s > 0, s, 1.0)
    return np.where(np.isfinite(z), z, 0.0)


def knn_mean_dist(q: np.ndarray, m: np.ndarray, q_log: np.ndarray,
                  m_log: np.ndarray, k: int, device: str) -> np.ndarray:
    """Mean Euclidean distance to k nearest memory rows (other logs)."""
    import torch
    qt = torch.from_numpy(q).to(device)
    mt = torch.from_numpy(m).to(device)
    out = np.zeros(len(q), dtype=np.float64)
    for i in range(0, len(q), 512):
        d = torch.cdist(qt[i:i + 512], mt)  # (b, M)
        same = torch.from_numpy(
            q_log[i:i + 512, None] == m_log[None, :]).to(device)
        d = d.masked_fill(same, float("inf"))
        top = torch.topk(d, min(k, d.shape[1] - 1), largest=False).values
        out[i:i + 512] = top.mean(1).cpu().numpy()
    return out


def knn_mean_label(q: np.ndarray, m: np.ndarray, q_log: np.ndarray,
                   m_log: np.ndarray, m_y: np.ndarray, k: int,
                   device: str) -> np.ndarray:
    """Mean neighbour label (n_targets,) for each query candidate."""
    import torch
    qt = torch.from_numpy(q).to(device)
    mt = torch.from_numpy(m).to(device)
    my = torch.from_numpy(m_y.astype(np.float32)).to(device)
    out = np.zeros((len(q), m_y.shape[1]), dtype=np.float64)
    for i in range(0, len(q), 512):
        d = torch.cdist(qt[i:i + 512], mt)
        same = torch.from_numpy(
            q_log[i:i + 512, None] == m_log[None, :]).to(device)
        d = d.masked_fill(same, float("inf"))
        idx = torch.topk(d, min(k, d.shape[1] - 1), largest=False).indices
        out[i:i + 512] = my[idx].mean(1).cpu().numpy()
    return out


def scene_ap(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision within one scene (float nan if no positive)."""
    order = np.argsort(-s)
    yy = y[order] > 0.5
    if yy.sum() == 0:
        return float("nan")
    tp = np.cumsum(yy)
    prec = tp / (np.arange(len(yy)) + 1)
    return float((prec * yy).sum() / yy.sum())


def boot_ci(vals: np.ndarray, num_boot: int, seed: int) -> Tuple[float, float, float]:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), (num_boot, len(vals)))
    ms = vals[idx].mean(1)
    return float(vals.mean()), float(np.quantile(ms, .025)), float(np.quantile(ms, .975))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[q11] loading rows", flush=True)
    rows = te.load_rows(Path(args.labels_dir), Path(args.export_dir),
                        args.dt_enter_thresh)
    # rows["group"] is already assigned by load_rows' caller path in
    # train_experience; load_rows itself does not set it, so re-split here
    # with the same helper/ratios/seed.
    from navsim.agents.drive_jepa_perception_based.experience.records import (
        split_logs_by_name,
    )
    groups = split_logs_by_name(np.unique(rows["scene_log"]),
                                ratios=(0.6, 0.2, 0.2),
                                seed=args.split_seed)
    group_of = {l: g for g, ls in groups.items() for l in ls}
    sc_group = np.array([group_of[l] for l in rows["scene_log"]])
    qv_scenes = np.flatnonzero(sc_group == "query_val")
    mem_row_mask = np.array(
        [group_of[l] == "memory" for l in rows["log"]])

    # ---------- per-scene qval arrays --------------------------------------
    K = 32
    scene_of_row = rows["scene_id"]
    sid = scene_of_row
    # rebuild per-scene arrays for qval scenes only
    y_all = rows["y"]                     # (N, 2) nc_unsafe, ttc_bad
    pdm_all = rows["pdm"]                 # (N,)
    fin_all = rows["final"]               # (N,)
    aux_all = rows["aux"]                 # (N, 5) conflict/itype/dt/pet/mindist
    desc_all = rows["desc"]               # (N, D_desc) raw NaN ok
    scene_list = rows["tokens"]
    S_q = len(qv_scenes)

    # map scene_id -> row slice (rows are appended per scene, K each)
    row_of = {s: np.flatnonzero(scene_of_row == s) for s in qv_scenes}
    y_q = np.stack([y_all[row_of[s]] for s in qv_scenes])      # (S,K,2)
    pdm_q = np.stack([pdm_all[row_of[s]] for s in qv_scenes])  # (S,K)
    fin_q = np.stack([fin_all[row_of[s]] for s in qv_scenes])  # (S,K)
    aux_q = np.stack([aux_all[row_of[s]] for s in qv_scenes])  # (S,K,5)
    desc_q = np.stack([desc_all[row_of[s]] for s in qv_scenes])
    q_tokens = np.array([scene_list[s] for s in qv_scenes])
    q_logs = np.array([rows["scene_log"][s] for s in qv_scenes])
    q_row_log = np.repeat(q_logs, K)

    # turn flag from proposals heading (export npz, aligned via tokens)
    print("[q11] turn flags", flush=True)
    turn_sel = np.zeros(S_q, dtype=bool)
    turn_any = np.zeros(S_q, dtype=bool)
    ed = Path(args.export_dir)
    for i, tok in enumerate(q_tokens):
        exp = load_npz(ed / f"{tok}.npz")
        prop = np.asarray(exp["proposals"], dtype=np.float64)  # (K,8,3)
        dh = np.abs(wrap(prop[:, -1, 2] - prop[:, 0, 2]))
        turn_any[i] = bool((dh > TURN_RAD).any())
        sel = int(np.argmax(pdm_q[i]))
        turn_sel[i] = bool(dh[sel] > TURN_RAD)

    # ---------- subset masks (scene level, on argmax candidate) ------------
    sel0 = np.argmax(pdm_q, axis=1)
    ar = np.arange(S_q)
    nc_sel = y_q[ar, sel0, 0] > 0.5
    ttc_sel = y_q[ar, sel0, 1] > 0.5
    has_safe = ((y_q[..., 0] < .5) & (y_q[..., 1] < .5)).any(1)
    has_unsafe = ((y_q[..., 0] > .5) | (y_q[..., 1] > .5)).any(1)
    aux_sel = aux_q[ar, sel0]  # (S,5)
    conf_s = aux_sel[:, 0] == 1.0
    dt_s = np.abs(aux_sel[:, 2])
    it_s = aux_sel[:, 1]

    subsets: List[Tuple[str, np.ndarray]] = [("all", np.ones(S_q, bool))]
    subsets.append(("S_err", (nc_sel | ttc_sel) & has_safe))
    subsets.append(("S_flip", has_unsafe & has_safe))
    for code, name in ITYPES:
        m = conf_s & (dt_s < args.dt_enter_thresh) & (it_s == code)
        subsets.append((f"S_int_{name}", m))
        subsets.append((f"S_int_{name}_turn", m & turn_sel))
    subsets.append(("S_int_any", conf_s & (dt_s < args.dt_enter_thresh)))

    # ---------- density tertiles -------------------------------------------
    print("[q11] density", flush=True)
    mem_desc = rows["desc"][mem_row_mask]
    mem_log = rows["log"][mem_row_mask]
    mu = np.nanmean(np.where(np.isfinite(mem_desc), mem_desc, np.nan), axis=0)
    sd = np.nanstd(np.where(np.isfinite(mem_desc), mem_desc, np.nan), axis=0)
    mdz = zscore(mem_desc, mu, sd)
    qdz = zscore(desc_q.reshape(-1, desc_q.shape[2]), mu, sd)
    q_density = knn_mean_dist(qdz, mdz, q_row_log, mem_log, KNN_Q,
                              args.device).reshape(S_q, K)
    # memory's own density: self is excluded automatically because the
    # row's own log is masked out
    m_density = knn_mean_dist(mdz, mdz, mem_log, mem_log, KNN_Q,
                              args.device)
    t1, t2 = np.quantile(m_density, [1 / 3, 2 / 3])
    print(f"[q11] density tertile edges {t1:.4f} {t2:.4f}", flush=True)
    cand_ter = np.digitize(q_density, [t1, t2])          # (S,K) 0/1/2
    scene_ter = cand_ter[ar, sel0]                       # argmax cand tertile

    # ---------- method scores ----------------------------------------------
    print("[q11] method scores", flush=True)
    scores: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    scores["native_pdm"] = (1.0 - pdm_q, 1.0 - pdm_q)

    rd = Path(args.qval_risk_dir)
    for v in RISK_NPZ_VARIANTS:
        fz = rd / f"{v}.npz"
        if not fz.is_file():
            print(f"[q11] missing {fz}", flush=True)
            continue
        z = load_npz(fz)
        t2i = {t: i for i, t in enumerate(z["tokens"].tolist())}
        idx = np.array([t2i[t] for t in q_tokens if t in t2i])
        keep = np.array([i for i, t in enumerate(q_tokens) if t in t2i])
        risk = np.zeros((S_q, K, 2))
        risk[keep] = z["risk"][idx]
        scores[v] = (risk[..., COL_NC], risk[..., COL_TTC])

    # oracle kNN on true descriptors
    m_y = rows["y"][mem_row_mask]
    knn_lab = knn_mean_label(qdz, mdz, q_row_log, mem_log, m_y,
                             KNN_ORACLE, args.device).reshape(S_q, K, 2)
    scores["oracle_knn"] = (knn_lab[..., 0], knn_lab[..., 1])

    # parametric desc: logreg desc -> label, trained on memory rows
    try:
        from sklearn.linear_model import LogisticRegression
        pd_scores = np.zeros((S_q, K, 2))
        for c in range(2):
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(mdz, m_y[:, c])
            pd_scores[..., c] = clf.predict_proba(
                qdz)[:, 1].reshape(S_q, K)
        scores["parametric_desc"] = (pd_scores[..., 0], pd_scores[..., 1])
    except Exception as e:  # noqa
        print(f"[q11] parametric_desc skipped: {e}", flush=True)

    # ---------- metrics -----------------------------------------------------
    print("[q11] metrics", flush=True)
    total_risk = {m: (a + b) for m, (a, b) in scores.items()}
    lines = ["# Q11 candidate-level diagnostic (query_val)", ""]
    lines.append(f"scenes: {S_q} | density tertile edges "
                 f"[{t1:.4f}, {t2:.4f}] (from memory) | "
                 f"turn>30deg sel: {turn_sel.sum()} scenes, "
                 f"any: {turn_any.sum()}")
    lines.append("")
    lines.append("| subset | tertile | method | n_scenes | n_pos_nc |"
                 " n_pos_ttc | AUPRC_nc [CI] | AUPRC_ttc [CI] |"
                 " rank_safest [CI] | top1_unsafe [CI] |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    rng = np.random.default_rng(0)
    for sname, smask in subsets:
        for tname, tval in (("all_den", -1), ("common", 0), ("mid", 1),
                            ("rare", 2)):
            scen_mask = smask if tval < 0 else smask & (scene_ter == tval)
            idx_s = np.flatnonzero(scen_mask)
            if len(idx_s) == 0:
                continue
            for mname, (s_nc, s_ttc) in scores.items():
                s_r = total_risk[mname]
                # candidate tertile mask within these scenes
                ter_c = np.ones((len(idx_s), K), bool) if tval < 0 else \
                    cand_ter[idx_s] == tval
                ap_nc, ap_tt, rk_safe, t1u = [], [], [], []
                for si_pos, si in enumerate(idx_s):
                    cmask = ter_c[si_pos]
                    if cmask.sum() < 2:
                        continue
                    yy_nc = y_q[si][cmask, 0]
                    yy_tt = y_q[si][cmask, 1]
                    ap_nc.append(scene_ap(yy_nc, s_nc[si][cmask]))
                    ap_tt.append(scene_ap(yy_tt, s_ttc[si][cmask]))
                    # safest candidate & its rank (scene-level)
                    safe = (y_q[si][:, 0] < .5) & (y_q[si][:, 1] < .5)
                    if safe.any():
                        best = np.flatnonzero(safe)[
                            np.argmax(fin_q[si][safe])]
                        order = np.argsort(s_r[si])  # ascending risk
                        rk_safe.append(
                            int(np.flatnonzero(order == best)[0]) + 1)
                        pick = order[0]
                        t1u.append(float(
                            (y_q[si][pick, 0] > .5)
                            or (y_q[si][pick, 1] > .5)))
                n_pos_nc = int(y_q[idx_s][..., 0].sum())
                n_pos_tt = int(y_q[idx_s][..., 1].sum())
                a_m, a_lo, a_hi = boot_ci(np.array(ap_nc), args.num_boot,
                                          int(rng.integers(1e9)))
                t_m, t_lo, t_hi = boot_ci(np.array(ap_tt), args.num_boot,
                                          int(rng.integers(1e9)))
                r_m, r_lo, r_hi = boot_ci(np.array(rk_safe, float),
                                          args.num_boot,
                                          int(rng.integers(1e9)))
                u_m, u_lo, u_hi = boot_ci(np.array(t1u, float),
                                          args.num_boot,
                                          int(rng.integers(1e9)))
                lines.append(
                    f"| {sname} | {tname} | {mname} | {len(idx_s)} | "
                    f"{n_pos_nc} | {n_pos_tt} | "
                    f"{a_m:.3f} [{a_lo:.3f},{a_hi:.3f}] | "
                    f"{t_m:.3f} [{t_lo:.3f},{t_hi:.3f}] | "
                    f"{r_m:.2f} [{r_lo:.2f},{r_hi:.2f}] | "
                    f"{u_m:.3f} [{u_lo:.3f},{u_hi:.3f}] |")

    (out_dir / "q11_results.md").write_text("\n".join(lines) + "\n")
    np.savez(out_dir / "q11_cache.npz", q_density=q_density,
             cand_ter=cand_ter, scene_ter=scene_ter,
             tokens=q_tokens)
    print(f"[q11] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
