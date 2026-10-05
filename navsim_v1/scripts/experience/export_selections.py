#!/usr/bin/env python3
"""Export per-token selected trajectories for the selection-replay agent.

For each variant risk file (``<variant>.npz`` from train_experience's
navtest_risk output) this writes ``<out_dir>/<variant>.npz`` with:

  tokens       (S,)   str
  trajectories (S,8,3) float32 -- proposals[argmax(pdm_score - lambda * r_hat)]

plus ``<out_dir>/original.npz`` with the plain argmax(pdm_score) selection.
Feed these to ``run_pdm_score agent=replay_selection_agent
agent.selection_file=<file>`` for the official PDMS of a re-ranked policy.

Usage:
  python scripts/experience/export_selections.py \
      --export_dir  $EXP/experience/navtest_export_full \
      --risk_dir    $EXP/experience/train_experience_full5/navtest_risk \
      --lambda_json $EXP/experience/rerank_eval_out/rerank_results.json \
      --out_dir     $EXP/experience/selections_navtest
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))

from navsim.agents.drive_jepa_perception_based.experience.records import (  # noqa: E402
    load_npz,
    save_npz,
)


def r_hat(risk: np.ndarray) -> np.ndarray:
    """risk (S,K,2) -> combined risk (S,K): 1 - (1-p_nc)(1-p_ttc)."""
    p = np.clip(risk.astype(np.float64), 0.0, 1.0)
    return 1.0 - (1.0 - p[..., 0]) * (1.0 - p[..., 1])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export_dir", required=True)
    p.add_argument("--risk_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--lambda_json", default=None,
                   help="rerank_results.json; per-variant tuned lambda under "
                        "'best_lambda' (missing entries -> 0)")
    p.add_argument("--gated_json", action="store_true",
                   help="read (lam,g) from 'best_gated' + apply conf gate")
    p.add_argument("--default_lambda", type=float, default=0.0)
    p.add_argument("--variants", nargs="*", default=None,
                   help="subset of risk files to export (default: all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = Path(args.export_dir)

    best_lam = {}
    best_gated = {}
    if args.lambda_json:
        rr = json.loads(Path(args.lambda_json).read_text())
        best_lam = rr["best_lambda"]
        best_gated = rr.get("best_gated", {})

    jobs = {"original": None}
    for rp in sorted(Path(args.risk_dir).glob("*.npz")):
        if args.variants and rp.stem not in args.variants:
            continue
        jobs[rp.stem] = rp

    for name, rp in jobs.items():
        if rp is None:
            risk_map = {}
            lam, gate = 0.0, 0.0
        else:
            risk_npz = load_npz(rp)
            risk_map = {str(t): i for i, t in enumerate(risk_npz["tokens"])}
            if args.gated_json and name in best_gated:
                lam, gate = [float(v) for v in best_gated[name]]
            else:
                lam, gate = float(best_lam.get(name, args.default_lambda)), 0.0
        tokens, trajs, n_fallback = [], [], 0
        for ep in sorted(export_dir.glob("*.npz")):
            exp = load_npz(ep)
            token = str(exp["token"].item())
            pdm = np.asarray(exp["pdm_score"], dtype=np.float64)
            props = np.asarray(exp["proposals"], dtype=np.float32)  # (K,8,3)
            if rp is not None and token in risk_map:
                r_ = risk_npz["risk"][risk_map[token]]
                if r_.shape[-1] == 1:               # residual: +lam * r_res
                    corr, sign = r_[..., 0].astype(np.float64), 1.0
                else:                                # risk: -lam * r_hat
                    corr, sign = r_hat(r_[None])[0], -1.0
                if gate > 0.0 and "conf" in risk_npz:
                    corr = corr * (risk_npz["conf"][risk_map[token]] > gate)
                sel = int(np.argmax(pdm + sign * lam * corr))
            else:
                sel = int(np.argmax(pdm))
                n_fallback += rp is not None
            tokens.append(token)
            trajs.append(props[sel])
        save_npz(out_dir / f"{name}.npz",
                 tokens=np.asarray(tokens),
                 trajectories=np.stack(trajs))
        print(f"[sel] {name}: {len(tokens)} scenes (lam={lam:g}, gate={gate:g}, "
              f"{n_fallback} fell back to argmax) -> {out_dir / (name + '.npz')}")


if __name__ == "__main__":
    main()
