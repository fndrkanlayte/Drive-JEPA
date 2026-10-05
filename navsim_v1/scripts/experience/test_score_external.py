#!/usr/bin/env python
"""EWM-JEPA Step 2 - Verify score_external reproduces native scores.

For each of N scenes:
  1. native forward -> proposals, pdm_score (ground truth)
  2. score_external_trajectories(image_feature, ego_status, proposals)
     -> must reproduce argmax(pdm_score) in >=99% of scenes and have high
     score correlation
  3. control: score the same proposals with reversed pose order
     -> argmax agreement should drop (proves conditioning on the supplied
     poses, not on stale cached slots)

Run from navsim_v1 root on a GPU box:

    python scripts/experience/test_score_external.py \
        --checkpoint $NAVSIM_EXP_ROOT/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl.ckpt \
        --navsim_log_path $NAVSIM_LOGS/navtrain \
        --sensor_blobs_path $TV_BLOBS --n_scenes 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))

import torch  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--navsim_log_path", required=True)
    p.add_argument("--sensor_blobs_path", required=True)
    p.add_argument("--n_scenes", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    import os
    os.chdir(NAVSIM_V1_ROOT)

    # reuse the agent builder + scene loader from export_latents
    sys.path.insert(0, str(NAVSIM_V1_ROOT / "scripts" / "experience"))
    from export_latents import build_agent, build_scene_loader

    from navsim.agents.drive_jepa_perception_based.experience.world_model import (
        score_external_trajectories,
        native_image_feature,
    )

    agent = build_agent(args.checkpoint, "drive_jepa_perception_based_agent",
                        ["agent.config.latent=False"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    scene_loader = build_scene_loader(
        Path(args.navsim_log_path), Path(args.sensor_blobs_path), agent.get_sensor_config()
    )
    feature_builders = agent.get_feature_builders()

    n_match = 0
    n_all_match = 0
    n_total = 0
    n_ctrl_match = 0
    corrs = []
    tokens_tried = 0
    for token in scene_loader.tokens:
        if n_total >= args.n_scenes:
            break
        try:
            scene = scene_loader.get_scene_from_token(token)
            agent_input = scene.get_agent_input()
        except Exception:
            continue
        tokens_tried += 1
        feats = {}
        for b in feature_builders:
            feats.update(b.compute_features(agent_input))
        batch = {k: v[None].to(device) for k, v in feats.items()}

        # native forward (on a fresh copy -- forward mutates the dict)
        with torch.no_grad():
            out = agent({k: v.clone() for k, v in batch.items()})
        pdm_native = out["pdm_score"][0].cpu().numpy()          # (32,)
        proposals = out["proposals"]                          # (1,32,8,3)

        # external path from cached latents
        with torch.no_grad():
            image_feature, ego_last = native_image_feature(agent._pad_model, batch)
            logit_ext, pdm_ext = score_external_trajectories(
                agent._pad_model, image_feature, ego_last, proposals, mode="last"
            )
            _, pdm_all = score_external_trajectories(
                agent._pad_model, image_feature, ego_last, proposals, mode="all"
            )
        pdm_ext = pdm_ext[0].cpu().numpy()

        arg_native = int(pdm_native.argmax())
        arg_ext = int(pdm_ext.argmax())
        n_match += int(arg_native == arg_ext)
        n_all_match += int(pdm_all[0].cpu().numpy().argmax() == arg_native)
        corrs.append(np.corrcoef(pdm_native, pdm_ext)[0, 1])

        # control: reversed-pose proposals should NOT reproduce the argmax
        tr_rev = torch.flip(proposals, dims=[2])
        with torch.no_grad():
            _, pdm_rev = score_external_trajectories(
                agent._pad_model, image_feature, ego_last, tr_rev, mode="last"
            )
        n_ctrl_match += int(pdm_rev[0].cpu().numpy().argmax() == arg_native)
        n_total += 1

    agree = n_match / max(n_total, 1)
    agree_all = n_all_match / max(n_total, 1)
    ctrl = n_ctrl_match / max(n_total, 1)
    mean_corr = float(np.nanmean(corrs))
    print(f"[score_external] scenes scored: {n_total} (tried {tokens_tried})")
    print(f"[score_external] own-proposal argmax agreement (mode=last): {agree*100:.1f}%  (need >=99%)")
    print(f"[score_external] own-proposal argmax agreement (mode=all):  {agree_all*100:.1f}%")
    print(f"[score_external] Pearson corr(pdm_native, pdm_ext): {mean_corr:.4f}")
    print(f"[score_external] control reversed-poses argmax agreement: {ctrl*100:.1f}% (expect low)")
    if agree < 0.99:
        print("[score_external] FAIL: agreement below 99%")
        sys.exit(1)
    print("[score_external] PASS")


if __name__ == "__main__":
    main()
