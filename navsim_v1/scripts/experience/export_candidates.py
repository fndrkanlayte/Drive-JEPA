#!/usr/bin/env python
"""A1 - Export Drive-JEPA candidate trajectories + candidate latents per scene.

Runs the frozen drive_jepa_perception_based model over a token list and saves,
per scene, one ``<token>.npz`` record with:

    token              str
    log_name           str
    proposals          (32, 8, 3)  float32  ego-frame poses
    proposal_feature   (32, 256)   float16  pooled candidate latent
                       (= bev_feature.reshape(B,32,8,D).amax(-2) inside Scorer)
    proposal_tokens    (32, 8, 256) float16 optional per-pose tokens (--save_tokens)
    pred_logit         (32, 6)     float32
    pdm_score          (32,)       float32
    selected_idx       ()          int64    argmax(pdm_score)
    trajectory         (8, 3)      float32  human GT (if targets available)
    ego_status         (11,)       float32  last history-frame ego features
                       [pose(3), velocity(2), acceleration(2), command(4)]

Feature sources (mutually exclusive):
  --feature_cache_dir : the Drive-JEPA training feature cache
                        ($NAVSIM_EXP_ROOT/train_drive_jepa_perception_based_cache),
                        layout <log>/<token>/drive_jepa_feature.gz (+ _target.gz).
                        No dataset/sensor blobs needed.
  --navsim_log_path   : raw navsim logs ($OPENSCENE_DATA_ROOT/navsim_logs/<split>);
                        builds features on the fly like run_pdm_score (needs
                        --sensor_blobs_path).

Run from the navsim_v1 root (the agent loads ./data/8192.npy relative to CWD):

    python scripts/experience/export_candidates.py \
        --checkpoint $NAVSIM_EXP_ROOT/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl.ckpt \
        --feature_cache_dir $NAVSIM_EXP_ROOT/train_drive_jepa_perception_based_cache \
        --out_dir $NAVSIM_EXP_ROOT/experience/navtrain_export --batch_size 8
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

NAVSIM_V1_ROOT = Path(__file__).resolve().parents[2]
if str(NAVSIM_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_V1_ROOT))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from navsim.planning.training.dataset import load_feature_target_from_pickle  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--feature_cache_dir", type=str,
                     help="Drive-JEPA feature cache root (<log>/<token>/*.gz)")
    src.add_argument("--navsim_log_path", type=str,
                     help="raw navsim logs dir (scene-loader mode)")
    p.add_argument("--sensor_blobs_path", type=str, default=None,
                   help="sensor blobs dir (required with --navsim_log_path)")
    p.add_argument("--checkpoint", type=str, required=True, help="model checkpoint (.ckpt)")
    p.add_argument("--agent_config", type=str, default="drive_jepa_perception_based_agent",
                   help="hydra agent config name under config/common/agent/")
    p.add_argument("--agent_overrides", type=str, nargs="*", default=["agent.config.latent=False"],
                   help="extra hydra overrides applied to the agent config")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_scenes", type=int, default=None)
    p.add_argument("--token_list", type=str, default=None,
                   help="optional file with one token per line")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--save_tokens", action="store_true",
                   help="also save per-pose tokens (32,8,256) float16")
    p.add_argument("--check_consistency", type=int, default=0,
                   help="re-run forward for the first N scenes and assert identical output")
    return p.parse_args()


def build_agent(checkpoint: str, agent_config: str, agent_overrides: List[str]):
    """Instantiate the agent exactly like run_pdm_score does (hydra compose)."""
    import hydra
    from hydra.utils import instantiate
    from hydra.core.global_hydra import GlobalHydra

    # Same config entrypoint as run_pdm_score.py: default_run_pdm_score pulls in
    # default_common + default_evaluation + an `agent` group we can override.
    config_dir = NAVSIM_V1_ROOT / "navsim" / "planning" / "script" / "config" / "pdm_scoring"
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(
            config_name="default_run_pdm_score",
            overrides=[f"agent={agent_config}", f"agent.checkpoint_path={checkpoint}"]
            + list(agent_overrides),
        )
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()
    return agent


def collate(feature_list: List[Dict[str, torch.Tensor]], device: torch.device) -> Dict[str, torch.Tensor]:
    keys = feature_list[0].keys()
    return {k: torch.stack([f[k] for f in feature_list]).to(device) for k in keys}


# --------------------------------------------------------------------------- #
# Feature sources
# --------------------------------------------------------------------------- #

def iter_cache_items(cache_dir: Path, feature_names: List[str]) -> List[Tuple[str, str, Path]]:
    """Return [(token, log_name, token_dir)] for scenes with all feature files."""
    items = []
    for log_dir in sorted(cache_dir.iterdir()):
        if not log_dir.is_dir():
            continue
        for token_dir in sorted(log_dir.iterdir()):
            if not token_dir.is_dir():
                continue
            if all((token_dir / f"{name}.gz").is_file() for name in feature_names):
                items.append((token_dir.name, log_dir.name, token_dir))
    return items


def load_cached_features(token_dir: Path, feature_names: List[str]) -> Dict[str, torch.Tensor]:
    feats: Dict[str, torch.Tensor] = {}
    for name in feature_names:
        feats.update(load_feature_target_from_pickle(token_dir / f"{name}.gz"))
    return feats


def load_cached_targets(token_dir: Path, target_name: str) -> Dict[str, torch.Tensor]:
    path = token_dir / f"{target_name}.gz"
    if not path.is_file():
        return {}
    return load_feature_target_from_pickle(path)


def build_scene_loader(navsim_log_path: Path, sensor_blobs_path: Path, sensor_config):
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter

    scene_filter = SceneFilter(
        num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True
    )
    return SceneLoader(
        sensor_blobs_path=Path(sensor_blobs_path),
        data_path=Path(navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )


# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.navsim_log_path and not args.sensor_blobs_path:
        raise ValueError("--sensor_blobs_path is required with --navsim_log_path")

    # The agent loads ./data/8192.npy relative to CWD -> run from navsim_v1 root.
    os.chdir(NAVSIM_V1_ROOT)

    agent = build_agent(args.checkpoint, args.agent_config, args.agent_overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    if device.type == "cpu":
        print("[export] WARNING: no CUDA device, running on CPU (slow)")

    config = agent._config
    proposal_num, num_poses, d_model = config.proposal_num, config.num_poses, config.tf_d_model

    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()
    feature_names = [b.get_unique_name() for b in feature_builders]
    target_name = target_builders[0].get_unique_name() if target_builders else None

    # ---- capture the candidate latent without touching the model file ----
    captured: Dict[str, torch.Tensor] = {}

    def _scorer_hook(module, inputs, output):
        # Scorer.forward(proposals, bev_feature): bev_feature (B, P*T, D)
        bev_feature = inputs[1]
        captured["tokens"] = bev_feature.reshape(
            bev_feature.shape[0], proposal_num, num_poses, d_model
        ).detach()

    agent._pad_model.scorer.register_forward_hook(_scorer_hook)

    # ---- token list ----
    token_filter: Optional[set] = None
    if args.token_list:
        token_filter = {l.strip() for l in open(args.token_list) if l.strip()}

    records: List[dict] = []  # manifest rows
    n_checked = 0

    def run_batch(feature_list, meta_list):
        nonlocal n_checked
        with torch.no_grad():
            batch = collate(feature_list, device)
            out = agent(batch)
            ref_out = None
            todo = min(len(meta_list), max(0, args.check_consistency - n_checked))
            if todo > 0:
                ref_out = agent(batch)
        tokens_feat = captured["tokens"]  # (B, P, T, D)
        proposal_feature = tokens_feat.amax(-2)  # (B, P, D)
        proposals = out["proposals"].float().cpu().numpy()
        pred_logit = out["pred_logit"].float().cpu().numpy()
        pdm_score = out["pdm_score"].float().cpu().numpy()
        selected = pdm_score.argmax(axis=1)
        traj = out["trajectory"].float().cpu().numpy()

        for i, meta in enumerate(meta_list):
            # selected trajectory must equal argmax(proposals) - cheap invariant
            assert np.allclose(traj[i], proposals[i, selected[i]], atol=1e-5), \
                f"trajectory != proposals[argmax] for {meta['token']}"
            save = dict(
                token=np.str_(meta["token"]),
                log_name=np.str_(meta["log_name"]),
                proposals=proposals[i].astype(np.float32),
                proposal_feature=proposal_feature[i].half().cpu().numpy(),
                pred_logit=pred_logit[i].astype(np.float32),
                pdm_score=pdm_score[i].astype(np.float32),
                selected_idx=np.int64(selected[i]),
                ego_status=feature_list[i]["ego_status"][-1].float().cpu().numpy().astype(np.float32),
            )
            if args.save_tokens:
                save["proposal_tokens"] = tokens_feat[i].half().cpu().numpy()
            if meta.get("trajectory") is not None:
                save["trajectory"] = np.asarray(meta["trajectory"], dtype=np.float32)
            from navsim.agents.drive_jepa_perception_based.experience.records import save_npz
            save_npz(out_dir / f"{meta['token']}.npz", **save)
            records.append(dict(
                token=meta["token"], log_name=meta["log_name"],
                file=f"{meta['token']}.npz", selected_idx=int(selected[i]),
                pdm_selected=float(pdm_score[i, selected[i]]),
                max_pdm_score=float(pdm_score[i].max()),
            ))
        # consistency: deterministic re-forward for the first N scenes
        if ref_out is not None:
            for i in range(min(len(meta_list), max(0, args.check_consistency - n_checked))):
                assert torch.allclose(ref_out["trajectory"][i], out["trajectory"][i], atol=1e-5), \
                    f"non-deterministic forward for {meta_list[i]['token']}"
            n_checked += todo

    if args.feature_cache_dir:
        cache_dir = Path(args.feature_cache_dir).resolve()
        items = iter_cache_items(cache_dir, feature_names)
        if token_filter is not None:
            items = [it for it in items if it[0] in token_filter]
        if args.max_scenes:
            items = items[: args.max_scenes]

        feats, metas = [], []

        def flush():
            nonlocal feats, metas
            if feats:
                run_batch(feats, metas)
                feats, metas = [], []

        for token, log_name, token_dir in tqdm(items, desc="export(cache)"):
            if (out_dir / f"{token}.npz").exists():
                continue
            f = load_cached_features(token_dir, feature_names)
            t = load_cached_targets(token_dir, target_name) if target_name else {}
            meta = {"token": token, "log_name": log_name}
            if "trajectory" in t:
                meta["trajectory"] = t["trajectory"].numpy()
            feats.append(f)
            metas.append(meta)
            if len(feats) >= args.batch_size:
                flush()
        flush()

    else:
        scene_loader = build_scene_loader(
            Path(args.navsim_log_path), Path(args.sensor_blobs_path), agent.get_sensor_config()
        )
        tokens = [t for t in scene_loader.tokens if token_filter is None or t in token_filter]
        if args.max_scenes:
            tokens = tokens[: args.max_scenes]

        feats, metas = [], []

        def flush():
            nonlocal feats, metas
            if feats:
                run_batch(feats, metas)
                feats, metas = [], []

        for token in tqdm(tokens, desc="export(scenes)"):
            if (out_dir / f"{token}.npz").exists():
                continue
            scene = scene_loader.get_scene_from_token(token)
            agent_input = scene.get_agent_input()
            f: Dict[str, torch.Tensor] = {}
            for b in feature_builders:
                f.update(b.compute_features(agent_input))
            meta = {"token": token, "log_name": scene.scene_metadata.log_name}
            try:
                meta["trajectory"] = scene.get_future_trajectory(
                    num_trajectory_frames=config.num_poses
                ).poses
            except Exception:
                pass
            feats.append(f)
            metas.append(meta)
            if len(feats) >= args.batch_size:
                flush()
        flush()

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="") as fcsv:
        writer = csv.DictWriter(
            fcsv, fieldnames=["token", "log_name", "file", "selected_idx", "pdm_selected", "max_pdm_score"]
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"[export] wrote {len(records)} records -> {out_dir} (manifest: {manifest})")


if __name__ == "__main__":
    main()
