#!/usr/bin/env python
"""EWM-JEPA Step 2 - Export Drive-JEPA scene latents per scene.

Runs the frozen drive_jepa_perception_based model over a token list and saves,
per scene, one ``<log>/<token>.npz`` shard with:

    token              str
    log_name           str
    image_feature      (512, 256)  float16  backbone feat_flatten (num_cam=1,
                       H*W=512, 256) -- the scene-level JEPA latent z_t
    bev_feature        (32, 8, 256) float16  final refiner output (proposal-
                       conditioned BEV slots, scorer input)
    proposal_feature   (32, 256)   float16  amax over pose slots
    proposals          (32, 8, 3)  float32  ego-frame poses
    pred_logit         (32, 6)     float32
    pdm_score          (32,)       float32
    selected_idx       ()          int64
    trajectory         (8, 3)      float32  expert/human GT (if available)
    ego_status         (11,)       float32  last history-frame ego features
    lidar2img          (1, 4, 4)   float32  ego->cam projection (post forward slice)
    img_shape          (4,)        float32  per-cam image shape entry

Everything needed to rebuild the backbone tuple for score_external is in the
npz (image_feature + lidar2img + img_shape + ego_status).

Feature sources (mutually exclusive):
  --feature_cache_dir : Drive-JEPA training feature cache (no sensor blobs).
  --navsim_log_path   : raw navsim logs; builds features on the fly like
                        run_pdm_score (needs --sensor_blobs_path).

Run from the navsim_v1 root:

    python scripts/experience/export_latents.py \
        --checkpoint $NAVSIM_EXP_ROOT/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl.ckpt \
        --navsim_log_path $NAVSIM_LOGS/navtrain \
        --sensor_blobs_path $TV_BLOBS \
        --out_dir $NAVSIM_EXP_ROOT/experience/latents_navtrain --batch_size 8
"""

import argparse
import csv
import os
import sys
import time
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
    p.add_argument("--io_workers", type=int, default=8,
                   help="threads preparing scenes (decode/features) per export proc")
    return p.parse_args()


def build_agent(checkpoint: str, agent_config: str, agent_overrides: List[str]):
    """Instantiate the agent exactly like run_pdm_score does (hydra compose)."""
    import hydra
    from hydra.utils import instantiate
    from hydra.core.global_hydra import GlobalHydra

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


def iter_cache_items(cache_dir: Path, feature_names: List[str]) -> List[Tuple[str, str, Path]]:
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.navsim_log_path and not args.sensor_blobs_path:
        raise ValueError("--sensor_blobs_path is required with --navsim_log_path")

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

    # ---- capture backbone output + scorer input without touching the model ----
    captured: Dict[str, torch.Tensor] = {}
    captured_meta: Dict[str, torch.Tensor] = {}

    def _backbone_hook(module, inputs, output):
        # output = (feat_flatten (num_cam,HW,B,D), spatial_shape, level_start, kwargs)
        # feat_flatten (num_cam=1, HW, B, D) -> (B, HW, D)
        captured["image_feature"] = output[0][0].permute(1, 0, 2).detach()
        # keep per-sample projection params for score_external reconstruction
        img_metas = output[3]["img_metas"]
        captured_meta["lidar2img"] = img_metas["lidar2img"].detach()
        ish = img_metas["img_shape"]
        captured_meta["img_shape"] = ish.detach() if torch.is_tensor(ish) else ish

    def _scorer_hook(module, inputs, output):
        # Scorer.forward(proposals, bev_feature): bev_feature (B, P*T, D)
        bev_feature = inputs[1]
        captured["tokens"] = bev_feature.reshape(
            bev_feature.shape[0], proposal_num, num_poses, d_model
        ).detach()

    agent._pad_model._backbone.register_forward_hook(_backbone_hook)
    agent._pad_model.scorer.register_forward_hook(_scorer_hook)

    token_filter: Optional[set] = None
    if args.token_list:
        token_filter = {l.strip() for l in open(args.token_list) if l.strip()}

    records: List[dict] = []
    n_skipped = 0
    skipped_tokens = []

    def run_batch(feature_list, meta_list):
        with torch.no_grad():
            batch = collate(feature_list, device)
            out = agent(batch)
            image_feature = captured["image_feature"]  # (B, 512, D)
            tokens_feat = captured["tokens"]           # (B, P, T, D)
            lidar2img = captured_meta["lidar2img"]     # (B, 1, 4, 4)
            img_shape = captured_meta["img_shape"]     # (B, ...)
        proposal_feature = tokens_feat.amax(-2)        # (B, P, D)
        proposals = out["proposals"].float().cpu().numpy()
        pred_logit = out["pred_logit"].float().cpu().numpy()
        pdm_score = out["pdm_score"].float().cpu().numpy()
        selected = pdm_score.argmax(axis=1)
        traj = out["trajectory"].float().cpu().numpy()

        for i, meta in enumerate(meta_list):
            assert np.allclose(traj[i], proposals[i, selected[i]], atol=1e-5), \
                f"trajectory != proposals[argmax] for {meta['token']}"
            save = dict(
                token=np.str_(meta["token"]),
                log_name=np.str_(meta["log_name"]),
                image_feature=image_feature[i].half().cpu().numpy(),
                bev_feature=tokens_feat[i].half().cpu().numpy(),
                proposal_feature=proposal_feature[i].half().cpu().numpy(),
                proposals=proposals[i].astype(np.float32),
                pred_logit=pred_logit[i].astype(np.float32),
                pdm_score=pdm_score[i].astype(np.float32),
                selected_idx=np.int64(selected[i]),
                ego_status=feature_list[i]["ego_status"][-1].float().cpu().numpy().astype(np.float32),
                lidar2img=lidar2img[i].float().cpu().numpy().astype(np.float32),
                img_shape=np.asarray(img_shape[i], dtype=np.float32)
                if not torch.is_tensor(img_shape[i]) else img_shape[i].float().cpu().numpy().astype(np.float32),
            )
            if meta.get("trajectory") is not None:
                save["trajectory"] = np.asarray(meta["trajectory"], dtype=np.float32)
            from navsim.agents.drive_jepa_perception_based.experience.records import save_npz
            shard = out_dir / meta["log_name"]
            shard.mkdir(exist_ok=True)
            save_npz(shard / f"{meta['token']}.npz", **save)
            records.append(dict(
                token=meta["token"], log_name=meta["log_name"],
                file=f"{meta['log_name']}/{meta['token']}.npz",
                selected_idx=int(selected[i]),
                pdm_selected=float(pdm_score[i, selected[i]]),
                max_pdm_score=float(pdm_score[i].max()),
            ))

    def scene_done(token, log_name):
        return (out_dir / log_name / f"{token}.npz").exists()

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
            if scene_done(token, log_name):
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

        consec_fail = 0
        prof = {"load": 0.0, "input": 0.0, "feat": 0.0, "fwd": 0.0, "n": 0}
        t_loop_start = time.time()

        def prepare(token):
            """IO/CPU stage (runs in worker threads): scene -> agent_input ->
            CPU features. GPU forward stays on the main thread."""
            scene = scene_loader.get_scene_from_token(token)
            log_name = scene.scene_metadata.log_name
            if scene_done(token, log_name):
                return ("done", token, log_name, None, None)
            agent_input = scene.get_agent_input()
            f: Dict[str, torch.Tensor] = {}
            for b in feature_builders:
                f.update(b.compute_features(agent_input))
            meta = {"token": token, "log_name": log_name}
            try:
                meta["trajectory"] = scene.get_future_trajectory(
                    num_trajectory_frames=config.num_poses
                ).poses
            except Exception:
                pass
            return ("ok", token, log_name, f, meta)

        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        n_io = args.io_workers
        print(f"[export] io_workers={n_io} batch={args.batch_size}", flush=True)
        pool = ThreadPoolExecutor(max_workers=n_io)
        pending = deque()           # (future, token) in submission order
        tok_iter = iter(tokens)
        exhausted = False
        pbar = tqdm(total=len(tokens), desc="export(scenes)")
        max_pending = max(2 * args.batch_size, n_io * 2)

        def fill():
            nonlocal exhausted
            while not exhausted and len(pending) < max_pending:
                try:
                    t = next(tok_iter)
                except StopIteration:
                    exhausted = True
                    break
                pending.append((pool.submit(prepare, t), t))

        fill()
        while pending:
            fu, token = pending.popleft()
            _t = time.time()
            try:
                status, token, log_name, f, meta = fu.result()
                if status == "done":
                    pbar.update(1)
                    fill()
                    continue
            except Exception as e:
                n_skipped += 1
                skipped_tokens.append((token, str(e)[:120]))
                consec_fail += 1
                if consec_fail >= 50:
                    raise RuntimeError(
                        f"{consec_fail} consecutive scenes failed "
                        f"(last: {e}) -- systemic problem (maps? paths?), "
                        f"not sparse missing frames"
                    )
                print(f"[export] WARNING: skipping {token}: {e}")
                pbar.update(1)
                fill()
                continue
            consec_fail = 0
            prof["input"] += time.time() - _t
            feats.append(f)
            metas.append(meta)
            if len(feats) >= args.batch_size:
                _t = time.time()
                flush()
                torch.cuda.synchronize() if device.type == "cuda" else None
                prof["fwd"] += time.time() - _t
            prof["n"] += 1
            pbar.update(1)
            fill()
            if prof["n"] % 100 == 0:
                print(
                    f"[profile] n={prof['n']} wall={(time.time() - t_loop_start) / prof['n']:.3f}s/scene "
                    f"wait={prof['input'] / prof['n']:.3f}s fwd={prof['fwd'] / prof['n']:.3f}s",
                    flush=True,
                )
        pool.shutdown(wait=True)
        _t = time.time()
        flush()
        torch.cuda.synchronize() if device.type == "cuda" else None
        prof["fwd"] += time.time() - _t
        if prof["n"]:
            print(
                f"[profile] FINAL n={prof['n']} wall={(time.time() - t_loop_start) / prof['n']:.3f}s/scene "
                f"wait={prof['input'] / prof['n']:.3f}s fwd={prof['fwd'] / prof['n']:.3f}s",
                flush=True,
            )

    manifest = out_dir / "manifest.csv"
    fieldnames = ["token", "log_name", "file", "selected_idx", "pdm_selected", "max_pdm_score"]
    # resumable manifest: merge with existing rows
    existing = {}
    if manifest.is_file():
        with open(manifest, newline="") as fcsv:
            for row in csv.DictReader(fcsv):
                existing[row["token"]] = row
    for r in records:
        existing[r["token"]] = r
    with open(manifest, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(existing.values(), key=lambda r: r["token"]))
    print(f"[export] wrote {len(existing)} records total ({len(records)} this run) -> {out_dir}")
    if n_skipped:
        print(f"[export] skipped scenes (missing data): {n_skipped}")
        if skipped_tokens:
            tag = Path(args.token_list).stem if args.token_list else "all"
            skip_path = out_dir / f"skipped_{tag}.txt"
            with open(skip_path, "a") as fh:
                for t, err in skipped_tokens:
                    fh.write(f"{t}\t{err}\n")
            print(f"[export] skipped tokens logged -> {skip_path}")
    if n_skipped and not existing:
        raise RuntimeError(
            "every attempted scene failed -- check the warnings above "
            "(systemic problem, e.g. maps/data paths)"
        )


if __name__ == "__main__":
    main()
