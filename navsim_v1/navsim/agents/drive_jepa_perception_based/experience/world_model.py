"""EWM-JEPA helpers: score ARBITRARY trajectories on a cached image_feature.

The native scorer only sees ``proposal_feature`` = per-proposal max-pooled BEV
slots (Scorer.forward, scorer.py:53). A trajectory reaches the latent only
through the shared ``Bev_refiner`` rounds: its poses serve as the deformable
sampling reference points (bev_refiner.py:107-114). So scoring an external
trajectory means re-running the refiner stack with ``pose=tau_ext`` while
keeping ``image_feature`` fixed -- the backbone never needs to be re-run.

``score_external_trajectories`` reproduces that path:

    image_feature (cached backbone output)
    + bev0 = hist_encoding(ego_status) + init_feature
    -> for each shared Traj_refiner: bev = Bev_refiner(tau_ext, bev, image_feature)
    -> proposal_feature = amax over pose slots -> pred_score MLP -> (B,N,6) logits

``pack_image_feature`` rebuilds the backbone output tuple from the cached
tensors stored by ``scripts/experience/export_latents.py`` (feat_flatten,
lidar2img, img_shape), so external scoring works fully offline per scene.
"""

from typing import Optional, Tuple

import torch

# fixed geometry of the frozen ViT-L JEPA grid (simple_image_encoder.py:49-64)
IMG_GRID_H, IMG_GRID_W = 16, 32


def pack_image_feature(
    feat_flatten: torch.Tensor,
    lidar2img: torch.Tensor,
    img_shape: torch.Tensor,
    grid_hw: Tuple[int, int] = (IMG_GRID_H, IMG_GRID_W),
):
    """Rebuild the backbone output tuple expected by ``Bev_refiner``.

    feat_flatten: (num_cam, H*W, B, D) backbone spatial tokens.
    lidar2img:    (B, num_cam, 4, 4) ego->image projection (post slice [:,1:2]).
    img_shape:    raw feature-dict ``img_shape`` tensor (used by point_sampling).
    """
    spatial_shape = torch.as_tensor([grid_hw], dtype=torch.long, device=feat_flatten.device)
    level_start_index = torch.cat(
        (spatial_shape.new_zeros(1), spatial_shape.prod(1).cumsum(0)[:-1])
    )
    img_metas = {"lidar2img": lidar2img, "img_shape": img_shape}
    return feat_flatten, spatial_shape, level_start_index, {"img_metas": img_metas}


@torch.no_grad()
def score_external_trajectories(
    model,
    image_feature,
    ego_status_last: torch.Tensor,
    trajectories: torch.Tensor,
    chunk: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Score externally supplied trajectories with a frozen DriveJEPAModel.

    Args:
        model:             DriveJEPAModel (eval mode).
        image_feature:     backbone output tuple (feat_flatten, spatial_shape,
                           level_start_index, kwargs) as returned by
                           model._backbone, or a tuple rebuilt by
                           ``pack_image_feature``.
        ego_status_last:   (B, 11) last history-frame ego features
                           [pose(3), vel(2), acc(2), command(4)]. Zero cols
                           1:3 yourself first if the model was trained b2d.
        trajectories:      (B, N, num_poses, 3) ego-frame poses to score.

    Returns:
        pred_logit:  (B, N, 6) raw scorer logits [NC,DAC,EP,TTC,Comfort,final]
        pdm_score:   (B, N) sigmoid of the last channel

    Trajectories are scored in chunks of ``chunk`` proposals (the BEV slot
    count is fixed at proposal_num*num_poses). When N < chunk the batch is
    padded with the last trajectory and the padding is discarded.
    """
    B, N = trajectories.shape[0], trajectories.shape[1]
    proposal_num = model._config.proposal_num
    num_poses = model.poses_num
    step = min(chunk, proposal_num)

    bev0 = model.hist_encoding(ego_status_last)[:, None] + model.init_feature.weight[None]

    logits = []
    for s in range(0, N, step):
        tr = trajectories[:, s : s + step]
        n = tr.shape[1]
        if n < step:
            pad = tr[:, -1:].expand(-1, step - n, -1, -1)
            tr = torch.cat([tr, pad], dim=1)
        if tr.shape[1] < proposal_num:
            # pad up to the fixed BEV slot count
            pad = tr[:, -1:].expand(-1, proposal_num - tr.shape[1], -1, -1)
            tr = torch.cat([tr, pad], dim=1)

        bev = bev0
        for refiner in model._trajectory_head:
            bev = refiner.Bev_refiner(tr.reshape(B, -1, model.state_size), bev, image_feature)

        proposal_feature = bev.reshape(B, proposal_num, num_poses, -1).amax(-2)
        logit = model.scorer.pred_score(proposal_feature).reshape(B, proposal_num, -1)
        logits.append(logit[:, :n])

    pred_logit = torch.cat(logits, dim=1)
    pdm_score = torch.sigmoid(pred_logit[..., -1])
    return pred_logit, pdm_score


def native_image_feature(model, features: dict):
    """Run only the backbone + ego encoding of a DriveJEPAModel.

    Returns (image_feature, ego_status_last) ready for
    ``score_external_trajectories``. Mirrors drive_jepa_model.forward lines
    48-67 without running the refiner/scorer.
    """
    ego_status = features["ego_status"][:, -1]

    if model._config.use_resnet:
        camera_feature = features["camera_feature"]
    else:
        features = dict(features)
        features["lidar2img"] = features["lidar2img"][:, 1:2]
        cam = torch.cat(
            [
                model.transform(features["camera_feature_2"])[:, None],
                model.transform(features["camera_feature_1"])[:, None],
            ],
            dim=1,
        )
        camera_feature = cam

    if model.b2d:
        ego_status = ego_status.clone()
        ego_status[:, 1:3] = 0

    image_feature = model._backbone(camera_feature, img_metas=features)
    return image_feature, ego_status
