import torch
import torch.nn as nn
import timm
from mmdet.models.necks.fpn import FPN

from .grid_mask import GridMask
from ..drive_jepa_config import DriveJEPAConfig


class ResNetImgEncoder(nn.Module):
    def __init__(self, config: DriveJEPAConfig, num_feature_levels=2):
        super().__init__()
        self.embed_dims = config.tf_d_model
        self.num_feature_levels = num_feature_levels
        self.num_cams = config.num_cams
        self.use_cams_embeds = True

        self.grid_mask = GridMask( True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = True

        self.img_backbone = timm.create_model(config.image_architecture, pretrained=True, features_only=True)

        self.num_outs = 1
        self.img_neck = FPN(
            in_channels=self.img_backbone.feature_info.channels()[-self.num_outs:],
            out_channels=self.embed_dims,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=self.num_outs,
            relu_before_extra_convs=True
        )
        self.level_embeds = nn.Parameter(torch.randn(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.randn([self.num_cams, self.embed_dims]))

    def forward(self, img, len_queue=None, **kwargs):

        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats[-self.num_outs:])

        # only the last (stride-32) level is used
        img_feat = img_feats[-1]
        BN, C, H, W = img_feat.size()
        feat = img_feat.view(B, int(BN / B), C, H, W)

        bs, num_cam, c, h, w = feat.shape
        spatial_shape = (h, w)
        feat = feat.flatten(3).permute(1, 0, 3, 2)  # num_cam, bs, h*w, c
        if self.use_cams_embeds:
            feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
        feat = feat + self.level_embeds[None, None, 0:1, :].to(feat.dtype)

        spatial_shape = torch.as_tensor(
            [spatial_shape], dtype=torch.long, device=feat.device)
        level_start_index = torch.cat((spatial_shape.new_zeros(
            (1,)), spatial_shape.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        return feat_flatten, spatial_shape, level_start_index, kwargs
