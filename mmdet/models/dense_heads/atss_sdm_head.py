# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList, reduce_mean)
from ..task_modules.prior_generators import anchor_inside_flags
from ..utils import images_to_levels, multi_apply, unmap
from .atss_head import ATSSHead
import torch.nn.functional as F

@MODELS.register_module()
class ATSSSDMHead(ATSSHead):
    """Detection Head of `ATSS <https://arxiv.org/abs/1912.02424>`_.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        pred_kernel_size (int): Kernel size of ``nn.Conv2d``
        stacked_convs (int): Number of stacking convs of the head.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to ``dict(type='GN', num_groups=32,
            requires_grad=True)``.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Defaults to False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_centerness (:obj:`ConfigDict` or dict): Config of centerness loss.
            Defaults to ``dict(type='CrossEntropyLoss', use_sigmoid=True,
            loss_weight=1.0)``.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    """
    def __init__(self,
                 num_memory=64,
                 num_sdm_c=256,
                 num_sdm_heads=8,
                 sdm_pos='before',
                 sdm_decay=0.005,
                 learnable_sdm=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.sdm_pos = sdm_pos
        self.num_sdm_c = num_sdm_c
        self.num_sdm_heads = num_sdm_heads
        self.sdm_decay = sdm_decay
        self.learnable_sdm = learnable_sdm

        num_q_c = self.feat_channels if sdm_pos == 'before' else \
                self.num_anchors * self.cls_out_channels

        # TODO check whether to perform SDM logits?before logits
        if self.learnable_sdm:
            self.sdm = nn.Parameter(torch.randn(num_memory, num_sdm_c))
        else:
            self.register_buffer('sdm', torch.randn(num_memory, num_sdm_c))

        self.k_proj = nn.Linear(num_sdm_c, num_sdm_c, bias=False)
        self.v_proj = nn.Linear(num_sdm_c, num_sdm_c, bias=False)
        self.q_proj = nn.Linear(num_q_c, num_sdm_c, bias=False)
        
    def sdm_layer(self, feat):
        bs, num_c, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(bs, h * w, num_c)
        querys = self.q_proj(feat)
        keys = self.k_proj(self.sdm).unsqueeze(0)
        values = self.v_proj(self.sdm).unsqueeze(0)

        # multi-head
        split_size = self.num_sdm_c // self.num_sdm_heads
        querys = torch.stack(torch.split(querys, split_size, dim=-1), dim=0)  # (h, bs, HW, num_c / h)
        keys = torch.stack(torch.split(keys, split_size, dim=-1), dim=0)  # (h, bs, N, num_c / h)
        values = torch.stack(torch.split(values, split_size, dim=-1), dim=0)  # (h, bs, N, num_c / h)

        scores = torch.matmul(querys, keys.transpose(2, 3)) # (h, bs, HW, N)
        scores = scores / (self.num_sdm_c ** 0.5)

        s1 = F.softmax(scores, dim=3)       # (h, bs, HW, N) 
        s2 = F.softmax(scores, dim=2).permute(0, 1, 3, 2)       # (h, bs, N, HW)

        memory_write = torch.matmul(s2, querys)     # (h, bs, N, num_c/h)
        memory_read = torch.matmul(s1, values)      # (h, bs, HW, num_c/h)

        memory_write = torch.cat(torch.split(memory_write, 1, dim=0), dim=3).squeeze(0).squeeze(0)  # (bs,N,num_c)
        memory_read = torch.cat(torch.split(memory_read, 1, dim=0), dim=3).squeeze(0)   #(bs,HW,num_c)
        # clip_grad
        if not self.learnable_sdm and self.training:
            memory_write_gather = concat_all_gather(memory_write)
            self.sdm = (1 - self.sdm_decay) * self.sdm + self.sdm_decay * torch.mean(memory_write_gather, dim=0).detach()
        feat = feat + memory_read
        # feat = memory_read
        feat = feat.reshape(bs, h, w, num_c).permute(0, 3, 1, 2)

        return feat

    def forward_single(self, x: Tensor, scale: Scale) -> Sequence[Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        if self.sdm_pos == 'before':
            cls_feat = self.sdm_layer(cls_feat)

        cls_score = self.atss_cls(cls_feat)

        if self.sdm_pos == 'after':
            cls_score = self.sdm_layer(cls_score)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output