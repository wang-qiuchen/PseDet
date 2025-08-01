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

from mmengine.runner import load_checkpoint

import math


@MODELS.register_module()
class ATSSSDMHeadV3(ATSSHead):
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
                 sdm_decay=0.01,
                 learnable_sdm=False,
                 domain_num: int = 4,
                 domain_name: str = "D0",
                 init_sdm_from_former = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.sdm_pos = sdm_pos
        self.num_memory = num_memory
        self.num_sdm_c = num_sdm_c
        self.num_sdm_heads = num_sdm_heads
        self.sdm_decay = sdm_decay
        self.learnable_sdm = learnable_sdm
        self.domain_num = domain_num
        self.domain_name = domain_name
        self.init_sdm_from_former = init_sdm_from_former

        num_q_c = self.feat_channels if sdm_pos == 'before' else \
                self.num_anchors * self.cls_out_channels

        # TODO check whether to perform SDM logits?before logits
        self.sdm_index = ['D0','D1','D2','D3'].index(self.domain_name) if self.domain_num == 4 else 0
        sdm_init = torch.randn(self.domain_num, self.num_memory, self.num_sdm_c)
        if self.learnable_sdm:
            self.sdm = nn.Parameter(sdm_init)
        else:
            self.register_buffer('sdm',sdm_init)

        self.k_proj = nn.ModuleList([nn.Linear(num_sdm_c, num_sdm_c, bias=False) for _ in range(domain_num)])
        self.v_proj = nn.ModuleList([nn.Linear(num_sdm_c, num_sdm_c, bias=False) for _ in range(domain_num)])
        self.q_proj = nn.ModuleList([nn.Linear(num_sdm_c, num_sdm_c, bias=False) for _ in range(domain_num)])

        pred_pad_size = self.pred_kernel_size // 2
        self.atss_cls = nn.ModuleList(
            [nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            self.pred_kernel_size,
            padding=pred_pad_size) for _ in range(self.domain_num)])
        self.atss_reg = nn.ModuleList(
            [nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 4,
            self.pred_kernel_size,
            padding=pred_pad_size) for _ in range(self.domain_num)])
        self.atss_centerness = nn.ModuleList(
            [nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 1,
            self.pred_kernel_size,
            padding=pred_pad_size) for _ in range(self.domain_num)])
        self.scales = nn.ModuleList(
            [nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides]) for _ in range(self.domain_num)]
            )

    def sdm_layer(self, feat):
        bs, num_c, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(bs, h * w, num_c)
        querys = self.q_proj[self.sdm_index](feat)
        keys = self.k_proj[self.sdm_index](self.sdm[self.sdm_index]).unsqueeze(0)
        values = self.v_proj[self.sdm_index](self.sdm[self.sdm_index]).unsqueeze(0)

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

        if not self.learnable_sdm:
            # self.sdm[self.sdm_index] = (1 - self.sdm_decay) * self.sdm[self.sdm_index] + self.sdm_decay * torch.mean(memory_write, dim=0)
            temp = self.sdm.clone()
            temp[self.sdm_index] = (1 - self.sdm_decay) * temp[self.sdm_index] + self.sdm_decay * torch.mean(memory_write, dim=0).detach()
            self.sdm = temp.clone()

        feat = feat + memory_read
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

        cls_score = self.atss_cls[self.sdm_index](cls_feat)

        if self.sdm_pos == 'after':
            cls_score = self.sdm_layer(cls_score)
        
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg[self.sdm_index](reg_feat)).float()
        centerness = self.atss_centerness[self.sdm_index](reg_feat)

        return cls_score, bbox_pred, centerness

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        return multi_apply(self.forward_single, x, self.scales[self.sdm_index])