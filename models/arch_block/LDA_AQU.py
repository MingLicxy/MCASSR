# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import xavier_init
from timm.models.layers import trunc_normal_
from einops import rearrange

"""
The code comes from https://github.com/duzw9311/LDA-AQU/blob/main/mmdet/models/necks/fpn_lau.py
paper: LDA-AQU: Adaptive Query-guided Upsampling via Local Deformable Attention
或许可用于改进CiaoSR中的特征聚合上采样
"""

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w').contiguous()


class LDA_AQU(nn.Module):
    def __init__(self, in_channels, reduction_factor=4, nh=1, scale_factor=2., k_e=3, k_u=3, n_groups=2, range_factor=11, rpb=True):
        super(LDA_AQU, self).__init__()
        self.k_u = k_u
        self.num_head = nh
        self.scale_factor = scale_factor
        self.n_groups = n_groups
        self.offset_range_factor = range_factor

        self.attn_dim = in_channels // (reduction_factor * self.num_head)
        self.scale = self.attn_dim ** -0.5
        self.rpb = rpb
        self.hidden_dim = in_channels // reduction_factor
        self.proj_q = nn.Conv2d(
            in_channels, self.hidden_dim,
            kernel_size=1, stride=1, padding=0, bias=False
        )

        self.proj_k = nn.Conv2d(
            in_channels, self.hidden_dim,
            kernel_size=1, stride=1, padding=0, bias=False
        )

        self.group_channel = in_channels // (reduction_factor * self.n_groups)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.group_channel, self.group_channel, 3, 1, 1,
                      groups=self.group_channel, bias=False),
            LayerNormProxy(self.group_channel),
            nn.GELU(),
            nn.Conv2d(self.group_channel, 2 * k_u ** 2, k_e, 1, k_e // 2)
        )
        self.layer_norm = LayerNormProxy(in_channels)

        self.pad = int((self.k_u - 1) / 2)
        base = np.arange(-self.pad, self.pad + 1).astype(np.float32)
        base_y = np.repeat(base, self.k_u)
        base_x = np.tile(base, self.k_u)
        base_offset = np.stack([base_y, base_x], axis=1).flatten()
        base_offset = torch.tensor(base_offset).view(1, -1, 1, 1)
        self.register_buffer("base_offset", base_offset, persistent=False)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.num_head, 1, self.k_u ** 2, self.hidden_dim // self.num_head))
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def extract_feats(self, x, Hout, Wout, offset, ks=3):
        B, C, Hin, Win = x.shape
        device = offset.device

        row_indices = torch.arange(Hout, device=device)
        col_indices = torch.arange(Wout, device=device)
        row_indices, col_indices = torch.meshgrid(row_indices, col_indices)
        index_tensor = torch.stack((row_indices, col_indices), dim=-1).view(1, Hout, Wout, 2)
        offset = rearrange(offset, "b (kh kw d) h w -> b kh h kw w d", kh=ks, kw=ks)
        offset = offset + index_tensor.view(1, 1, Hout, 1, Wout, 2)
        offset = offset.contiguous().view(B, ks * Hout, ks * Wout, 2)

        offset[..., 0] = (2 * offset[..., 0] / (Hout - 1) - 1)
        offset[..., 1] = (2 * offset[..., 1] / (Wout - 1) - 1)
        offset = offset.flip(-1)

        out = nn.functional.grid_sample(x, offset, mode="bilinear", padding_mode="zeros", align_corners=True)
        out = rearrange(out, "b c (ksh h) (ksw w) -> b (ksh ksw) c h w", ksh=ks, ksw=ks)
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        out_H, out_W = int(H * self.scale_factor), int(W * self.scale_factor)
        v = x
        x = self.layer_norm(x).contiguous()
        q = self.proj_q(x)
        k = self.proj_k(x)

        q = torch.nn.functional.interpolate(q, (out_H, out_W), mode="bilinear", align_corners=True)
        # q = torch.nn.functional.interpolate(q, (out_H, out_W), mode="nearest")

        q_off = q.view(B * self.n_groups, -1, out_H, out_W)
        pred_offset = self.conv_offset(q_off).contiguous()
        offset = pred_offset.tanh().mul(self.offset_range_factor) + self.base_offset.to(x.dtype)

        k = k.view(B * self.n_groups, self.hidden_dim // self.n_groups, H, W)
        v = v.view(B * self.n_groups, C // self.n_groups, H, W)
        k = self.extract_feats(k, out_H, out_W, offset=offset, ks=self.k_u)
        v = self.extract_feats(v, out_H, out_W, offset=offset, ks=self.k_u)

        q = rearrange(q, "b (nh c) h w -> b nh (h w) () c", nh=self.num_head)
        k = rearrange(k, "(b g) n c h w -> b (h w) n (g c)", g=self.n_groups)
        v = rearrange(v, "(b g) n c h w -> b (h w) n (g c)", g=self.n_groups)
        k = rearrange(k, "b n1 n (nh c) -> b nh n1 n c", nh=self.num_head)
        v = rearrange(v, "b n1 n (nh c) -> b nh n1 n c", nh=self.num_head)

        if self.rpb:
            k = k + self.relative_position_bias_table
        q = q * self.scale
        attn = q @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, "b nh (h w) t c -> b (nh c) (t h) w", h=out_H)
        return out

