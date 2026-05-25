import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import models
from models.galerkin import simple_attn
from models import register
from utils import make_coord
from models.arch_ciaosr.arch_csnln import CrossScaleAttention
from utils import show_feature_map

################################ SRONET_CSNLN(SRNO) ##################################
# 相比于SRNO，SRONET_CSNLN新增了特征展开以及多尺度非局部特征
@register('sronet_csnln')
class SRNO_CSNLN(nn.Module):

    def __init__(self,
                 encoder_spec,
                 width=256,
                 blocks=16,
                 local_ensemble=True, # SRNO的局部聚合方式不同于LIIF
                 feat_unfold=False,
                 cell_decode=True,
                 non_local_attn=True,
                 multi_scale=[2] # 这里可以是 [2,3,4]
                 ):
        super().__init__()
        self.width = width
        self.blocks = blocks
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.non_local_attn = non_local_attn # 非局部注意力
        self.multi_scale = multi_scale # 全局多尺度
        # 编码器
        self.encoder = models.make(encoder_spec)
        imnet_in_dim = imnet_dim = self.encoder.out_dim # 64
        if self.feat_unfold: # 特征展开
            imnet_in_dim = imnet_dim * 9

        imnet_in_dim += 2 # 相对坐标

        if self.non_local_attn: # csnln多尺度非局部特征
            imnet_in_dim += imnet_dim * len(multi_scale)
            self.non_local_attn_dim = imnet_dim * len(multi_scale)
            # 多尺度非局部注意力（Scale-aware Attention）
            self.cs_attn = CrossScaleAttention(channel=imnet_dim, scale=multi_scale)

        if self.local_ensemble: # 局部聚合
            imnet_in_dim *= 4

        if self.cell_decode: # 像素单元
            imnet_in_dim += 2

        print('######################################', imnet_in_dim) #(64*9+64+2)*4+2

        

        #TODO 利用1X1Conv代替MLP的动机是其更适合GPU计算（利用KAN代替Conv）
        #TODO 由于需要卷积运算所以采用适应'sr-implicit-downsampled-fast'
        #self.conv00 = nn.Conv2d((64 + 2)*4+2, self.width, 1) # 通道卷积 
        self.conv00 = nn.Conv2d(imnet_in_dim, self.width, 1) 

        self.conv0 = simple_attn(self.width, blocks) # 256, 16, 256/16=16
        self.conv1 = simple_attn(self.width, blocks)
        #self.conv2 = simple_attn(self.width, blocks)
        #self.conv3 = simple_attn(self.width, blocks)
        
        # 解码器
        self.fc1 = nn.Conv2d(self.width, 256, 1)
        self.fc2 = nn.Conv2d(256, 3, 1)
        
    def gen_feat(self, inp):
        self.inp = inp
        self.feat = feat = self.encoder(inp)
        #print('####################  feat.shape #######################', self.feat.shape)# [16, 64, 48, 48]
        B, C, H, W = feat.shape

        #TODO 计算Scale-aware Attention(csnln)聚合非局部多尺度特征
        if self.non_local_attn:
            crop_h, crop_w = 48, 48 # patch-size
            if H * W > crop_h * crop_w:
                # Fixme: generate cross attention by image patches
                self.non_local_feat = torch.zeros(B, self.non_local_attn_dim, H, W).cuda()
                for i in range(H // crop_h):
                    for j in range(W // crop_w):
                        i1, i2 = i * crop_h, ((i + 1) * crop_h if i < H // crop_h - 1 else H)
                        j1, j2 = j * crop_w, ((j + 1) * crop_w if j < W // crop_w - 1 else W)

                        padding = 3 // 2
                        pad_i1, pad_i2 = (padding if i1 - padding >= 0 else 0), (
                            padding if i2 + padding <= H else 0)
                        pad_j1, pad_j2 = (padding if j1 - padding >= 0 else 0), (
                            padding if j2 + padding <= W else 0)

                        crop_feat = feat[:, :, i1 - pad_i1:i2 + pad_i2, j1 - pad_j1:j2 + pad_j2]

                        #TODO 计算patch的非局部特征，并将结果存入self.non_local_feat_v的对应位置
                        crop_non_local_feat = self.cs_attn(crop_feat)
                        self.non_local_feat[:, :, i1:i2, j1:j2] = crop_non_local_feat[:, :,
                                                               pad_i1:crop_non_local_feat.shape[-2] - pad_i2,
                                                               pad_j1:crop_non_local_feat.shape[-1] - pad_j2]
            else:
                self.non_local_feat = self.cs_attn(feat)  # [16, 64, 48, 48]

        return self.feat
        
    def query_rgb(self, coord, cell):      
        feat = self.feat
        grid = 0

        # 获取LR对应坐标网格
        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        #print('###########################################', pos_lr.shape)[16, 2, 48, 48]


        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        #TODO # 集成非局部多尺度特征
        if self.non_local_attn:
            feat = torch.cat([feat, self.non_local_feat], dim=1)



        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        rel_coords = []
        feat_s = []
        areas = []

        # 采用类似LIIF的局部聚合
        for vx in vx_lst:
            for vy in vy_lst:

                coord_ = coord.clone()
                #TODO 适应'sr-implicit-downsampled-fast'的数据维度安排
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                
                # 利用最邻近插值获取与查询坐标最接近的latent code及其坐标
                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)
                #print('###########################################', feat_.shape)[16, 64, 48, 48]

                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)

                # 相对坐标
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]
                #print('#############################', rel_coord.shape)# [16, 2, 48, 48]

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                rel_coords.append(rel_coord) # [bs, 2, h, w]
                feat_s.append(feat_) # [bs, c, h, w]
                
        rel_cell = cell.clone()
        rel_cell[:,0] *= feat.shape[-2]
        rel_cell[:,1] *= feat.shape[-1] # [bs, q]

        tot_area = torch.stack(areas).sum(dim=0)
        #print('#########################################', tot_area.shape)# [16, 48, 48]
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        # 局部特征加权，未聚合（解码之前）
        for index, area in enumerate(areas):
            #print('#########################################', tot_area.shape)# [16, 48, 48]
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)

        # 将局部区域上的特征/相对坐标在通道维度上拼接 [bs, (c+2)*4+2, h, w]  c=64
        grid = torch.cat([*rel_coords, *feat_s, \
            rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1,1,coord.shape[1],coord.shape[2])],dim=1)

        x = self.conv00(grid) # [bs, width=256, h, w]

        #TODO galerkin注意力（引入空间非局部性）用于调制特征
        #TODO 多层注意力架构有利于高频特征学习（对注意力层数做消融）
        x = self.conv0(x, 0) # [bs, 256, h, w]
        x = self.conv1(x, 1) # [bs, 256, h, w]

        #TODO 局部聚合解码过程
        feat = x
        ret = self.fc2(F.gelu(self.fc1(feat)))
        

        # 残差连接
        ret = ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
                                padding_mode='border', align_corners=False)
        return ret

    def forward(self, inp, coord, cell):
        #print('####################  inp.shape #######################', inp.shape) # [16, 3, 48, 48]
        #print('####################  coord.shape #######################', coord.shape) # [16, 48, 48, 2]
        #print('####################  cell.shape #######################', cell.shape) # [16, 2]
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

