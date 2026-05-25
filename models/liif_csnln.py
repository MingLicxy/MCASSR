import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
#TODO 引入跨尺度非局部注意力
from models.arch_ciaosr.arch_csnln import CrossScaleAttention

################################ LIIF_CSNLN ##################################
# 相比于LIIF，LIIF_CSNLN新增了多尺度非局部特征

#TODO 注意：LIIF没有学习残差；LIIF中的特征展开策略在其他方法中未被采用
@register('liif_csnln')
class LIIF_CSNLN(nn.Module):

    def __init__(self,
                 encoder_spec,
                 imnet_spec=None,
                 local_ensemble=True,
                 feat_unfold=True,
                 cell_decode=True,
                 non_local_attn=True, # 采用csnln
                 multi_scale=[2], # 这里可以是 [2,3,4]
                 ):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        #TODO 与csnln相关参数
        self.non_local_attn = non_local_attn 
        self.multi_scale = multi_scale
        
        # 创建编码器[RDN,EDSR,RCAN]
        self.encoder = models.make(encoder_spec)
        imnet_dim = self.encoder.out_dim # 64

        if imnet_spec is not None:
            imnet_in_dim = imnet_dim # 64
            if self.feat_unfold: # 特征展开
                imnet_in_dim = imnet_dim * 9
            imnet_in_dim += 2 # 相对坐标信息
            if self.cell_decode: # 像素尺寸信息
                imnet_in_dim += 2

            #TODO 由于采用csnln带来的通道维变化
            if self.non_local_attn:
                imnet_in_dim += imnet_dim * len(multi_scale)
                self.non_local_attn_dim = imnet_dim * len(multi_scale)
                # 多尺度非局部注意力（Scale-aware Attention）
                self.cs_attn = CrossScaleAttention(channel=imnet_dim, scale=multi_scale)

            # 创建解码器MLP，out_dim=3
            #TODO 针对WireNet
            #self.imnet = models.make(imnet_spec, args={'in_features': imnet_in_dim})
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

        



    def gen_feat(self, inp): # [1, 3, 135, 180]
        self.inp = inp
        self.feat = feat = self.encoder(inp)
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

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        # 无解码器直接最邻近插值（应用feat，coord）
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret
        
        # 特征图展开，以便更好地捕获局部特征（以每一个像素坐标为中心的3x3区域）
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        #TODO 集成非局部多尺度特征
        if self.non_local_attn:
            feat = torch.cat([feat, self.non_local_feat], dim=1)
            
        # 局部集合方法（找到查询点最邻近的4个隐式特征点）
        if self.local_ensemble:
            vx_lst = [-1, 1] # 定义可能的偏移方向
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2 # 定义偏移量
        ry = 2 / feat.shape[-1] / 2

        # 计算特征图中每个像素点对应的坐标，并拓展batch维度
        # [2,h,w]->[w,2,h]->[1,w,2,h]->[bs,2,h,w]
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []

        #TODO 遍历寻找与查询坐标最邻近的4个latent code位置
        for vx in vx_lst:
            for vy in vy_lst:
                # 依据偏移量调整坐标
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # 利用最邻近插值获取与查询坐标最接近的latent code及其坐标
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                # 计算查询坐标与最邻近插值坐标之间的偏移
                rel_coord = coord - q_coord
                # 相对偏移转化为绝对偏移
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # MLP解码器的输入
                inp = torch.cat([q_feat, rel_coord], dim=-1)
                #print('#################################################', inp.shape) # [8, 2304, 29]

                # 像素尺寸作为额外输入
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)
                    #print('#################################################', inp.shape) # [8, 2304, 31]

                # coord维度[bs,q,2]，q是每个样本的查询点数量
                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred) # 查询结果

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9) # 区域面积

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0

        #TODO 根据区域面积进行加权求和
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
            
            #TODO liif学习的不是残差
            # ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
            #           padding_mode='border', align_corners=False)[:, :, 0, :] \
            #           .permute(0, 2, 1)
        return ret

    def forward(self, inp, coord, cell):
        self.inp = inp
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
