import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

################################ LIIF_MC ##################################
#TODO 注意：LIIF没有学习残差；LIIF中的特征展开策略在其他方法中未被采用
@register('liif_mc')
class LIIF_MC(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        
        # 创建编码器[RDN,EDSR,RCAN]
        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim # 64
            if self.feat_unfold: # 特征展开
                imnet_in_dim *= 9
            imnet_in_dim += 2 # attach coord 坐标信息
            if self.cell_decode: # 像素尺寸信息
                imnet_in_dim += 2

            # 创建解码器MLP，out_dim=3
            #TODO 针对WireNet
            #self.imnet = models.make(imnet_spec, args={'in_features': imnet_in_dim})
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp, ref_lr, ref_hr): # [1, 2, 135, 180]
        #TODO 这里ref_hr不发挥作用，利用通道连接作为模态融合方式
        inp_fus = torch.cat((inp, ref_lr), dim=1) # [1, 4, 135, 180]
        self.feat = self.encoder(inp_fus)
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
                #TODO 针对GroupKANMLP的设置
                pred = self.imnet(inp)
                #pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
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

    def forward(self, inp, coord, cell, ref, ref_hr):
        self.inp = inp 
        self.gen_feat(inp, ref, ref_hr)
        return self.query_rgb(coord, cell)
