import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

################################ MetaSR_MC ##################################
@register('metasr_mc')
class MetaSR_MC(nn.Module):

    def __init__(self, encoder_spec):
        super().__init__()

        self.encoder = models.make(encoder_spec)
        imnet_spec = {
            'name': 'mlp',
            'args': {
                'in_dim': 3, # 注意MetaSR中MLP的输入是相对坐标与尺度因子的cat维度为3
                'out_dim': self.encoder.out_dim * 9 * 2, # Meta-SR中MLP预测的不是RGB而是滤波器权重
                'hidden_list': [256]
            }
        }
        self.imnet = models.make(imnet_spec)

    def gen_feat(self, inp, ref_lr, ref_hr):
        #TODO
        inp_fus = torch.cat((inp, ref_lr), dim=1)
        self.feat = self.encoder(inp_fus)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        feat = F.unfold(feat, 3, padding=1).view(
            feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        
        # 生成与特征图尺寸匹配的坐标网格（包含特征图每个像素中心坐标的张量）
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda()

        # 将坐标中心与像素中心对齐
        feat_coord[:, :, 0] -= (2 / feat.shape[-2]) / 2
        feat_coord[:, :, 1] -= (2 / feat.shape[-1]) / 2

        # [h,w,2]->[2,h,w]->[1,2,h,w]->[bs,2,h,w]
        feat_coord = feat_coord.permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        # 查询坐标网格（考虑像素尺寸）
        coord_ = coord.clone()
        coord_[:, :, 0] -= cell[:, :, 0] / 2
        coord_[:, :, 1] -= cell[:, :, 1] / 2
        coord_q = (coord_ + 1e-6).clamp(-1 + 1e-6, 1 - 1e-6)

        # 利用最邻近插值获取与查询坐标最接近的latent code及其坐标
        q_feat = F.grid_sample(
            feat, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        # 计算并处理相对坐标
        rel_coord = coord_ - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2] / 2
        rel_coord[:, :, 1] *= feat.shape[-1] / 2

        # 滤波器权重预测网络的输入inp包含坐标，像素尺寸以及缩放因子等信息
        r_rev = cell[:, :, 0] * (feat.shape[-2] / 2)
        
        #TODO 直接输入相对坐标与像素尺寸，而不使用位置编码
        inp = torch.cat([rel_coord, r_rev.unsqueeze(-1)], dim=-1)

        bs, q = coord.shape[:2]
        # 利用MLP预测滤波器权重
        pred = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 2)
        #print('####################################', pred.shape) # [36864, 576, 3]
        #print('####################################', q_feat.contiguous().view(bs * q, 1, -1).shape) # [36864, 1, 576]
        # 对滤波器权重与查询到的latent code做批量矩阵乘法获得SR的RGB值
        pred = torch.bmm(q_feat.contiguous().view(bs * q, 1, -1), pred)
        pred = pred.view(bs, q, 2)
        return pred

    def forward(self, inp, coord, cell, ref, ref_hr):
        self.inp = inp
        self.gen_feat(inp, ref, ref_hr)
        return self.query_rgb(coord, cell)
