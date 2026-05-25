import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

import numpy as np

################################ LTE_MC ##################################
@register('lte_mc')
class LTE_MC(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256):
        super().__init__()        
        self.encoder = models.make(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim//2, bias=False)        

        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})

    def gen_feat(self, inp, ref_lr, ref_hr):
        self.inp = inp # 后续作为残差
        self.feat_coord = make_coord(inp.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:])
        
        inp_fus = torch.cat((inp, ref_lr), dim=1)
        self.feat = self.encoder(inp_fus)
        #TODO 这里的特征提取是否可以在频域上进行
        self.coeff = self.coef(self.feat)
        self.freqq = self.freq(self.feat)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        coef = self.coeff
        freq = self.freqq

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6 

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1) # [16, 2304, 256]
                q_freq = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1) # [16, 2304, 256]
            

                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1] # [16, 2304, 2]

                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                
                #TODO basis generation 核心代码
                bs, q = coord.shape[:2]

                # 将q_freq在最后一个维度上每两个元素进行分割，然后在新增加的最后一个维度上堆叠这些分割的张量
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1) # [16, 2304, 2, 128]

                #TODO 利用torch.mul()实现向量内积
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1)) # [16, 2304, 2, 128]
                q_freq = torch.sum(q_freq, dim=-2) # [16, 2304, 128] 

                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1) # [16, 2304, 128]
                q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1) # [16, 2304, 256]
                
                #TODO 逐元素相乘
                inp = torch.mul(q_coef, q_freq)  # [16, 2304, 256]            

                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        
        # 同LIIF的局部聚合
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return ret

    def forward(self, inp, coord, cell, ref, ref_hr):
        self.inp = inp
        self.gen_feat(inp, ref, ref_hr)
        return self.query_rgb(coord, cell)
