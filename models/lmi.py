import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


################################ LMI(COZ) ##################################    
@register('lmi')
class LMI(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True, with_area=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.with_area = with_area
        
        # 编码器
        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim # 令 k=self.encoder.out_dim

            # 解码器
            self.imnet = models.make(imnet_spec, args={'dim': imnet_in_dim, 'num_patch': 16})
        else:
            self.imnet = None

        metanet_spec = {
            'name': 'mlp',
            'args': {
                'in_dim': 33,
                'out_dim': imnet_in_dim*16, # 16*k
                'hidden_list': [imnet_in_dim] # [k]
            }
        }
        #TODO MSMM
        self.metanet = models.make(metanet_spec)


    def gen_feat(self, inp):
        #self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        # coord: [bs, q, 2] cell: [bs, q, 2]
        feat = self.feat # [bs, c, h, w]

        #TODO LMI在推理查询点RGB值时需要原始RGB值
        feat_rgb = self.inp # [bs, 3, h, w]      

        if self.local_ensemble: # 决定局部聚合范围 4X4=16
            vx_lst = [-3, -1, 1, 3]
            vy_lst = [-3, -1, 1, 3]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # 获取LR对应坐标网格
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:]) # [bs, 2, h, w]

        #TODO 存储区域面积、特征值、相对坐标和RGB值
        areas = []
        inps = []
        rel_coords = []
        inps_rgb=[]

        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6) # [bs, q, 2]

                #TODO 最邻近插值获取特征与RGB值
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1) # [bs, q, c]
                
                q_feat_rgb = F.grid_sample(
                    feat_rgb, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1) # [bs, q, 3]
                # RGB值
                inps_rgb.append(q_feat_rgb.unsqueeze(2)) # [bs, q, 3, 1]

                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1) # [bs, q, 2]
                
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1] # [bs, q, 2]
                # 特征值
                inps.append(q_feat.unsqueeze(2)) # [bs, q, c, 1]

                # 相对坐标
                rel_coords.append(rel_coord.unsqueeze(2)) # [bs, q, 2, 1]

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1]) # [bs, q]
                # 区域面积
                areas.append(area + 1e-9) # [bs, q]
            
        bs, q = coord.shape[:2]

        rel_cell = cell.clone() # [bs, q, 2]
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell = rel_cell[:,:,0:1].unsqueeze(2) # [bs, q, 1, 1]

        #TODO 将像素尺寸与相对坐标作为MSMM的输入
        rel_coords.append(rel_cell)
        meta_inp = torch.cat(rel_coords,dim=-1) # [bs, q, 2, 16+1] 4X4=16
        meta_mix = self.metanet(meta_inp.view(bs*q,-1)) # imnet输入之一 [bs*q, 16*k]

        inp = torch.cat(inps, dim=2) # imnet输入之二 [bs, q, c, 16]

        rel_cell = rel_cell.view(bs*q,1,-1).repeat(1,16,1) # imnet输入之三 [bs*q, 1, 1]->[bs*q, 16, 1]
        rel_coord = torch.cat(rel_coords[0:16],dim=2) # 取前16个元素 [bs, q, 2, 16]

        inp_rgb = torch.cat(inps_rgb,dim=2) # [bs, q, 3, 16]
        
        #TODO 将相对坐标与RGB值作为QMM的输入
        rel_coord = torch.cat([inp_rgb,rel_coord],dim=-1) # imnet输入之四 [bs, q, 5, 32]
        inp = inp.contiguous()

        #TODO 利用imnet网络进行预测
        # inp: [bs*q, 16, c] rel_coord: [bs*q, 16, 5] meta_mix: [bs*q, 16*k, 16] rel_cell: [bs*q, 16, 1] pred: [bs, q, 16, m=3]
        preds= self.imnet(inp.view(bs*q, len(inps),-1), rel_coord.view(bs*q,len(inps),-1), meta_mix.view(bs*q,-1,16), rel_cell).view(bs, q, 16,-1)
        #TODO 将结果分块成16个部分，后续再进行局部聚合
        preds = torch.chunk(preds,16,dim=2) # [bs, q, 1, m=3]

        tot_area = torch.stack(areas).sum(dim=0)

        # 同LIIF进行局部区域聚合
        if self.local_ensemble:
            for i in range(8):
                t = areas[i]; areas[i] = areas[15-i]; areas[15-i] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred.squeeze(2) * (area / tot_area).unsqueeze(-1)

        return ret # [bs, q, m=3]

    def forward(self, inp, coord, cell):
        self.inp = inp
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)