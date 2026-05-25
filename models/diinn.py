import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from models import register
from utils import make_coord


################################ DIINN ##################################
#TODO DIINN广泛使用卷积因此训练测试使用'-fast'的数据处理方式
@register('diinn')
class DIINN(nn.Module):
    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble # 局部聚合
        self.feat_unfold = feat_unfold # 特征展开
        self.cell_decode = cell_decode # 像素尺寸
        
        # 创建编码器[RDN,EDSR,RCAN]
        self.encoder = models.make(encoder_spec)
        # 自定义解码器
        decoder_in_dim = self.encoder.out_dim # 64
        if self.feat_unfold: # 特征展开
            decoder_in_dim *= 9
        self.decoder = models.make(imnet_spec,  args={'in_dim': decoder_in_dim})
    
    # 获取latent code
    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        #print('####################  feat.shape #######################', self.feat.shape)# [16, 64, 48, 48]
        return self.feat
    

    def query_rgb(self, coord, cell=None):
        feat = self.feat # [16, 64, 48, 48]
       

        # 特征图展开，以便更好地捕获局部特征（以每一个像素坐标为中心的3x3区域）
        if self.feat_unfold: #TODO 注意与解码器通道数对齐
            feat = F.unfold(feat, 3, padding=1).view( # [16, 64*9, 48, 48]
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
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:]) # [16, 2, 48, 48]


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
                q_feat = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)
                #print('###########################################', feat_.shape)[16, 64, 48, 48]

                q_coord = F.grid_sample(feat_coord, coord_.flip(-1), mode='nearest', align_corners=False)
                #print('###########################################', feat_.shape)[16, 2, 48, 48]

                # 相对坐标
                rel_coord = coord.permute(0, 3, 1, 2) - q_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]
                #print('#############################', rel_coord.shape)# [16, 2, 48, 48]

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:,0] *= feat.shape[-2]
                    rel_cell[:,1] *= feat.shape[-1] # [16, 2]

                    #TODO rel_cell的形状应该与q_feat相同而不是feat
                    rel_cell = rel_cell.unsqueeze(2).unsqueeze(3).expand(-1, -1, q_feat.shape[-2], q_feat.shape[-1]) # [16, 2, 48, 48]
                    
                    #print('##########################################', rel_cell.shape)
                    q_coce = torch.cat((rel_coord, rel_cell), dim=1) #TODO 注意与解码器通道数对齐 [16, 4, 48, 48] 
                
                bs, h, w = coord.shape[:3] # 16, 48, 48
                pred = self.decoder(q_feat, q_coce) #TODO 主要由通道卷积层组成 [16, 3, 48, 48]
                preds.append(pred) 

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)
        
        tot_area = torch.stack(areas).sum(dim=0)
        #print('#########################################', tot_area.shape)# [16, 48, 48]
        if self.local_ensemble:
           t = areas[0]; areas[0] = areas[3]; areas[3] = t
           t = areas[1]; areas[1] = areas[2]; areas[2] = t 
        ret = 0

        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(1) 
            
            #TODO liif学习的不是残差
            # 残差连接
            # ret = ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
            #                     padding_mode='border', align_corners=False)
        return ret

    
    def forward(self, inp, coord, cell):
        #print('####################  inp.shape #######################', inp.shape) # [16, 3, 48, 48]
        #print('####################  coord.shape #######################', coord.shape) # [16, 48, 48, 2]
        #print('####################  cell.shape #######################', cell.shape) # [16, 2]
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)




