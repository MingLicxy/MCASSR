import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

################################ CUF ##################################
@register('cuf')
class CUF(nn.Module):

    def __init__(self, encoder_spec, pe_spec, feat_unfold=True, unfold_scale=3):
        super().__init__()
        
        self.feat_unfold = feat_unfold
        self.unfold_scale = unfold_scale # K=3
        # 编码器
        self.encoder = models.make(encoder_spec)
        # 位置编码器
        self.pe_encoder = models.make(pe_spec).cuda()
        self.hidden_dim = self.encoder.out_dim # 64

        # 超网络(Hyper-network)
        hynet_spec = {
            'name': 'mlp',
            'args': {
                'in_dim': 256,  #TODO 输入维度还未确定Ce=64  [256, 128, 64]
                'out_dim': self.hidden_dim, # Ce=64
                'hidden_list': [256, 256, 256], # Ch=32 共4层 [32, 32, 32]
                'final_act': True # 输出层Rule激活
            }
        }
        self.hynet = models.make(hynet_spec)

        # 解码器
        imnet_spec = {
            'name': 'mlp',
            'args': {
                'in_dim': self.hidden_dim,  # Ce=64
                'out_dim': 3, # CUF中的解码器输出RGB值
                'hidden_list': [256]
            }
        }
        self.imnet = models.make(imnet_spec)

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat # [16, 64, 48, 48]

    def query_rgb(self, coord, cell=None):
        feat = self.feat 
        

        # 特征展开（K=3）
        if self.feat_unfold: # [16, 64*9=576, 48, 48]
           #TODO unfold_scale应当是奇数且padding=(unfold_scale-1)/2
           feat = F.unfold(feat, self.unfold_scale, padding=(self.unfold_scale-1)//2).view(
               feat.shape[0], feat.shape[1] * self.unfold_scale * self.unfold_scale, feat.shape[2], feat.shape[3])
        
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
            .permute(0, 2, 1) # [16, 2304, 64*9=576]
        q_coord = F.grid_sample(
            feat_coord, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1) # # [16, 2304, 2]

        # 计算并处理相对坐标
        rel_coord = coord_ - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2] / 2
        rel_coord[:, :, 1] *= feat.shape[-1] / 2

        # 滤波器权重预测网络的输入inp包含坐标，像素尺寸以及缩放因子等信息
        rel_cell = cell * (feat.shape[-2] / 2) # [16, 2304, 2]
        
        
        
        #################################TODO 直接输入相对坐标与像素尺寸，而不使用位置编码(需要实现CUF中特有的余弦位置编码) #####################################
        #TODO 采用'ipe'位置编码同时考虑相对坐标与像素尺寸（作为超网络的输入用于预测滤波器）
        inp, _ = self.pe_encoder(rel_coord, rel_cell) # [16, 2304, 64]  [16, 2304, 128]



        #TODO 相对坐标与像素尺寸分别'sinusoid'编码再在通道维cat
        #coord_enc, _ = self.pe_encoder(rel_coord) # [16, 2304, 64]
        #cell_enc, _ = self.pe_encoder(rel_cell) # [16, 2304, 64]
        #inp = torch.cat((coord_enc, cell_enc), dim=-1) # [16, 2304, 128]
        ###################################################################################################################################################
        


        bs, q = coord.shape[:2]
        # 利用MLP预测滤波器权重
        kernel = self.hynet(inp.view(bs * q, -1)).view(bs * q, self.hidden_dim).unsqueeze(-1) # [16*2301=36864, 64, 1]
        
        # 对滤波器权重与查询到的latent code做批量矩阵乘法获得SR的RGB值
        kernel = torch.mul(q_feat.contiguous().view(bs * q, self.hidden_dim, -1), kernel).sum(dim=-1) # [36864, 64]

        # 解码得到RGB值
        pred = self.imnet(kernel).view(bs, q, -1)
    
        return pred

    def forward(self, inp, coord, cell):
        # inp:[16, 3, 48, 48]
        # coord:[16, 2304, 2]
        # cell:[16, 2304, 2]
        
        self.inp = inp
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
