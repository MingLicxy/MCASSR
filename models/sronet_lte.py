import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.galerkin import simple_attn
from models import register
from utils import make_coord

import numpy as np

################################ LTE ##################################
@register('sronet_lte')
class SRNO_LTE(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=64, galer_dim=256,  heads=16):
        super().__init__()  

        self.hidden_dim = hidden_dim # lte调制特征的隐藏维度
        self.galer_dim = galer_dim # galerkin聚合特征的输入维度
        # 编码器    
        self.encoder = models.make(encoder_spec)

        # 处理feat
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1) # encoder.out_dim=64
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)

        # 处理cell
        self.phase = nn.Linear(2, hidden_dim//2, bias=False) # 2->128      

        #TODO 通道卷积（有关这里通道维设计还有待研究）
        #self.conv00 = nn.Conv2d(256*4, self.hidden_dim, 1) # 通道卷积 
        #self.conv00 = nn.Conv2d(self.hidden_dim*4, self.galer_dim, 1)
        self.conv00 = nn.Conv2d((self.hidden_dim+2)*4+2, self.galer_dim, 1)

        # 利用galerkin注意力进行特征调制/聚合
        self.conv0 = simple_attn(self.galer_dim, heads) # 256, 16, 256/16=16
        self.conv1 = simple_attn(self.galer_dim, heads)

        # 解码器
        #self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})
        #TODO 解码器（SRNO中都喜欢通道卷积代替MLP）
        self.fc1 = nn.Conv2d(self.galer_dim, 256, 1)
        self.fc2 = nn.Conv2d(256, 3, 1)

    def gen_feat(self, inp):
        self.inp = inp # [16, 3, 48, 48]
        self.feat = self.encoder(inp) # [16, 64, 48, 48]

        #TODO 这里的特征提取是否可以在频域上进行
        self.coeff = self.coef(self.feat) # [16, 256, 48, 48]
        self.freqq = self.freq(self.feat) # [16, 256, 48, 48]
        return self.feat

    def query_rgb(self, coord, cell=None):
        inp = self.inp
        feat = self.feat
        coef = self.coeff
        freq = self.freqq

        # 获取LR对应坐标网格
        self.feat_coord = make_coord(inp.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:]) # [16, 2, 48, 48]

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6 

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord

        rel_coords = []
        mixfeats = []
        areas = []

        # 获取局部聚合范围内的相对坐标，调制特征以及区域面积
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone() # [16, 48, 48, 2]

                #TODO 适应'sr-implicit-downsampled-fast'的数据维度安排
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # 利用最邻近插值获取与查询坐标最接近的latent code及其坐标
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1),
                    mode='nearest', align_corners=False).permute(0, 2, 3, 1) # [16, 256, 48, 48]->[16, 48, 48, 256]
                q_freq = F.grid_sample(
                    freq, coord_.flip(-1),
                    mode='nearest', align_corners=False).permute(0, 2, 3, 1)# [16, 48, 48, 256]
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1),
                    mode='nearest', align_corners=False).permute(0, 2, 3, 1)# [16, 48, 48, 2]
                
                # 相对坐标
                rel_coord = coord - q_coord # [16, 48, 48, 2]
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]
                rel_coords.append(rel_coord) #TODO 循环之外不一定用得到
                

                # 像素尺寸（在循环之外也能获取）
                rel_cell = cell.clone() # [16, 2]
                rel_cell[:,0] *= feat.shape[-2]
                rel_cell[:,1] *= feat.shape[-1]
                
                #TODO basis generation 核心代码  
                bs, h, w = coord.shape[:3]   # bs=16 h=w=48

                # 将q_freq在最后一个维度上每两个元素进行分割，然后在新增加的最后一个维度上堆叠这些分割的张量
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1) # [16, 48, 48, 2, 128]

                #TODO 利用torch.mul()实现向量内积
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1)) # [16, 48, 48, 2, 128]
                q_freq = torch.sum(q_freq, dim=-2) # [16, 48, 48, 128] 

                #TODO [16, 2]->[16, 128]->[16, 1, 1, 128]->[16, 48, 48, 128]
                q_freq += self.phase(rel_cell).unsqueeze(1).unsqueeze(2) # [16, 48, 48, 128]
                q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1) # [16, 48, 48, 256]
                
                #TODO 逐元素相乘
                # 调制特征
                mixfeat = torch.mul(q_coef, q_freq).permute(0, 3, 1, 2)  # [16, 48, 48, 256]->[16, 256, 48, 48]            
                mixfeats.append(mixfeat)
                
                # 区域面积
                area = torch.abs(rel_coord[:, :, :, 0] * rel_coord[:, :, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        # 局部特征加权，未聚合（解码之前）
        for index, area in enumerate(areas):
            #print('#########################################', mixfeats[index].shape)#[16, 256, 48, 48]
            mixfeats[index] = mixfeats[index] * (area / tot_area).unsqueeze(1)
        

        #TODO 整合通道维（SRNO在这里进行相对坐标与像素尺度的注入太过随便，mixfeats采用LTE的特征调制策略）
        #grid = torch.cat(mixfeats, dim=1) # [16, self.hidden_dim*4, 48, 48]
        #TODO 注意训练时h==w但是测试时不一定，不要把h和w的顺序搞混
        rel_coords = [tensor.permute(0, 3, 1, 2) for tensor in rel_coords] # [16, 48, 48, 2]->[16, 2, 48, 48]
        grid = torch.cat([*rel_coords, *mixfeats, \
            rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1,1,coord.shape[1],coord.shape[2])],dim=1)
        x = self.conv00(grid)

        
        
         #TODO galerkin注意力由于调制特征
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
        # inp:[16, 3, 48, 48]
        # coord:[16, 48, 48, 2]
        # cell:[16, 2]
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

# if __name__ == '__main__':
    
    

#     x = torch.randn((16, 2304, 3)).cuda()
#     x = model(x)
#     print(x.shape)
#     print(model)