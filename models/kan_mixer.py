import os
import sys
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import register
from models.arch_kan.efficient_kan import KANLinear # 内存爆炸
from models.arch_kan.fast_kan import FastKANLayer
from models.arch_kan.faster_kan import FasterKANLayer
from models.arch_kan.wave_kan import WaveKANLinear

################################ KAN-Mixer(COZ) ##################################  
class FeedForward_sr64(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.0):
        super().__init__()

        # FastKAN超参
        grid_min: float = -2.
        grid_max: float = 2.
        num_grids: int = 8
        use_base_update: bool = True
        base_activation = F.silu
        spline_weight_init_scale: float = 0.1

        self.net = nn.Sequential(
            FastKANLayer(
                    dim, 
                    2*hidden_dim, 
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                ),
            FastKANLayer(
                    2*hidden_dim, 
                    hidden_dim, # 256
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                )
        )
    def forward(self, x):
        return self.net(x)



class MixerBlock_noLnorm_sr64(nn.Module):

    def __init__(self, dim, num_patch,  dropout = 0.):
        super().__init__()

        self.rearrange = Rearrange('b n d -> b d n')

        # token-mixing MLPs
        self.token_mix = nn.Sequential(
            # nn.LayerNorm(dim),
            # Rearrange('b n d -> b d n'),
            FeedForward_sr64(num_patch+16, num_patch, dropout), # 16+16=32->16
            Rearrange('b d n -> b n d')
        )
        
        # channel-mixing MLPs
        self.channel_mix = nn.Sequential(
            # nn.LayerNorm(dim),
            FeedForward_sr64(dim+6, dim, dropout),
        )

    #TODO 还原MSMM和QMM的计算逻辑
    def forward(self, x, coord, rel_cell,scale):
        tmp = self.rearrange(x) # [bs*q, c, 16]
        tmp = torch.cat([tmp,rel_cell],dim=-1) # [bs*q, 16*k, 32]
        x = x + self.token_mix(tmp) # [bs*q, 16*k, 32]->[bs*q, 16*k, 16]->[bs*q, 16, 16*k]

        tmp = torch.cat([x,coord,scale],dim=-1) # [bs*q, 16, c+6]
        x = x + self.channel_mix(tmp) # # [bs*q, 16, c+6]-># [bs*q, 16, c]

        return x

    
    
@register('KAN-mixer-all-no_norm-sr64-localE')
class KANMixer_NoNorm_all_sr64_localE(nn.Module):

    def __init__(self, dim, num_patch, out_dim, hidden_list,
                 depth=3, token_dim=2, channel_dim=256):
        super().__init__() # num_patch=16

        self.mixer_blocks = nn.ModuleList([])

        # depth决定网络层数
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock_noLnorm_sr64(dim, num_patch))
        
        # self.mlp_head = MLP(in_dim=dim, out_dim=channel_dim//2, hidden_list = hidden_list)

        # 线性投影层
        self.out = nn.Linear(dim,out_dim)

        # self.out =  FastKANLayer(
        #                 dim, 
        #                 out_dim, # 256
        #                 grid_min=grid_min,
        #                 grid_max=grid_max,
        #                 num_grids=num_grids,
        #                 use_base_update=use_base_update,
        #                 base_activation=base_activation,
        #                 spline_weight_init_scale=spline_weight_init_scale,
        #         )

    # inp: [bs*q, 16, c] rel_coord: [bs*q, 16, 5] meta_mix: [bs*q, 16*k, 16] rel_cell: [bs*q, 16, 1] pred: [bs, q, 16, m=3]
    def forward(self, x,coord,rel_cell,scale):

        res = x
        for mixer_block in self.mixer_blocks:
            #TODO 每层之间有残差连接
            x =res+ mixer_block(x,coord ,rel_cell,scale)

        # x += res
        # x = self.layer_norm(x)
        # x = x.mean(dim=1)
        return self.out(x)
    
    
    



    
    
    

    
    