import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from models import register



################################ PositionEncoder(CLIT/LIT) ##################################
@register('posenc')
class PositionEncoder(nn.Module):
    def __init__(
        self,
        posenc_type=None,
        complex_transform=False,
        posenc_scale=6,
        gauss_scale=1, # 为随机高斯位置编码服务
        in_dims=2,
        enc_dims=256,
        hidden_dims=32,
        head=1,
        gamma=1,
        seq_len=2304 # 序列长度
    ):
        super().__init__()

        self.posenc_type = posenc_type # 位置编码类型
        self.complex_transform = complex_transform # 复数变换
        self.posenc_scale = posenc_scale # 10
        self.gauss_scale = gauss_scale
        #TODO in_dims->hidden_dims->enc_dims->head
        self.in_dims = in_dims # 2 
        self.enc_dims = enc_dims # 64
        self.hidden_dims = hidden_dims # 64
        self.head = head
        self.gamma = gamma # 1
        self.seq_len = seq_len # 2304

        self.define_parameter()

    #TODO Rope位置编码专用函数
    # 准备阶段
    def precompute_freqs_cis(self, dim: int, seq_len: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # [128]
        t = torch.arange(seq_len, device=freqs.device) # [2304]
        freqs = torch.outer(t, freqs).float() # 计算向量外积 [2304, 128]
        #print(freqs.shape)

        # 假设 freqs = [x, y] 则reqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # 计算复数向量 [2304, 128]
        #print(freqs_cis.shape)
        return freqs_cis
        
    # 前向传播阶段
    def apply_rotary_emb(self, coords: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        coords_ = coords.float().reshape(*coords.shape[:-1], -1, 2) # [16，2304, 1, 2]
        # 最后一维的两个值分别表示复数的实部和虚部
        coords_ = torch.view_as_complex(coords_) # [16，2304, 1]
        # [16，2304, 1]*[2304, 128]->[16，2304, 128]->[16, 2304, 128, 2]->[16, 2304, 256]
        coords_out = torch.view_as_real(coords_ * freqs_cis).flatten(2)
        #print(coords_out.shape)
        return coords_out.type_as(coords)

    # torch.sinc()的重新实现
    def custom_sinc(self, x):
         return torch.where(x == 0, torch.tensor(1.0, device=x.device), torch.sin(x) / x)
    

    #TODO 依据位置编码类型定义网络层与参数
    # ['sinusoid','ipe','learn','dpb']
    def define_parameter(self):
        if self.posenc_type == 'sinusoid' or self.posenc_type == 'ipe':
            # 获取位置编码中的系数
            self.b_vals = 2.**torch.linspace(
                0, self.posenc_scale, self.enc_dims // 4
            ) - 1  # -1 -> (2 * pi) [enc_dims // 4]
            self.b_vals = torch.stack([self.b_vals, torch.zeros_like(self.b_vals)], dim=-1) # [enc_dims // 4, 2]
            self.b_vals = torch.cat([self.b_vals, torch.roll(self.b_vals, 1, -1)], dim=0) # [enc_dims // 4, 2]
            #print('##################################', self.b_vals.shape) # [128, 2]

            self.a_vals = torch.ones(self.b_vals.shape[0]) # [enc_dims // 4]
            #print('##################################', self.a_vals.shape) # [128]

            self.proj = nn.Linear(self.enc_dims, self.head) # 线性投影层

        elif self.posenc_type == 'learn': # self.in_dims=2
            self.Wr = nn.Linear(self.in_dims, self.hidden_dims // 2, bias=False)
            self.mlp = nn.Sequential(
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.Linear(self.hidden_dims, self.hidden_dims),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.Linear(self.hidden_dims, self.enc_dims)
            )
            self.proj = nn.Sequential(nn.GELU(), nn.Linear(self.enc_dims, self.head)) # 线性投影层
            self.init_weight()

        elif self.posenc_type == 'dpb':
            self.mlp = nn.Sequential(
                nn.Linear(2, self.hidden_dims), # 直接输入坐标
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.ReLU(),
                nn.Linear(self.hidden_dims, self.hidden_dims),
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.ReLU(),
                nn.Linear(self.hidden_dims, self.enc_dims)
            )
            self.proj = nn.Sequential(
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.ReLU(),
                nn.Linear(self.enc_dims, self.head) # 线性投影层
            )
        elif self.posenc_type == 'basic' or self.posenc_type == 'log_linear' or self.posenc_type == 'rff':    #TODO 随机傅里叶编码
            self.proj = nn.Linear(self.enc_dims, self.head)

            if self.posenc_type == 'log_linear':
                self.sigma = self.posenc_scale  # 超参数 1000
                self.m = self.enc_dims // 2

            elif self.posenc_type == 'rff':
                self.B = torch.randn((self.in_dims, self.enc_dims // 2)) * self.gauss_scale
        
        elif self.posenc_type == 'rope':
            self.freqs_cis = self.precompute_freqs_cis(self.enc_dims, self.seq_len, theta = 10.0) # 这里的theta是超参数
            self.proj = nn.Linear(self.enc_dims, self.head)





    # 初始化网络参数
    def init_weight(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    
    def forward(self, positions, cells=None):

        # 直接输出输入的位置坐标[batch_size, num_positions, 2]
        if self.posenc_type is None:
            return positions

        if self.posenc_type == 'sinusoid' or self.posenc_type == 'ipe':
            self.b_vals = self.b_vals.cuda()
            self.a_vals = self.a_vals.cuda()

            # 计算位置编码的正弦和余弦部分
            # b, q, 1, c (x -> c/2, y -> c/2)
            sin_part = self.a_vals * torch.sin(
                torch.matmul(positions, self.b_vals.transpose(-2, -1))
            ) # [batch_size, num_positions, self.enc_dims // 2]
            #print('##################################', sin_part.shape) # [16, 2304, 128]

            cos_part = self.a_vals * torch.cos(
                torch.matmul(positions, self.b_vals.transpose(-2, -1))
            ) # [batch_size, num_positions, self.enc_dims // 2]
            #print('##################################', cos_part.shape) # [16, 2304, 128]

            #TODO 考虑到像素单元尺寸(IPE)
            if self.posenc_type == 'ipe':
                # b, q, 2
                #bs, q = cells.shape[:2]
                cell = cells.clone()
                # cell_part = torch.sinc(
                #     torch.matmul((1 / np.pi * cell), self.b_vals.transpose(-2, -1))
                # )
                #TODO 这里的sinc()是必要的
                cell_part = self.custom_sinc(
                    torch.matmul((1 / np.pi * cell), self.b_vals.transpose(-2, -1))
                )

                sin_part = sin_part * cell_part
                cos_part = cos_part * cell_part

            # 将正弦和余弦视为复数的实部和虚部
            if self.complex_transform:
                # [batch_size, num_positions, self.enc_dims // 2]
                pos_encoding = torch.view_as_complex(torch.stack([cos_part, sin_part], dim=-1))
            else:
                # 两种输出
                # [batch_size, num_positions, self.enc_dims]
                pos_encoding = torch.cat([sin_part, cos_part], dim=-1)
                # [batch_size, num_positions, self.head]
                pos_bias = self.proj(pos_encoding)

        #TODO 学习位置编码中的系数
        elif self.posenc_type == 'learn':
            projected_pos = self.Wr(positions)

            sin_part = torch.sin(projected_pos)
            cos_part = torch.cos(projected_pos)

            if self.complex_transform:
                pos_encoding = 1 / np.sqrt(self.hidden_dims) * torch.view_as_complex(
                    torch.stack([cos_part, sin_part], dim=-1)
                )
            else:
                pos_encoding = 1 / np.sqrt(self.hidden_dims
                                           ) * torch.cat([sin_part, cos_part], dim=-1)
                pos_encoding = self.mlp(pos_encoding)

        #TODO 直接学习位置编码
        elif self.posenc_type == 'dpb':
            pos_encoding = self.mlp(positions)


        # 基础傅里叶位置编码 γ(v) = [cos(2πvv), sin(2πv)]T
        elif self.posenc_type == 'basic':  # 添加basic编码逻辑
            m = self.enc_dims // 2
            freqs = torch.ones(2, m, device=positions.device)
            proj_positions = 2 * np.pi * torch.matmul(positions, freqs)
            sin_part = torch.sin(proj_positions)
            #print('##################################', sin_part.shape) # [16, 2304, 128]
            cos_part = torch.cos(proj_positions)
            #print('##################################', cos_part.shape) # [16, 2304, 128]

            pos_encoding = torch.cat([cos_part, sin_part], dim=-1)
            
        # 正弦位置编码γ(v) = [. . . , cos(2πσ^(j/m)v), sin(2πσ^(j/m)v)]T
        elif self.posenc_type == 'log_linear':
            m = self.enc_dims // 2
            j = torch.arange(m, device=positions.device, dtype=positions.dtype)
            freqs = self.sigma ** (j / m)
            # γ(v) = [. . . , cos(2πv/σ^(j/m)), sin(2πv/σ^(j/m))]T
            freqs = 1/freqs

            proj_positions = 2 * np.pi * torch.matmul(positions, freqs.unsqueeze(0).repeat(2, 1))
            sin_part = torch.sin(proj_positions)
            #print('##################################', sin_part.shape) # [16, 2304, 128]
            cos_part = torch.cos(proj_positions)
            #print('##################################', cos_part.shape) # [16, 2304, 128]

            pos_encoding = torch.cat([cos_part, sin_part], dim=-1)
            

        #TODO 高斯随机傅里叶位置编码 γ(v) = [cos(2πBv), sin(2πBv)]T
        elif self.posenc_type == 'rff':
            B = self.B.to(positions.device) # [2, 128]
            
            proj_positions = 2 * np.pi * torch.matmul(positions, B)
            sin_part = torch.sin(proj_positions)
            #print('##################################', sin_part.shape) # [16, 2304, 128]
            cos_part = torch.cos(proj_positions)
            #print('##################################', cos_part.shape) # [16, 2304, 128]

            pos_encoding = torch.cat([cos_part, sin_part], dim=-1)
        
        elif self.posenc_type == 'rope':
            batch_size, num_positions, _ = positions.shape # 16 2304
            freqs_cis = self.freqs_cis[:num_positions].to(positions.device) # [2304, 128]
            pos_encoding = self.apply_rotary_emb(positions, freqs_cis)


        pos_bias = None if self.complex_transform else self.proj(pos_encoding)

        return pos_encoding, pos_bias
    
if __name__ == '__main__':
    model = PositionEncoder(
                posenc_type='ipe',
                complex_transform=False,
                posenc_scale=6,
                gauss_scale=1,
                in_dims=2,
                enc_dims=64,
                hidden_dims=32,
                head=8,
                gamma=1
            ).cuda().eval()
    

    pos = torch.randn((16, 2304, 2)).cuda()
    cell = torch.randn((16, 2304, 2)).cuda()
    x = model(pos, cell)
    print(x[0].shape) # [16, 2304, 256]
    print(x[1].shape) # [16, 2304, 1]