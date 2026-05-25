import torch
import torch.nn as nn

from models import register

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out

################################ GALERKIN(SRNO) ##################################
#TODO 可以看作一种特征调制的手段
@register('galerkin')
class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads # 每个注意力头的特征通道数
        self.heads = heads # 注意力头数
        self.midc = midc # 中间特征通道数

        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1) # Q,K,V投影层（通道卷积）
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()
    
    def forward(self, x, name='0'):
        B, C, H, W = x.shape # [bs, midc, h, w]
        bias = x

        # [bs, midc, h, w]->[bs, 3*midc, h, w]->[bs, h, w, 3*midc]->[bs, h*w, heads, 3*headc]
        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)
        qkv = qkv.permute(0, 2, 1, 3) # [bs, heads, h*w, 3*headc]
        q, k, v = qkv.chunk(3, dim=-1) # [bs, heads, h*w, headc]

        k = self.kln(k)
        v = self.vln(v)

        
        v = torch.matmul(k.transpose(-2,-1), v) / (H*W) # [bs, heads, headc, headc]
        v = torch.matmul(q, v) # [bs, heads, h*w, headc]
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C) # [bs, h, w, midc] midc=heads*headc

        ret = v.permute(0, 3, 1, 2) + bias # [bs, midc, h, w]
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias
    