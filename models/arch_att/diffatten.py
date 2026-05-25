"""
The code come from https://github.com/Jaykef/ai-algorithms/blob/main/DIFF_Transformer.ipynb
"DIFFERENTIAL TRANSFORMER"
"""
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 128 # 通道数
n_head = 4 # 头数
n_layer = 4 # 层数
dropout = 0.2

# λinit
def lambda_init(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))

# Multi-head Differential Attention
class MultiHeadDiffAttention(nn.Module):
    def __init__(self, n_embd, n_head, layer_idx=None):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head

        #TODO λinit与层索引相关
        self.lambda_init = lambda_init(layer_idx) 

        # split qkv
        self.q1_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.q2_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k1_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k2_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, 2 * n_embd, bias=False)  # V projects to 2 * n_embd

        self.c_proj = nn.Linear(2 * n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 归一化层采用GroupNorm
        self.subln = nn.LayerNorm(2 * self.head_size, elementwise_affine=False)

        # Init λ across heads
        self.lambda_q1 = nn.Parameter(torch.randn(n_head, self.head_size) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(n_head, self.head_size) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(n_head, self.head_size) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(n_head, self.head_size) * 0.1)

    def forward(self, x):
        B, T, C = x.shape # T = hxw

        # Project x to get q1, q2, k1, k2, v  (B, n_head, T, head_size)
        q1 = self.q1_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q2 = self.q2_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k1 = self.k1_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k2 = self.k2_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, 2 * self.head_size).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_size)

        #TODO 分别计算两个注意力图
        att1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        att2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        attn_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att1 = att1.masked_fill(attn_mask == 0, float('-inf'))
        att2 = att2.masked_fill(attn_mask == 0, float('-inf'))

        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)

        # Compute λ for each head separately 重参化
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        #TODO 计算差分注意力图
        att = att1 - lambda_full * att2

        att = self.attn_dropout(att)

        print("#######################################", att.shape) # [1, 2, 1200, 1200]
        print("#######################################", v.shape) # [1, 2, 1200, 48]

        y = torch.matmul(att, v)  # [B, n_head, T, 2 * head_size]
        y = self.subln(y)
        y = y * (1 - self.lambda_init)

        y = y.transpose(1, 2).contiguous().view(B, T, 2 * C)
        y = self.resid_dropout(self.c_proj(y))
        return y
    

# Traditional Multi-head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, layer_idx=None):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 计算注意力图（改进点）
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        #  掩码操作
        attn_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(attn_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
    
# 模型测试
if __name__== '__main__':
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    
    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    x = torch.randn(1, 1200, 48)
    # x = torch.randn(1, 3, 256, 256)

    model = MultiHeadDiffAttention(48, 1, 1)
    # model = SAFMN(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)