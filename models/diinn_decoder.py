import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from models import register
from utils import make_coord


@register('diinn_decoder')
class DINNDecoder(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_list,
                 mode=1,       
                 init_q=False,
                 **kwargs):
        super().__init__()

        #TODO 决定解码器逻辑的特有超参
        self.mode = mode
        self.init_q = init_q

        #last_dim_K = in_dim * 9
        #print('#######################################', last_dim_K) # 64*9*9=5184
        last_dim_K = in_dim #TODO 有关特征展开的代码逻辑在diinn.py中
        
        if self.init_q: # 决定是否有第一层（输入坐标特征）
            self.first_layer = nn.Sequential(nn.Conv2d(4, in_dim * 9, 1),
                                            SineAct()) #TODO 这里维度一定要*9？
            last_dim_Q = in_dim * 9
        else:
            last_dim_Q = 4 #TODO [coord,cell]=4  [coord,scale]=3

        #TODO K是内容特征解码器组件，Q是坐标特征解码器组件序列
        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()

        #TODO model=[1,2,3,4]决定解码器前向传播的流程
        if self.mode == 1:
            for hidden_dim in hidden_list:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                            nn.ReLU()))
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim
                last_dim_Q = hidden_dim
        elif self.mode == 2:
            for hidden_dim in hidden_list:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                            nn.ReLU()))
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim + in_dim * 9 # 对应cat操作
                last_dim_Q = hidden_dim
        elif self.mode == 3:
            for hidden_dim in hidden_list:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                            nn.ReLU()))
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim + in_dim * 9
                last_dim_Q = hidden_dim
        elif self.mode == 4:
            for hidden_dim in hidden_list:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                            nn.ReLU()))
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim + in_dim * 9
                last_dim_Q = hidden_dim 
        if self.mode == 4:
            self.last_layer = nn.Conv2d(hidden_list[-1], out_dim, 3, padding=1, padding_mode='reflect')
        else:
            self.last_layer = nn.Conv2d(hidden_list[-1], out_dim, 1)

   

    def step(self, x, y): 
        #TODO x = self.feat [bs, 64, h, w]
        #TODO y = [rel_coord, rel_xell] [bs, 3, h, w] 
        if self.init_q:
            y = self.first_layer(y)
            x = y * x     
        if self.mode == 1:
            k = self.K[0](x)
            q = k*self.Q[0](y)        
            for i in range(1, len(self.K)):
                k = self.K[i](k)
                q = k*self.Q[i](q)
            q = self.last_layer(q)
            return q
        elif self.mode == 2:
            k = self.K[0](x)
            q = k*self.Q[0](y)
            for i in range(1, len(self.K)):
                k = self.K[i](torch.cat([k,x], dim=1))
                q = k*self.Q[i](q)
            q = self.last_layer(q)
            return q
        elif self.mode == 3:
            k = self.K[0](x)
            q = k*self.Q[0](y)
            for i in range(1, len(self.K)):
                k = self.K[i](torch.cat([q,x], dim=1))
                q = k*self.Q[i](q)
            q = self.last_layer(q)
            return q
        elif self.mode == 4:
            k = self.K[0](x)
            q = k*self.Q[0](y)
            for i in range(1, len(self.K)):
                k = self.K[i](torch.cat([q,x], dim=1))
                q = k*self.Q[i](q)
            q = self.last_layer(q)
            return q
    # 类似批量预测
    def batched_step(self, x, y, bsize):
        with torch.no_grad():
            h, w = y.shape[-2:]
            ql = 0
            preds = []
            while ql < w:
                qr = min(ql + bsize//h, w)
                pred = self.step(x[:, :, :, ql: qr], y[:, :, :, ql: qr])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=-1)
        return pred

    #TODO 解码器的输入为内容特征以及位置特征
    def forward(self, con, pos, bsize=None):
        # con: [16, 64, 48, 48]
        # pos: [16, 4, 48, 48]
        if bsize is None:
            pred = self.step(con, pos)
        else:
            pred = self.batched_step(con, pos, bsize)
        return pred
    
# 正弦激活函数
class SineAct(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)

# 局部归一化提升训练稳定性
# def patch_norm_2d(x, kernel_size=3):
#     #B, C, H, W = x.shape
#     #var, mean = torch.var_mean(F.unfold(x, kernel_size=kernel_size, padding=padding).view(B, C,kernel_size**2, H, W), dim=2, keepdim=False)
#     #return (x - mean) / torch.sqrt(var + 1e-6)
#     mean = F.avg_pool2d(x, kernel_size=kernel_size, padding=kernel_size//2)
#     mean_sq = F.avg_pool2d(x**2, kernel_size=kernel_size, padding=kernel_size//2)
#     var = mean_sq - mean**2
#     return (x-mean)/(var + 1e-6)


