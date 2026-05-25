import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from torch.nn import TransformerEncoderLayer
from models import register
#from torch.utils.cpp_extension import load

NEG_INF = -1000000
T_MAX = 128*128    # 512*512
# RWKV
# wkv_cuda = load(name="wkv", sources=["Arbitrary-scale/liif-main/models/rwkv_cuda/wkv_op.cpp", "Arbitrary-scale/liif-main/models/rwkv_cuda/wkv_cuda.cu"],
#                 verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])
def load_wkv_cuda(T_MAX):
    from torch.utils.cpp_extension import load
    return load(
        name="wkv", 
        sources=[
            "Arbitrary-scale/liif-main/models/rwkv_cuda/wkv_op.cpp", 
            "Arbitrary-scale/liif-main/models/rwkv_cuda/wkv_cuda.cu"
        ],
        verbose=True, 
        extra_cuda_cflags=[
            '-res-usage', 
            '--maxrregcount 60', 
            '--use_fast_math', 
            '-O3', 
            '-Xptxas -O3', 
            f'-DTmax={T_MAX}'
        ]
    )
wkv_cuda = load_wkv_cuda(T_MAX)



###############------RCAB(RCAN的基础模块)------###############
## padding=(kernel_size//2)保证Conv输入输出维度一致（在FusionBlock以及CNN_branch中使用）
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

## 用于图像归一化预处理（不采用）
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG) RCAN的基础模块
class CNN_RG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(CNN_RG, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=nn.ReLU(True),
                res_scale=1) 
            for _ in range(n_resblocks)
        ]
        # 每个CNN_RG中也有最后一层卷积
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
###############------RCAB(RCAN的基础模块)------###############


###############------Mamba-CNN融合模块（from ACT）------###############
class FusionBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=False, act=nn.ReLU(True)):
        super(FusionBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0:
                modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
###############------Mamba-CNN融合模块（from ACT）------###############




###############------Mamba-CNN注意力融合模块------###############
## SFCA多头注意力机制
class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # 为每个头设置控制参数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # k和v的投影矩阵，通道数X2
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        # 深度可分离卷积
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # 后处理
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        # k和v来自同一个输入y
        kv = self.kv_dwconv(self.kv(y))
        # k和v按通道维度上连接在一起，处理完成后沿通道维度分开
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        # 变形
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力图
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # 后处理
        out = self.project_out(out)
        return out
    
class FusionBlock_att(nn.Module):
    def __init__(self, channels):
        super(FusionBlock_att, self).__init__()
        # 频域和空域的预处理
        self.mam = nn.Conv2d(channels, channels, 3, 1, 1)
        self.cnn = nn.Conv2d(channels, channels, 3, 1, 1)
        # mam->cnn和cnn->mam互融合注意力
        self.mam_att = Attention(dim=channels)
        self.cnn_att = Attention(dim=channels)
        # nn.Sequential()用于将多个层按顺序组合在一起，形成一个"层序列"
        self.fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 3, 1, 1), nn.Conv2d(channels, 2*channels, 3, 1, 1), nn.Sigmoid())

    #TODO 两个输入对应两个输出
    def forward(self, mam, cnn):
        #ori = cnn
        mam = self.mam(mam)
        cnn = self.cnn(cnn)
        mam = self.mam_att(mam, cnn)+mam
        cnn = self.cnn_att(cnn, mam)+cnn
        fuse = self.fuse(torch.cat((mam, cnn), 1))
        mam_a, cnn_a = fuse.chunk(2, dim=1)
        cnn = cnn_a * cnn
        mam = mam * mam_a
        #res = mam + cnn
        
        #TODO 替换极端数值
        #res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        mam = torch.nan_to_num(mam, nan=1e-5, posinf=1e-5, neginf=1e-5)
        cnn = torch.nan_to_num(cnn, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return mam, cnn # mam, cnn = my_function()
###############------Mamba-CNN注意力融合模块------###############




###############------Channel_Attention_Block(CAB)------###############
class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)
###############------Channel_Attention_Block(CAB)------###############



###############------Multi_Layer_Perceptron------###############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
###############------Multi_Layer_Perceptron------###############






###############------动态位置编码(未采用)------###############
class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops
###############------动态位置编码(未采用)------###############



###############------RWKV核心模块------###############
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())
###############------RWKV核心模块------###############


class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False) 
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True) 
        

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias = False) 
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x) 
        # import pdb 
        # pdb.set_trace() 
        
        
        out = self.alpha[0]*x + self.alpha[1]*out1x1 + self.alpha[2]*out3x3 + self.alpha[3]*out5x5
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution 
        
        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2)) 
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1)) 
        
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2)) 
        
        combined_weight = self.alpha[0]*identity_weight + self.alpha[1]*padded_weight_1x1 + self.alpha[2]*padded_weight_3x3 + self.alpha[3]*self.conv5x5.weight 
        
        device = self.conv5x5_reparam.weight.device 

        combined_weight = combined_weight.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)


    def forward(self, x): 
        
        if self.training: 
            self.repram_flag = True
            out = self.forward_train(x) 
        elif self.training == False and self.repram_flag == True:
            self.reparam_5x5() 
            self.repram_flag = False 
            out = self.conv5x5_reparam(x)
        elif self.training == False and self.repram_flag == False:
            out = self.conv5x5_reparam(x)
        
        return out 


###############------空间混合------###############
class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy', 
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        
        
        self.dwconv = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False) 
        
        self.recurrence = 2 
        
        self.omni_shift = OmniShift(dim=n_embd)


        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False) 


        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd))) 
            self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd))) 



    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr

        
        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')    


        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v


    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution) 
        
        for j in range(self.recurrence): 
            if j%2==0:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v) 
            else:
                h, w = resolution 
                k = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w) 
                v = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w) 
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v) 
                k = rearrange(k, 'b (w h) c -> b (h w) c', h=h, w=w) 
                v = rearrange(v, 'b (w h) c -> b (h w) c', h=h, w=w) 
                

        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x
###############------空间混合------###############



###############------通道混合------###############
class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd



        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False) 
        
        self.omni_shift = OmniShift(dim=n_embd)
        
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)



    def forward(self, x, resolution):

        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')    


        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv 

        return x
###############------通道混合------###############


###############------RRWKVBlock------###############
class RRWKVBlock(nn.Module):
    def __init__(self,
                 n_embd: int = 0, # 通道维度
                 drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 n_layer: int = 0, #TODO
                 layer_id: int = 0, #TODO
                 hidden_rate=4,
                 init_mode='fancy',
                 key_norm=False,
                 **kwargs):
        super().__init__()

        self.layer_id = layer_id 
         
        # 初始化归一化层
        self.ln1 = norm_layer(n_embd)
        self.ln2 = norm_layer(n_embd) 

        # 初始化空间混合层与通道混合层
        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, init_mode,
                                   key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, hidden_rate,
                                   init_mode, key_norm=key_norm)

        #TODO 提高模型泛化性
        self.drop_path = DropPath(drop_path)

        #TODO 残差连接的尺度 
        self.skip_scale1= nn.Parameter(torch.ones(n_embd))
        self.skip_scale2= nn.Parameter(torch.ones(n_embd))

        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    #TODO 输入张量维度为[B, C, H, W]，说明未经过PatchEmbed()
    def forward(self, x): 
        b, c, h, w = x.shape # [B, C, H, W]
        
        resolution = (h, w)

        # x = self.dwconv1(x) + x
        x = rearrange(x, 'b c h w -> b (h w) c')

        #TODO x = x * self.skip_scale1 + self.gamma1 * self.drop_path(self.att(self.ln1(x), resolution))
        x = x + self.gamma1 * self.att(self.ln1(x), resolution) 
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        # x = self.dwconv2(x) + x
        x = rearrange(x, 'b c h w -> b (h w) c')  
        #TODO x = x * self.skip_scale2 + self.gamma2 * self.drop_path(self.ffn(self.ln2(x), resolution))   
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution) 
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x
###############------RRWKVBlock------###############



#########################------包装多层RRWKVBlock------#########################
class BasicLayer(nn.Module):
    def __init__(self,
                 dim,  # 通道维度
                 #input_resolution, # 在本项目中可能用不到
                 depth, # ResidualGroup中包含RRWKVBlock层数
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()

        self.dim = dim
        #self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(RRWKVBlock(
                n_embd=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                n_layer=depth,
                layer_id=i,
                #input_resolution=input_resolution, # 用不到
            ))

        # if downsample is not None:
        #     self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        # else:
        #     self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # if self.downsample is not None:
        #     x = self.downsample(x)
        return x
#########################------包装多层RRWKVBlock------#########################


##############################------RWKVIR基础模块------#############################
class RWKV_RG(nn.Module):
    def __init__(self,
                 dim,
                 #input_resolution,
                 depth,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 #img_size=None, # 与PatchEmbed相关参数
                 #patch_size=None,
                 resi_connection='1conv'):
        super(RWKV_RG, self).__init__()

        self.dim = dim
        #self.input_resolution = input_resolution # [64, 64]


        self.residual_group = BasicLayer(
            dim=dim,
            #input_resolution=input_resolution,
            depth=depth,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        # 每个RSSG中的最后一层卷积
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        #TODO 本项目不必分块（后续可以尝试分块）
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        # self.patch_unembed = PatchUnEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x):
        #return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x
        return self.conv(self.residual_group(x)) + x
##############################------RWKVIR基础模块------#############################



##################################------RWKVIR------#################################
class RWKV_CNN_1(nn.Module):
    def __init__(self,
                 conv = default_conv, #TODO 默认卷积
                 #img_size=48,
                 #patch_size=1,
                 in_chans=3,
                 rwkv_dim=64, # rwkv_branch中间层特征通道维 60
                 cnn_dim=64, # cnn_branch中间层特征通道维
                 #embed_dim=96, #TODO 这里的通道设置对于整个程序运行有影响
                 depths=(2, 2, 2, 2), # (6, 6, 6, 6, 6, 6)
                 n_resblocks = 12, #TODO RG中RASB的数量
                 reduction = 16, # CNN分支相关
                 mlp_ratio=2.,
                 drop_rate=0.,
                 expansion_ratio = 1, #TODO 与num_feat共同决定hidden_dim
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=1,
                 img_range=1.,
                 upsampler='none', # 默认不使用上采样
                 resi_connection='1conv',
                 **kwargs):
        super(RWKV_CNN_1, self).__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans

        self.rwkv_dim = rwkv_dim # 64
        self.cnn_dim = cnn_dim # 64
        self.fus_dim = fus_dim = rwkv_dim + cnn_dim # 128

        #TODO 定义伪上采样层的输出维度
        num_feat = 64
        self.out_dim = num_feat

        #TODO 隐藏层通道维拓展
        rwkv_hidden = rwkv_dim * expansion_ratio 
        cnn_hidden = cnn_dim * expansion_ratio
         
        # 用于图像标准化
        # self.img_range = img_range
        # if in_chans == 3:
        #     rgb_mean = (0.4488, 0.4371, 0.4040)
        #     self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        # else:
        #     self.mean = torch.zeros(1, 1, 1, 1)

        self.upscale = upscale # 1
        self.upsampler = upsampler # 'none'


        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, rwkv_dim, 3, 1, 1)  # 64

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths) # ResidualGroup数量 4
        self.pos_drop = nn.Dropout(p=drop_rate) # drop_rat=0.
        
        #TODO 与Mamba中分块操作相关
        #self.patch_norm = patch_norm
        #self.mlp_ratio = mlp_ratio


        # self.patch_embed = PatchEmbed(
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     in_chans=embed_dim,
        #     embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        
        # num_patches = self.patch_embed.num_patches
        # patches_resolution = self.patch_embed.patches_resolution
        # self.patches_resolution = patches_resolution

        # self.patch_unembed = PatchUnEmbed(
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     in_chans=embed_dim,
        #     embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)

        
        ################################ cnn_branch #################################
        self.cnn_branch = nn.ModuleList() #TODO nn.ModuleList()需要自定义forward()
        for i_layer in range(self.num_layers): # 4
            layer = CNN_RG(
                conv=conv,
                n_feat=cnn_dim, # 64 CNN_RG不改变通道维度
                kernel_size=3, 
                reduction=reduction,
                n_resblocks=n_resblocks, # 12
            )
            self.cnn_branch.append(layer)


       
        ################################# rwkv_branch #################################
        self.rwkv_branch = nn.ModuleList()
        for i_layer in range(self.num_layers): # 4
            layer = RWKV_RG(
                dim=rwkv_dim,
                depth=depths[i_layer], # 每个RSSG中RSSB数量
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                resi_connection=resi_connection)
            self.rwkv_branch.append(layer)
            
        self.norm = norm_layer(self.rwkv_dim) # 定义归一化层 dim=64



        ################################# fusion_block #################################
        self.fusion_block = nn.ModuleList()
        for i_layer in range(self.num_layers): # 4
            # layer = nn.Sequential(
            #     FusionBlock(conv, fus_dim, 1, act=nn.ReLU(True)), # 64+64=128
            #     FusionBlock(conv, fus_dim, 1, act=nn.ReLU(True)),
            #     FusionBlock(conv, fus_dim, 1, act=nn.ReLU(True)),
            #     FusionBlock(conv, fus_dim, 1, act=nn.ReLU(True)),
            # )
            layer = FusionBlock_att(channels=rwkv_dim) #64
            self.fusion_block.append(layer)

        ## fusion_block连接rwkv_branch
        self.fusion_rwkv = nn.ModuleList()
        for i_layer in range(self.num_layers): # 4
            layer = nn.Sequential( #TODO 论文中残差连接未体现出来
                nn.LayerNorm(rwkv_dim), #TODO 注意层归一化，需要通道维再最后一维
                nn.Linear(rwkv_dim, rwkv_dim),
                nn.GELU(),
                nn.Linear(rwkv_dim, rwkv_dim),
            )
            self.fusion_rwkv.append(layer)
        
        ## fusion_block连接cnn_branch
        self.fusion_cnn = nn.ModuleList()
        for i_layer in range(self.num_layers): # 4
            layer = nn.Sequential(
                conv(cnn_dim, cnn_dim, 3),
                nn.ReLU(True),
                conv(cnn_dim, cnn_dim, 3)
            )
            self.fusion_cnn.append(layer)


        ################################# last_conv(Mamba_CNN) #################################
        if resi_connection == '1conv': #TODO 最后一层Conv负责将fusion_feat的维度降低
            self.conv_after_body = nn.Conv2d(rwkv_dim, rwkv_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(rwkv_dim, rwkv_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(rwkv_dim // 4, rwkv_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(rwkv_dim // 4, rwkv_dim, 3, 1, 1))



        # ------------------------- restoration module ------------------------- #
        if self.upsampler == 'none': #TODO
            self.conv_before_upsample = nn.Sequential( # 180->64
                nn.Conv2d(rwkv_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        elif self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(rwkv_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, rwkv_dim, num_out_ch)
        else:
            # for image denoising
            self.conv_last = nn.Conv2d(rwkv_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    

    #TODO 定义浅层特征提取之后的逻辑（x是提取的浅层特征）
    def forward_features(self, x):

        #TODO 注意RWKV不用分块
        x_cnn = x_rwkv = x
        #x_rwkv = self.pos_drop(x)

        for i in range(self.num_layers): # 4
            x_rwkv = self.rwkv_branch[i](x_rwkv) # [bs, c, h, w]
            x_cnn = self.cnn_branch[i](x_cnn) # [bs, c, h, w]
            x_rwkv_res, x_cnn_res = x_rwkv, x_cnn # 残差 [1, 64, 128, 128]
            #print('####################################', x_rwkv_res.shape)

            # x_fus = torch.cat((x_rwkv, x_cnn), 1) # 沿通道维cat [bs, c, h, w]
            # x_fus = x_fus + self.fusion_block[i](x_fus) #TODO 最后一个融合模块的输出 128 [bs, c, h, w]
            #TODO FusionBlock_att两个输入两个输出
            x_rwkv, x_cnn = self.fusion_block[i](x_rwkv, x_cnn)

            #TODO 最后的融合模块（不一定是加法）
            if i == (self.num_layers-1): 
                x_fus = x_rwkv + x_cnn   # 64 [bs, c, h, w]

            #TODO 这里的操作是否需要可以作为变量进行研究（fusion_rwkv[i]有LayerNorm）
            x_rwkv = self.fusion_rwkv[i](x_rwkv.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + x_rwkv_res
            x_cnn = self.fusion_cnn[i](x_cnn) + x_cnn_res

        #BUG 是否需要这个归一化层（只接受[bs, seq_len, c]）
        x_fus = x_fus.permute(0, 2, 3, 1) # [bs, c, h, w]->[bs, h, w, c]
        x = self.norm(x_fus).permute(0, 3, 1, 2) # [bs, h, w, c]->[bs, c, h, w]

        return x


    def forward(self, x):
        #self.mean = self.mean.type_as(x)
        #x = (x - self.mean) * self.img_range

        if self.upsampler == 'none':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
        elif self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        else:
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res) # [4, 64, 48, 48]


        #x = x / self.img_range + self.mean
        #print('#########################################', x.shape)
        return x
    
##################################------RWKVIR------#################################    





#TODO 基于RWKV的超分辨率框架是不必分块的吗？
##############################------Image-to-Patchs/Patchs-to-Image------#############################
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops
##############################------Image-to-Patchs/Patchs-to-Image------#############################



##############################------Upsample------#############################
class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)
##############################------Upsample------#############################

# def buildMambaIR(upscale=2):
#     return MambaIR(img_size=64,
#                    patch_size=1,
#                    in_chans=3,
#                    embed_dim=180,
#                    depths=(6, 6, 6, 6, 6, 6),
#                    mlp_ratio=2.,
#                    drop_rate=0.,
#                    norm_layer=nn.LayerNorm,
#                    patch_norm=True,
#                    use_checkpoint=False,
#                    upscale=upscale,
#                    img_range=1.,
#                    upsampler='pixelshuffle',
#                    resi_connection='1conv')

@register('rwkv_cnn_1')
def make_rwkvir(no_upsampling=True):
    return RWKV_CNN_1()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 
if __name__ == "__main__":
    import os 
    os.environ['CUDA_VISIBLE_DEVICES']='0' 

    import time 
    from thop import profile, clever_format
    
    '''
    !!!!!!!!
    Caution: Please comment out the code related to reparameterization and retain only the 5x5 convolutional layer in the OmniShift.
    !!!!!!!!
    '''
    
    #TODO 批量B与通道C的选取是有条件的
    x=torch.zeros((1, 3, 128, 128)).type(torch.FloatTensor).cuda() 
    model = RWKV_CNN_1() 
    model.cuda() 
    
    since = time.time()
    y=model(x)
    print("time", time.time()-since) 
    
    flops, params = profile(model, inputs=(x, ))  
    flops, params = clever_format([flops, params], '%.6f') 
    print('flops',flops)
    print('params', params) 
    print(count_parameters(model)/1e6)
    # print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    # print("Params=", str(params/1e6)+'{}'.format("M"))