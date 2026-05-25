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
T_MAX = 128*128 #256*256
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

        #BUG 必须成立
        assert T <= T_MAX # T_MAX = 128*128
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
        #TODO T=h*w T_MAX = 128*128 要求T<=T_MAX恒成立
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
class ResidualGroup(nn.Module):
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
        super(ResidualGroup, self).__init__()

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
class RWKVIR(nn.Module):
    def __init__(self,
                 #img_size=48,
                 #patch_size=1,
                 in_chans=3,
                 embed_dim=64, #TODO 这里的通道设置对于整个程序运行有影响 64
                 depths=(2, 2, 2, 2), # (6, 6, 6, 6, 6, 6)
                 drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='none', # 默认不使用上采样
                 resi_connection='1conv',
                 **kwargs):
        super(RWKVIR, self).__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans

        num_feat = 64
        self.img_range = img_range
         
        # 用于图像标准化
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.upscale = upscale
        self.upsampler = upsampler

        #TODO 同SwinIR
        self.out_dim = num_feat

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths) # ResidualGroup数量 4
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

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

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList()
        #TODO 模型主体结构
        for i_layer in range(self.num_layers): # ResidualGroup数量
            layer = ResidualGroup(
                dim=embed_dim,
                #input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer], # 每个ResidualGroup中RRWKVBlock数量
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                #img_size=img_size,
                #patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        
        # RWKVIR中的最后一层卷积
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- restoration module ------------------------- #
        if self.upsampler == 'none': #TODO
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        elif self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
        else:
            # for image denoising
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

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

    def forward_features(self, x):
        #x_size = (x.shape[2], x.shape[3])
        #x = self.patch_embed(x) # N,L,C
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)

        #TODO LayerNorm()接受任意张量类型但是只在最后一个维度上进行初始化
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)  # b seq_len c 
        x = x.permute(0, 3, 1, 2)
        #x = self.patch_unembed(x, x_size)

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

@register('rwkvir')
def make_rwkvir(no_upsampling=True):
    return RWKVIR()



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
    model = RWKVIR() 
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