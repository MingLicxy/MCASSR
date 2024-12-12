import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from torch.nn import TransformerEncoderLayer
from models import register
NEG_INF = -1000000

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




##########################------MambaIR核心(包含2D-SSM的VSSM)------#########################
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        #TODO 从mamba_ssm中导入的函数
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError


        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D


    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
##########################------MambaIR核心(包含2D-SSM的VSSM)------#########################



#########################------VSSBlock(包含VSSM的RSSB)------#########################
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input, x_size):
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x
#########################------VSSBlock(包含VSSM的RSSB)------#########################


#########################------包装VSSBlock(包装RSSB)------#########################
class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 mlp_ratio=2.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                mlp_ratio=mlp_ratio,
                d_state=16,
                input_resolution=input_resolution,
            ))

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
#########################------包装VSSBlock(包装RSSB)------#########################


##############################------MambaIR基础模块(RSSG)------#############################
class Mamba_RG(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 mlp_ratio=2.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv'):
        super(Mamba_RG, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            mlp_ratio=mlp_ratio,
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

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops
##############################------MambaIR基础模块(RSSG)------#############################



##################################------MambaIR------#################################
class Mamba_CNN_1(nn.Module):
    def __init__(self,
                 conv = default_conv, #TODO 默认卷积
                 img_size=48,
                 patch_size=1,
                 in_chans=3,
                 mam_dim=64, # mamba_branch中间层特征通道维 60 96
                 cnn_dim=64, # cnn_branch中间层特征通道维 96
                 depths= (4, 4, 4, 4), #TODO 决定FusionBlock数量 (4, 4, 4, 4)(2, 2, 2, 2)
                 n_resblocks = 12, #TODO RG中RASB的数量
                 reduction = 16,
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
        super(Mamba_CNN_1, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans

        self.mam_dim = mam_dim
        self.cnn_dim = cnn_dim
 
        self.img_size = img_size # 用于定义norm

        ###########################TODO mam_dim与cnn_dim可以不同 TODO#########################
        #TODO 由于CNN_RG和Mamba_RG都不改变特征维度，导致浅层特征提取模块的输出维度shallow_out_dim=mam_dim=cnn_dim相等
        #TODO Mamba_CNN特征提取主干的输出特征维度是fus_dim=mam_dim+cnn_dim
        #TODO 最后一层Conv需要与浅层特征进行残差连接则lasconvt_out_dim=shallow_out_dim
        self.mam_dim = mam_dim # 64
        self.cnn_dim = cnn_dim # 64
        self.fus_dim = fus_dim = mam_dim + cnn_dim # 128
        #####################################################################################

        #TODO 定义伪上采样层的输出维度
        num_feat = 64  
        self.out_dim = num_feat
        
        #TODO 隐藏层通道维拓展
        mam_hidden = mam_dim * expansion_ratio 
        cnn_hidden = cnn_dim * expansion_ratio

        self.img_range = img_range

        # 有关图像归一化预处理
        # if in_chans == 3:
        #     rgb_mean = (0.4488, 0.4371, 0.4040)
        #     self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        # else:
        #     self.mean = torch.zeros(1, 1, 1, 1)

        self.upscale = upscale # 1
        self.upsampler = upsampler # 'none'
        

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, mam_dim, 3, 1, 1)  # 64

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths) # 4
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        

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
        



        ################################# mamba_branch #################################
        ## 图像分块处理
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=mam_dim, # 64
            embed_dim=mam_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=mam_dim,
            embed_dim=mam_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        #TODO drop_rate=0
        #self.pos_drop = nn.Dropout(p=drop_rate) 

        #TODO Mamba分支主体结构
        self.mamba_branch = nn.ModuleList()
        for i_layer in range(self.num_layers): # RSSG数量
            layer = Mamba_RG(
                dim=mam_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer], # 每个RSSG中RSSB数量
                mlp_ratio=self.mlp_ratio,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.mamba_branch.append(layer)
            
        self.norm = norm_layer(self.mam_dim) # 定义归一化层 dim=64



        ################################# fusion_block #################################
        self.fusion_block = nn.ModuleList()
        for i_layer in range(self.num_layers): # 4
            # layer = nn.Sequential(
            #     FusionBlock(conv, mam_dim * 2, 1, act=nn.ReLU(True)),
            #     FusionBlock(conv, mam_dim * 2, 1, act=nn.ReLU(True)),
            #     FusionBlock(conv, mam_dim * 2, 1, act=nn.ReLU(True)),
            #     FusionBlock(conv, mam_dim * 2, 1, act=nn.ReLU(True)),
            # )
            layer = FusionBlock_att(channels=mam_dim) #64
            self.fusion_block.append(layer)

        ## fusion_block连接mamba_branch
        self.fusion_mam = nn.ModuleList()
        for i_layer in range(self.num_layers): # 4
            layer = nn.Sequential( #TODO 论文中残差连接未体现出来
                nn.LayerNorm(mam_dim),
                nn.Linear(mam_dim, mam_dim),
                nn.GELU(),
                nn.Linear(mam_dim, mam_dim),
            )
            self.fusion_mam.append(layer)
        
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
            self.conv_after_body = nn.Conv2d(mam_dim, mam_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(mam_dim, mam_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(mam_dim // 4, mam_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(mam_dim // 4, mam_dim, 3, 1, 1))



        # ------------------------- restoration module ------------------------- #
        if self.upsampler == 'none': #TODO
            self.conv_before_upsample = nn.Sequential( # 180->64
                nn.Conv2d(mam_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        elif self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(mam_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, mam_dim, num_out_ch)
        else:
            # for image denoising
            self.conv_last = nn.Conv2d(mam_dim, num_out_ch, 3, 1, 1)

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
        x_cnn = x
        x_size = (x.shape[2], x.shape[3]) # [h, w]
        x_mam = self.patch_embed(x) # [bs, seq_len, c]
        #x_mam = self.pos_drop(x)

        for i in range(self.num_layers): # 4
            x_mam = self.mamba_branch[i](x_mam, x_size) # [bs, seq_len, c]
            x_cnn = self.cnn_branch[i](x_cnn) # [bs, c, h, w]
            x_mam_res, x_cnn_res = x_mam, x_cnn # 残差

            #TODO x_mam与x_cnn的形状不同
            x_mam = self.patch_unembed(x_mam, x_size) # [bs, c, h, w]
            #TODO FusionBlock_att两个输入两个输出
            x_mam, x_cnn = self.fusion_block[i](x_mam, x_cnn)

            #TODO 最后的融合模块
            if i == (self.num_layers-1): 
                x_fus = x_mam + x_cnn   # 64 [bs, c, h, w]

            #TODO 这里应该将mamba的形状转回去
            x_mam = self.patch_embed(x_mam) # [bs, seq_len, c]

            #TODO 这里的操作是否需要可以作为变量进行研究
            x_mam = self.fusion_mam[i](x_mam) + x_mam_res
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
            #TODO 最后一层Conv
            x = self.conv_after_body(self.forward_features(x)) + x
            #TODO 相当于上采样层
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

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops
##################################------MambaIR------#################################    






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

if __name__== '__main__':
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    
    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    x = torch.randn(1, 3, 48, 48).cuda()
    # x = torch.randn(1, 3, 256, 256)

    model = Mamba_CNN_1().cuda()
    # model = SAFMN(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)

@register('mamba_cnn_1')
def make_mambair(no_upsampling=True):
    return Mamba_CNN_1()
