
import torch
import math
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange
from models import register

####################################### ACT(transformer+CNN) #############################################


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )

# 对图像进行颜色通道的均值和标准差调整
class MeanShift(nn.Conv2d):
    def __init__(self,
                 rgb_range,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0),
                 sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

# 残差块
class ResBlock(nn.Module):
    def __init__(self,
                 conv,
                 n_feats,
                 kernel_size,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(True),
                 res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

# 超分上采样模块
class Upsampler(nn.Sequential):
    def __init__(self, 
                 conv, 
                 scale, 
                 n_feats, 
                 bn=False, 
                 act=False, 
                 bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

# 层归一化（稳定训练，加速收敛）
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm2(x2), **kwargs)

# 前馈神经网络FRFN
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# 自注意力
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


# 交叉注意力
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x_q, x_kv):
        _, _, dim, heads = *x_q.shape, self.heads
        _, _, dim_large = x_kv.shape 

        assert dim == dim_large

        q = self.to_q(x_q)

        q = rearrange(q, 'b n (h d) -> b h n d', h=heads)

        kv = self.to_kv(x_kv).chunk(2, dim=-1)
        
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=heads), kv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


# 通道注意力
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
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## 残差通道注意力块 Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self,
                 conv,
                 n_feat,
                 kernel_size,
                 reduction,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(True),
                 res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Group (RG) 
# 由n_resblocks个RCAB组成RG
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(conv,
                 n_feat,
                 kernel_size,
                 reduction,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(True),
                 res_scale=1)
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

#TODO Transformer与CNN的融合模块
class FB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=False, act=nn.ReLU(True)):
        super(FB, self).__init__()
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

########################################### ACT主体 ###############################################
class ACT(nn.Module):
    def __init__(self,
                 conv = default_conv,
                 task = 'none', 
                 scale = 1,
                 n_feats = 64,
                 rgb_range = 225,
                 n_colors = 3,
                 n_resgroups = 4,
                 n_resblocks = 12,
                 reduction = 16, # 
                 n_heads = 8,
                 n_layers = 8,
                 dropout_rate = 0,
                 expansion_ratio = 4,
                 n_fusionblocks = 4,
                 token_size = 3,):

        super(ACT, self).__init__()

        conv = default_conv
        self.n_feats = n_feats #64

        #TODO num_feat是通过重建模块的通道维
        self.out_dim = n_feats #64

        self.token_size = token_size
        self.n_fusionblocks = n_fusionblocks
        self.embedding_dim = embedding_dim = n_feats * (self.token_size**2) # 一个token对应的特征嵌入维度

        flatten_dim = embedding_dim
        hidden_dim = embedding_dim * expansion_ratio
        dim_head = embedding_dim // n_heads

        # 用作图像预处理
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # head module includes two residual blocks
        self.head = nn.Sequential(
            conv(n_colors, n_feats, 3),
            ResBlock(conv, n_feats, 5, act=nn.ReLU(True)),
            ResBlock(conv, n_feats, 5, act=nn.ReLU(True)),
        )

        # linear encoding after tokenization
        self.linear_encoding = nn.Linear(flatten_dim, embedding_dim)

        #TODO conventional self-attention block inside Transformer Block
        self.mhsa_block = nn.ModuleList([
            nn.ModuleList([
                PreNorm(
                    embedding_dim, 
                    SelfAttention(embedding_dim, n_heads, dim_head, dropout_rate)
                ),
                PreNorm(
                    embedding_dim, 
                    FeedForward(embedding_dim, hidden_dim, dropout_rate)
                ),
            ]) for _ in range(n_layers // 2)
        ])

        #TODO cross-scale token attention block inside Transformer Block
        self.csta_block = nn.ModuleList([
            nn.ModuleList([
                #TODO FFN for large tokens before the cross-attention
                nn.Sequential(
                    nn.LayerNorm(embedding_dim * 2),
                    nn.Linear(embedding_dim * 2, embedding_dim // 2),
                    nn.GELU(),
                    nn.Linear(embedding_dim // 2, embedding_dim // 2)
                ),
                # Two cross-attentions
                PreNorm2(
                    embedding_dim // 2,
                    CrossAttention(embedding_dim // 2, n_heads // 2, dim_head, dropout_rate)
                ),
                PreNorm2(
                    embedding_dim // 2,
                    CrossAttention(embedding_dim // 2, n_heads // 2, dim_head, dropout_rate)
                ),
                #TODO FFN for large tokens after the cross-attention
                nn.Sequential(
                    nn.LayerNorm(embedding_dim // 2),
                    nn.Linear(embedding_dim // 2, embedding_dim // 2),
                    nn.GELU(),
                    nn.Linear(embedding_dim // 2, embedding_dim * 2)
                ),
                # conventional FFN after the attention
                PreNorm(
                    embedding_dim,
                    FeedForward(embedding_dim, hidden_dim, dropout_rate)
                )
            ]) for _ in range(n_layers // 2)
        ])

        # CNN Branch borrowed from RCAN
        modules_body = [
            ResidualGroup(conv, n_feats, 3, reduction, n_resblocks=n_resblocks)
            for _ in range(n_resgroups) #4
        ]

        modules_body.append(conv(n_feats, n_feats, 3))
        self.cnn_branch = nn.Sequential(*modules_body)

        #TODO Fusion Blocks
        self.fusion_block = nn.ModuleList([
            nn.Sequential(
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True)),
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True)),
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True)),
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True)),
            ) for _ in range(n_fusionblocks) #4
        ])

        #TODO Fusion Block连接Transformer branch
        self.fusion_mlp = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embedding_dim),
            ) for _ in range(n_fusionblocks - 1) #3
        ])

        #TODO Fusion Block连接CNN branch
        self.fusion_cnn = nn.ModuleList([
            nn.Sequential(
                conv(n_feats, n_feats, 3), nn.ReLU(True), conv(n_feats, n_feats, 3)
            ) for _ in range(n_fusionblocks - 1) #3
        ])

        # single convolution to lessen dimension after body module
        self.conv_last = conv(n_feats * 2, n_feats, 3)

        ############################# tail module ##############################
        if task == 'none':
            self.tail = nn.Identity() # 恒等层
        elif task == 'sr':
            self.tail = nn.Sequential(
                Upsampler(conv, scale, n_feats, act=False), # 上采样
                conv(n_feats, n_colors, 3),
            )
        elif task == 'car':
            self.tail = conv(n_feats, n_colors, 3)

    def forward(self, x): # [bs, c, h, w]
        h, w = x.shape[-2:]

        #x = self.sub_mean(x)
        
        # 浅层特征提取
        x = self.head(x) 
        identity = x # 残差

        # 将特征图切成不重叠patch
        x_tkn = F.unfold(x, self.token_size, stride=self.token_size) # [bs, c*ts*ts, np] np=(h/ts)*(w/ts)
        x_tkn = rearrange(x_tkn, 'b d t -> b t d') # [bs, np, c*ts*ts]

        x_tkn = self.linear_encoding(x_tkn) + x_tkn # 线性编码 [bs, np, ed] ed=c*ts*ts

        #TODO 最外层模块的数量都由n_fusionblocks决定
        for i in range(self.n_fusionblocks): #4
            # MHSA
            x_tkn = self.mhsa_block[i][0](x_tkn) + x_tkn
            x_tkn = self.mhsa_block[i][1](x_tkn) + x_tkn # [bs, np, ed]

            #TODO CSTA跨尺度令牌注意力机制
            x_tkn_a, x_tkn_b = torch.split(x_tkn, self.embedding_dim // 2, -1) # [bs, np, ed/2]

            x_tkn_b = rearrange(x_tkn_b, 'b t d -> b d t') # # [bs, ed/2, np]
            x_tkn_b = F.fold(x_tkn_b, (h, w), self.token_size, stride=self.token_size) # [bs, c/2, h, w]

            x_tkn_b = F.unfold(x_tkn_b, self.token_size * 2, stride=self.token_size) #  [bs, ed/2*4, (h/ts/2)*(w/ts/2)]
            x_tkn_b = rearrange(x_tkn_b, 'b d t -> b t d')  # [bs, (h/ts/2)*(w/ts/2),  ed/2*4]

            x_tkn_b = self.csta_block[i][0](x_tkn_b) # [bs, (h/ts/2)*(w/ts/2),  ed/2]
            _x_tkn_a, _x_tkn_b = x_tkn_a, x_tkn_b # 残差
            
            #TODO CSTA核心操作
            x_tkn_a = self.csta_block[i][1](x_tkn_a, _x_tkn_b) + x_tkn_a # [bs, (h/ts)*(w/ts),  ed/2]
            x_tkn_b = self.csta_block[i][2](x_tkn_b, _x_tkn_a) + x_tkn_b # [bs, (h/ts/2)*(w/ts/2),  ed/2]

            x_tkn_b = self.csta_block[i][3](x_tkn_b) # [bs, (h/ts/2)*(w/ts/2),  ed*2]

            x_tkn_b = rearrange(x_tkn_b, 'b t d -> b d t') # [bs, ed*2, (h/ts/2)*(w/ts/2)]
            x_tkn_b = F.fold(x_tkn_b, (h, w), self.token_size * 2, stride=self.token_size) # [bs, c/2, h, w]

            x_tkn_b = F.unfold(x_tkn_b, self.token_size, stride=self.token_size) # [bs, ed/2, (h/ts)*(w/ts)]
            x_tkn_b = rearrange(x_tkn_b, 'b d t -> b t d') # [bs, (h/ts)*(w/ts), ed/2]

            x_tkn = torch.cat((x_tkn_a, x_tkn_b), -1) # [bs, (h/ts)*(w/ts), ed]
            x_tkn = self.csta_block[i][4](x_tkn) + x_tkn # [bs, (h/ts)*(w/ts), ed]

            #TODO Transformer与CNN的融合过程
            x = self.cnn_branch[i](x) # [bs, c, h, w]

            x_tkn_res, x_res = x_tkn, x # 残差

            x_tkn = rearrange(x_tkn, 'b t d -> b d t') # [bs, ed, (h/ts)*(w/ts)]
            x_tkn = F.fold(x_tkn, (h, w), self.token_size, stride=self.token_size) # [bs, c, h, w]

            f = torch.cat((x, x_tkn), 1) # [bs, 2c, h, w]
            f = f + self.fusion_block[i](f) # [bs, 2c, h, w]

            if i != (self.n_fusionblocks - 1):
                x_tkn, x = torch.split(f, self.n_feats, 1) # [bs, c, h, w]

                x_tkn = F.unfold(x_tkn, self.token_size, stride=self.token_size) # [bs, ed, np]
                x_tkn = rearrange(x_tkn, 'b d t -> b t d') # [bs, np, ed]
                # Transformer branch
                x_tkn = self.fusion_mlp[i](x_tkn)+ x_tkn_res
                # CNN branch
                x = self.fusion_cnn[i](x) + x_res

        # 最后一层conv梳理特征
        x = self.conv_last(f)

        # 残差连接
        x = x + identity 

        # 上采样层
        x = self.tail(x)
        #x = self.add_mean(x)

        return x

@register('act')
def make_act(no_upsampling=True):
    return ACT()