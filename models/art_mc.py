# Attention Retractable Transformer (ART) model
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

from timm.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from models import register

NEG_INF = -1000000

################################ ART(稀疏窗口注意力，密集窗口与稀疏窗口交替使用) ##################################
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

    # def flops(self, N):
    #     flops = N * 2 * self.pos_dim
    #     flops += N * self.pos_dim * self.pos_dim
    #     flops += N * self.pos_dim * self.pos_dim
    #     flops += N * self.pos_dim * self.num_heads
    #     return flops

#TODO 重写注意力逻辑，使其输入为qkv数组（由x得到qkv的逻辑写在下一级模块）
class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    #TODO 改为直接输入qkv数组，由x投影得到qkv数组的逻辑写在下一级模块
    def forward(self, qkv, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (H, W)  # Gh, Gw

        #qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        #q, k, v = qkv[0], qkv[1], qkv[2]
        # qkv[3, B_, hesds, N, C//heads]
        q, k, v = qkv[0], qkv[1], qkv[2]
        #print("##########################################", q.shape) # [1024, 4, 64, 16]
        #BUG
        B_, heads, N, head_C = q.shape
        C = head_C * heads
        assert N == H * W, "flatten img_tokens has wrong size"

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing='ij'))  # 2, 2Gh-1, 2W2-1
            #biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Gh, Gw
            #coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


    # def forward(self, x, H, W, mask=None):
    #     """
    #     Args:
    #         x: input features with shape of (num_groups*B, N, C)
    #         mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
    #         H: height of each group
    #         W: width of each group
    #     """
    #     group_size = (H, W)
    #     B_, N, C = x.shape
    #     assert H * W == N
    #     qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
    #     q, k, v = qkv[0], qkv[1], qkv[2]

    #     q = q * self.scale
    #     attn = (q @ k.transpose(-2, -1))  # (B_, self.num_heads, N, N), N = H*W

    #     if self.position_bias:
    #         # generate mother-set
    #         position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
    #         position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
    #         biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing='ij'))  # 2, 2Gh-1, 2W2-1
    #         biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

    #         # get pair-wise relative position index for each token inside the window
    #         coords_h = torch.arange(group_size[0], device=attn.device)
    #         coords_w = torch.arange(group_size[1], device=attn.device)
    #         coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Gh, Gw
    #         coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
    #         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
    #         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
    #         relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
    #         relative_coords[:, :, 1] += group_size[1] - 1
    #         relative_coords[:, :, 0] *= 2 * group_size[1] - 1
    #         relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

    #         pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
    #         # select position bias
    #         relative_position_bias = pos[relative_position_index.view(-1)].view(
    #             group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
    #         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
    #         attn = attn + relative_position_bias.unsqueeze(0)

    #     if mask is not None:
    #         nP = mask.shape[0]
    #         attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
    #             0)  # (B, nP, nHead, N, N)
    #         attn = attn.view(-1, self.num_heads, N, N)
    #         attn = self.softmax(attn)
    #     else:
    #         attn = self.softmax(attn)

    #     attn = self.attn_drop(attn)

    #     x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x

#TODO 融合了ART重要逻辑的基础模块
class ARTTransformerBlock(nn.Module):
    r""" ART Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size: window size of dense attention
        interval: interval size of sparse attention
        ds_flag (int): use Dense Attention or Sparse Attention, 0 for DAB and 1 for SAB.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7, # 定义密集窗口尺寸
                 interval=8, # 定义窗口的稀疏程度
                 ds_flag=0, # 决定是否采用稀疏窗口
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.interval = interval
        self.ds_flag = ds_flag
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        #TODO qkv线性投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 

        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    #TODO 特征提取主干中的每一个基础快都是两个输入，两个并行的DAB/SAB块内做互换的交叉注意力
    def forward(self, x, ref, x_size):
        H, W = x_size
        B, L, C = x.shape #[1, 65536, 64]
        assert L == H * W, "input feature has wrong size %d, %d, %d" % (L, H, W)
        
        #TODO 分窗（与SwinIR逻辑不同）
        if min(H, W) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.ds_flag = 0
            self.window_size = min(H, W)

        shortcut = x   # x作为主体留下残差
        
        #TODO x与x_ref的尺寸完全相同，应该对他们进行完全一致的处理
        x = self.norm1(x)
        x_ref = self.norm1(ref)

        x = x.view(B, H, W, C)
        x_ref = x_ref.view(B, H, W, C)

        # padding
        size_par = self.interval if self.ds_flag == 1 else self.window_size
        pad_l = pad_t = 0
        pad_r = (size_par - W % size_par) % size_par
        pad_b = (size_par - H % size_par) % size_par

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x_ref = F.pad(x_ref, (0, 0, pad_l, pad_r, pad_t, pad_b))

        _, Hd, Wd, _ = x.shape  # x.shape == x_ref.shape

        mask = torch.zeros((1, Hd, Wd, 1), device=x.device) # x与x_ref采用的mask也相同
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1
        
        ############################ TODO 决定DAB/SAB的关键代码（利用mask逻辑完成） ############################
        # partition the whole feature map into several groups
        if self.ds_flag == 0:  # Dense Attention
            G = Gh = Gw = self.window_size

            x = x.reshape(B, Hd // G, G, Wd // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            x_ref = x_ref.reshape(B, Hd // G, G, Wd // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.reshape(B * Hd * Wd // G ** 2, G ** 2, C)
            x_ref = x_ref.reshape(B * Hd * Wd // G ** 2, G ** 2, C)

            nP = Hd * Wd // G ** 2 # number of partitioning groups
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Hd // G, G, Wd // G, G, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
                mask = mask.reshape(nP, 1, G * G)
                attn_mask = torch.zeros((nP, G * G, G * G), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None
        if self.ds_flag == 1: # Sparse Attention
            I, Gh, Gw = self.interval, Hd // self.interval, Wd // self.interval

            x = x.reshape(B, Gh, I, Gw, I, C).permute(0, 2, 4, 1, 3, 5).contiguous()
            x_ref = x_ref.reshape(B, Gh, I, Gw, I, C).permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.reshape(B * I * I, Gh * Gw, C)
            x_ref = x_ref.reshape(B * I * I, Gh * Gw, C)

            nP = I ** 2  # number of partitioning groups
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Gh, I, Gw, I, 1).permute(0, 2, 4, 1, 3, 5).contiguous()
                mask = mask.reshape(nP, 1, Gh * Gw)
                attn_mask = torch.zeros((nP, Gh * Gw, Gh * Gw), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None
        ###################################################################################################
        #TODO 计算注意力之前先投影获取qkv，再交换x与x_ref的qkv [B, L=H*W, 3, heads, C//heads] -> [3, B, hesds, L=H*W, C//heads]
        #print("########################################", x.shape)  #[1024, 64, 64]
        B_, N, C = x.shape  # x与x_ref前面被处理过
        assert Gh * Gw == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", qkv.shape) # [3, 1, 4, 65536, 16]
        qkv_ref = self.qkv(x_ref).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        qkv[1, :, :, :, :] = qkv_ref[1]  # ref_k
        qkv[2, :, :, :, :] = qkv_ref[2]  # ref_v

        #TODO 计算注意力图MSA
        #x = self.attn(x, Gh, Gw, mask=attn_mask)  # nP*B, Gh*Gw, C
        #print("######################################", Gh, Gw)  # 8, 8
        x = self.attn(qkv, Gh, Gw, mask=attn_mask) 

        # merge embeddings
        if self.ds_flag == 0:
            x = x.reshape(B, Hd // G, Wd // G, G, G, C).permute(0, 1, 3, 2, 4,
                                                                5).contiguous()  # B, Hd//G, G, Wd//G, G, C
        else:
            x = x.reshape(B, I, I, Gh, Gw, C).permute(0, 3, 1, 4, 2, 5).contiguous()  # B, Gh, I, Gw, I, C
        x = x.reshape(B, Hd, Wd, C)

        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, ds_flag={self.ds_flag}, mlp_ratio={self.mlp_ratio}"

    # def flops(self):
    #     flops = 0
    #     H, W = self.input_resolution
    #     # norm1
    #     flops += self.dim * H * W
    #     # Dense and Sparse Attention
    #     nW = H * W / self.window_size / self.window_size
    #     flops += nW * self.attn.flops(self.window_size * self.window_size)
    #     # mlp
    #     flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
    #     # norm2
    #     flops += self.dim * H * W
    #     return flops

# DAB+SAB
class BasicLayer(nn.Module):
    """ A basic ART Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): dense window size.
        interval: sparse interval size
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 interval,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
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
        for i in range(depth): # depth表示DAB与SAB的数量之和，且是偶数
            ds_flag = 0 if (i % 2 == 0) else 1  # 交替采用DAB与SAB
            self.blocks.append(ARTTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                interval=interval,
                ds_flag=ds_flag,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    
    #TODO 这个x_size对应x的[H,W]
    def forward(self, x, ref, x_size):
        # blk接受两个输入对应一个输出
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, ref, x_size)
            else:
                x = blk(x, ref, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    # def flops(self):
    #     flops = 0
    #     for blk in self.blocks:
    #         flops += blk.flops()
    #     if self.downsample is not None:
    #         flops += self.downsample.flops()
    #     return flops


class ResidualGroup(nn.Module):
    """Residual group including some ART Transformer Blocks (ResidualGroup).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): dense window size.
        interval: sparse interval size
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 interval,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv'):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            interval=interval,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        # build the last conv layer in each residual group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))
            
        # 图像分块时用到img_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, ref, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, ref, x_size), x_size))) + x

    # def flops(self):
    #     flops = 0
    #     flops += self.residual_group.flops()
    #     h, w = self.input_resolution
    #     flops += h * w * self.dim * self.dim * 9
    #     flops += self.patch_embed.flops()
    #     flops += self.patch_unembed.flops()

    #     return flops


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

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
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    # def flops(self):
    #     flops = 0
    #     h, w = self.img_size
    #     if self.norm is not None:
    #         flops += h * w * self.embed_dim
    #     return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

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
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    # def flops(self):
    #     flops = 0
    #     return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


######################################################## ART主体 ####################################################
class ART(nn.Module):
    r""" ART
        A PyTorch impl of : `Accurate Image Restoration with Attention Retractable Transformer`.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each ART residual group.
        num_heads (tuple(int)): Number of attention heads in different layers.
        interval(tuple(int)): Interval size of sparse attention in different residual groups
        window_size (int): Window size of dense attention. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 img_size=48,   # 4
                 patch_size=1,   # 像素级分块
                 in_chans=2,
                 embed_dim=64, #96 180
                 depths=(4, 4, 4, 4),   # (6, 6, 6, 6)
                 num_heads=(4, 4, 4, 4), # (6, 6, 6, 6)
                 interval=(8, 8, 8, 8),  #TODO 重要参数
                 window_size=8,  #7
                 mlp_ratio=2.,  #4.
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='none', # 无上采样模块
                 resi_connection='1conv',
                 **kwargs):
        super(ART, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        #TODO 最后的融合步骤沿通道cat 128->64
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        #TODO num_feat是通过重建模块的通道维
        self.out_dim = num_feat

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Group including ART Transformer blocks (ResidualGroup)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                interval=interval[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- restoration module ------------------------- #
        if self.upsampler == 'none': #TODO 
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim*2, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))  
        elif self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim*2, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        else:
            # for image denoising
            self.conv_last = nn.Conv2d(embed_dim*2, num_out_ch, 3, 1, 1)

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

    def forward_features(self, x, ref):
        # x_size不需要直接在最外层输入
        x_size = (x.shape[2], x.shape[3])

        x = self.patch_embed(x)
        ref = self.patch_embed(ref)

        if self.ape:
            x = x + self.absolute_pos_embed
            ref = ref + self.absolute_pos_embed
        x = self.pos_drop(x)
        ref = self.pos_drop(ref)

        for layer in self.layers:
            x = layer(x, ref, x_size)
        
        # 两个输入对应一个输出
        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x, ref):
        #self.mean = self.mean.type_as(x)
        #x = (x - self.mean) * self.img_range

        #TODO 不采用上采样器
        if self.upsampler == 'none':
            x = self.conv_first(x)
            ref = self.conv_first(ref)

            ################################ TODO 两次交替输入后cat ################################
            out_x = self.conv_after_body(self.forward_features(x, ref)) + x
            out_ref = self.conv_after_body(self.forward_features(ref, x)) + ref
            out_fused = torch.cat([out_x, out_ref], dim=1) # dim: 64->128
            x = self.conv_before_upsample(out_fused)
            #######################################################################################

        elif self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        else:
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        #x = x / self.img_range + self.mean

        return x

    # def flops(self):
    #     flops = 0
    #     h, w = self.patches_resolution
    #     flops += h * w * 3 * self.embed_dim * 9
    #     flops += self.patch_embed.flops()
    #     for layer in self.layers:
    #         flops += layer.flops()
    #     flops += h * w * 3 * self.embed_dim * self.embed_dim
    #     flops += self.upsample.flops()
    #     return flops

@register('art_mc')
def make_art(no_upsampling=True):
    return ART()


# 模型测试
if __name__== '__main__':
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    
    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    x = torch.randn(1, 3, 48, 48).cuda()
    ref = torch.randn(1, 3, 48, 48).cuda()
    # x = torch.randn(1, 3, 256, 256)

    model = ART().cuda()
    # model = SAFMN(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    #print(flop_count_table(FlopCountAnalysis(model, x, ref), activations=ActivationCountAnalysis(model, x, ref)))
    output = model(x, ref)
    print(output[0].shape)