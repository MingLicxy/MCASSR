import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd 

try:
    import _gridencoder as _backend
except ImportError:
    from .backend import _backend

_gridtype_to_id = {
    'hash': 0,
    'tiled': 1,
}

class _grid_encode(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, gridtype=0, align_corners=False):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()

        B, D = inputs.shape # batch size, coord dim
        L = offsets.shape[0] - 1 # level
        C = embeddings.shape[1] # embedding dim for each level
        S = np.log2(per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = base_resolution # base resolution

        # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        # if C % 2 != 0, force float, since half for atomicAdd is very slow.
        if torch.is_autocast_enabled() and C % 2 == 0:
            embeddings = embeddings.to(torch.half)

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None

        _backend.grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, dy_dx, gridtype, align_corners)

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H, gridtype]
        ctx.align_corners = align_corners

        return outputs
    
    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, grad):

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype = ctx.dims
        align_corners = ctx.align_corners

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        _backend.grid_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners)

        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        return grad_inputs, grad_embeddings, None, None, None, None, None, None
        


grid_encode = _grid_encode.apply


#TODO GridEncoder是一个用于对输入坐标进行编码的网络层，通常用于3D坐标或2D坐标的嵌入
class GridEncoder(nn.Module):
    def __init__(self,
                 input_dim=3, 
                 num_levels=16, # 编码器的层数，每层具有不同的分辨率
                 level_dim=2, # 编码维度，决定每层生成多少特征
                 per_level_scale=2, # 分辨率缩放倍数,每一层的分辨率是上一层的2倍
                 base_resolution=16, # 基础层的分辨率，决定了最底层网格的大小
                 log2_hashmap_size=19, # 哈希表的大小的对数形式（即哈希表最大参数数量为 2^19）
                 desired_resolution=None, # 如果提供则使用该参数覆盖per_level_scale，用于设置最后一层的目标分辨率
                 gridtype='hash', # 网格类型，可以是哈希网格或平铺网格
                 align_corners=False, # 决定是否将坐标与网格角对齐
                 allocate_params=True, # 是否分配参数
                 levels_omit=[] # 允许忽略某些层
                 ):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        # 如果提供了desired_resolution，根据它重新计算每层的缩放比例，以确保最细的一层能达到期望的分辨率
        if desired_resolution is not None and num_levels > 1:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype] # "tiled" or "hash"
        self.align_corners = align_corners

        # 根据levels_omit确定哪些层级的通道要保留，并计算输出维度
        channels_keep = []
        for i in range(num_levels):
            if i not in levels_omit:
                channels_keep += [i*level_dim+x for x in range(level_dim)]
        channels_keep = torch.tensor(channels_keep, dtype=torch.long)
        self.register_buffer('channels_keep', channels_keep)
        self.output_dim_orig = num_levels * level_dim
        self.output_dim = channels_keep.shape[0]

        # allocate parameters
        # 计算每个层级的分辨率和参数数量，确保总参数数目不超过2^log2_hashmap_size
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            params_in_level = min(self.max_params, (resolution if align_corners else resolution + 1) ** input_dim) # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8) # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)

        # 总参数数目 
        self.n_params = offsets[-1] * level_dim

        # parameters
        if allocate_params:
            self.embeddings = nn.Parameter(torch.empty(offset, level_dim))
            self.reset_parameters()
    
    # 参数初始化
    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} gridtype={self.gridtype} align_corners={self.align_corners}"
    
    def forward(self, inputs, bound=1):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # return: [..., num_levels * level_dim]

        inputs = (inputs + bound) / (2 * bound) # map to [0, 1]

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        #TODO 调用低级的编码函数 grid_encode，将坐标编码为高维特征
        outputs = grid_encode(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad, self.gridtype_id, self.align_corners)
        outputs = outputs.view(prefix_shape + [self.output_dim_orig])

        return outputs[..., self.channels_keep]
