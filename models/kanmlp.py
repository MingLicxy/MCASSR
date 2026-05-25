import os
import sys
import torch
import math
from torch import nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import register
# from models.arch_kan.efficient_kan import KANLinear
# from models.arch_kan.fast_kan import FastKANLayer
# from models.arch_kan.faster_kan import FasterKANLayer
# from models.arch_kan.wave_kan import WaveKANLinear
from kat_rational import KAT_Group


################################ KANMLP解码器 ##################################
@register('kanmlp')
class KANMLP(nn.Module):
    def __init__(self, in_dim, hidden_list=None, out_dim=None, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_list = hidden_list or in_dim
        self.dim = in_dim 
        
        # KAN超参
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        # 网络层数组
        self.layers = torch.nn.ModuleList()
        lastv = in_dim # 580
        for hidden in hidden_list:
            self.layers.append(
                KANLinear(
                        lastv, 
                        hidden, # 256
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
                )
            lastv = hidden
        # 输出层
        self.layers.append(
            KANLinear(
                        lastv,
                        out_dim, # 3
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
        )


        #self.drop = nn.Dropout(drop) 
        #self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, update_grid=False):
        shape = x.shape[:-1] # x最后一维
        x = x.view(-1, x.shape[-1])
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x.view(*shape, -1)


################################ FastKANMLP解码器 ##################################
@register('fastkanmlp')
class FastKANMLP(nn.Module):
    def __init__(self, in_dim, hidden_list=None, out_dim=None, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_list = hidden_list or in_dim
        self.dim = in_dim 
        
        # FastKAN超参
        grid_min: float = -2.
        grid_max: float = 2.
        num_grids: int = 8
        use_base_update: bool = True
        base_activation = F.silu
        spline_weight_init_scale: float = 0.1

        # 网络层数组
        self.layers = torch.nn.ModuleList()
        lastv = in_dim # 580
        for hidden in hidden_list:
            self.layers.append(
                FastKANLayer(
                    lastv, 
                    hidden, # 256
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                )
            )
            lastv = hidden
        # 输出层
        self.layers.append(
            FastKANLayer(
                lastv, 
                out_dim, # 3
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            )
        )


        #self.drop = nn.Dropout(drop) 
        #self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, update_grid=False):
        shape = x.shape[:-1] # x最后一维
        x = x.view(-1, x.shape[-1])
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x.view(*shape, -1)


################################ ResFastKANMLP解码器 ##################################
@register('resfastkanmlp')
class ResFastKANMLP(nn.Module):
    def __init__(self, in_dim, hidden_list=None, out_dim=None, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_list = hidden_list or in_dim
        self.dim = in_dim 
        
        # FastKAN超参
        grid_min: float = -2.
        grid_max: float = 2.
        num_grids: int = 8
        use_base_update: bool = True
        base_activation = F.silu
        spline_weight_init_scale: float = 0.1

        # 网络层数组
        self.layers = torch.nn.ModuleList()
        lastv = in_dim # 580
        
        #TODO 残差块FastKANResBlock的应用没有排除输入层
        #TODO 可以单独创建输入层，使得残差块仅用于隐藏层
        #TODO hidden_list=[256, 256]对应两个FastKANResBlock，第一个包含输入层
        for hidden in hidden_list:
            self.layers.append(
                FastKANResBlock(lastv, hidden)
            )
            lastv = hidden
        # 输出层
        self.layers.append(
            FastKANLayer(
                lastv, 
                out_dim, # 3
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            )
        )


        #self.drop = nn.Dropout(drop) 
        #self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, update_grid=False):
        shape = x.shape[:-1] # x最后一维
        x = x.view(-1, x.shape[-1])
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x.view(*shape, -1)
    
# 残差块仅仅包含隐藏层
@register('resfastkanmlp_1')
class ResFastKANMLP_1(nn.Module):
    def __init__(self, in_dim, hidden_list=None, out_dim=None, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_list = hidden_list or in_dim
        self.dim = in_dim 
        
        # FastKAN超参
        grid_min: float = -2.
        grid_max: float = 2.
        num_grids: int = 8
        use_base_update: bool = True
        base_activation = F.silu
        spline_weight_init_scale: float = 0.1

        # 网络层数组
        self.layers = torch.nn.ModuleList()
        lastv = hidden_list[0] 

        # 输入层
        self.layers.append(
            FastKANLayer(
                in_dim, 
                lastv, 
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            )
        )

        #TODO 隐藏层：[256,256]包含一个FastKANResBlock
        for hidden in hidden_list[1:]:
            self.layers.append(
                FastKANResBlock(lastv, hidden)
            )
            lastv = hidden

        # 输出层
        self.layers.append(
            FastKANLayer(
                lastv, 
                out_dim, # 3
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            )
        )


        #self.drop = nn.Dropout(drop) 
        #self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, update_grid=False):
        shape = x.shape[:-1] # x最后一维
        x = x.view(-1, x.shape[-1])
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x.view(*shape, -1)

class FastKANResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        # FastKAN超参
        grid_min: float = -2.
        grid_max: float = 2.
        num_grids: int = 8
        use_base_update: bool = True
        base_activation = F.silu
        spline_weight_init_scale: float = 0.1

        self.fasterkanlayer1 = FastKANLayer(
                              in_dim, 
                              out_dim, # 3
                              grid_min=grid_min,
                              grid_max=grid_max,
                              num_grids=num_grids,
                              use_base_update=use_base_update,
                              base_activation=base_activation,
                              spline_weight_init_scale=spline_weight_init_scale,
                            )
        self.fasterkanlayer2 = FastKANLayer(
                              out_dim, 
                              out_dim, # 3
                              grid_min=grid_min,
                              grid_max=grid_max,
                              num_grids=num_grids,
                              use_base_update=use_base_update,
                              base_activation=base_activation,
                              spline_weight_init_scale=spline_weight_init_scale,
                            )
        #TODO 应对残差连接维度不同（注意KAN代替Linear+act，单独的Linear不用KAN代替）
        if in_dim != out_dim:
            self.residual_layer = nn.Linear(in_dim, out_dim)
            # self.residual_layer = FastKANLayer(
            #                   in_dim, 
            #                   out_dim, # 3
            #                   grid_min=grid_min,
            #                   grid_max=grid_max,
            #                   num_grids=num_grids,
            #                   use_base_update=use_base_update,
            #                   base_activation=base_activation,
            #                   spline_weight_init_scale=spline_weight_init_scale,
            #                 )
        else:
            self.residual_layer = None

        #TODO 残差块只包含隐藏层时in_dim=out_dim
        #assert in_dim == out_dim
        
    def forward(self, x):

        if self.residual_layer:
            residual = self.residual_layer(x)
        else:
            residual = x
        out = self.fasterkanlayer1(x)
        out = self.fasterkanlayer2(out)
        out += residual
        return out




################################ FasterKANMLP解码器 ##################################
@register('fasterkanmlp')
class FasterKANMLP(nn.Module):
    def __init__(self, in_dim, hidden_list=None, out_dim=None, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_list = hidden_list or in_dim
        self.dim = in_dim 
        
        # FasterKAN超参
        grid_min: float = -1.2
        grid_max: float = 0.2
        num_grids: int = 8
        exponent: int = 2
        inv_denominator: float = 0.5
        train_grid: bool = False     
        train_inv_denominator: bool = False
        #use_base_update: bool = True,
        base_activation = None
        spline_weight_init_scale: float = 1.0

        # 网络层数组
        self.layers = torch.nn.ModuleList()
        lastv = in_dim # 580

        for hidden in hidden_list:
            self.layers.append(
                FasterKANLayer(
                    lastv, 
                    hidden, # 256
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    exponent = exponent,
                    inv_denominator = inv_denominator,
                    train_grid = train_grid ,
                    train_inv_denominator = train_inv_denominator,
                    #use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                )
            )
            lastv = hidden
        # 输出层
        self.layers.append(
            FasterKANLayer(
                lastv, 
                out_dim, # 3
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                exponent = exponent,
                inv_denominator = inv_denominator,
                train_grid = train_grid ,
                train_inv_denominator = train_inv_denominator,
                #use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            )
        )


        #self.drop = nn.Dropout(drop) 
        #self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, update_grid=False):
        shape = x.shape[:-1] # x最后一维
        x = x.view(-1, x.shape[-1])
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x.view(*shape, -1)


################################ WaveKANMLP解码器 ##################################
@register('wavekanmlp')
class WaveKANMLP(nn.Module):
    def __init__(self, in_dim, hidden_list=None, out_dim=None, wavelet_type='mexican_hat', drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_list = hidden_list or in_dim
        self.dim = in_dim 
        

        # 网络层数组
        self.layers = torch.nn.ModuleList()
        lastv = in_dim # 580
        for hidden in hidden_list:
            self.layers.append(
                WaveKANLinear(lastv, hidden, wavelet_type)
                )
            lastv = hidden
        # 输出层
        self.layers.append(
            WaveKANLinear(lastv, out_dim, wavelet_type)
            )


        #self.drop = nn.Dropout(drop) 
        #self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, update_grid=False):
        shape = x.shape[:-1] # x最后一维
        x = x.view(-1, x.shape[-1])
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x.view(*shape, -1)
    

################################ GroupKANMLP解码器 ##################################
@register('groupkanmlp')
class GroupKANMLP(nn.Module):
    def __init__(
            self,
            in_dim,
            out_act, #输出层对应激活 "gelu"
            out_dim,
            act_init_list, # ["identity", "gelu", "gelu"] 输入层与隐藏层激活
            hidden_list, # [256, 256, 256] 一共四层
            bias=True, # 默认为true
            drop=0., # dropout
            **kwargs):
        super().__init__()

        layers = []
        lastv = in_dim
        #TODO 同时遍历输入层与隐藏层的维度与KAT激活
        for act_init, hidden in zip(act_init_list, hidden_list):
            layers.append(KAT_Group(mode = act_init))
            layers.append(nn.Linear(lastv, hidden, bias=bias))

            lastv = hidden
        # 输出层
        layers.append(KAT_Group(mode = out_act))
        layers.append(nn.Linear(lastv, out_dim, bias=bias))
        self.layers = nn.Sequential(*layers)

    
    #TODO 每一层都有自定义KAT激活，且激活在前
    def forward(self, x):
        #BUG KAT只接受(B,L,C)形式的Tensor作为输入
        # 在LIIF/LTE中倾向于将[B,L,C]转换为[B*L,C]输入MLP解码后恢复[B,L,C]
        # 如果采用GroupKANMLP作为解码器需要改进对应LIIF/LTE中的代码

        # shape = x.shape[:-1]
        # x = self.layers(x.view(-1, x.shape[-1]))
        # return x.view(*shape, -1)

        return self.layers(x)








if __name__ == '__main__':
    model = GroupKANMLP(
        in_dim=3,
        out_act="gelu",
        out_dim=3,
        act_init_list= ["identity", "gelu", "gelu"],
        hidden_list=[256, 256, 256],
    ).cuda().eval()
    

    x = torch.randn((16, 2304, 3)).cuda()
    x = model(x)
    print(x.shape)
    print(model)

# ResFastKANMLP(
#   (layers): ModuleList(
#     (0): FastKANResBlock(
#       (fasterkanlayer1): FastKANLayer(
#         (layernorm): LayerNorm((3,), eps=1e-05, elementwise_affine=True)
#         (rbf): RadialBasisFunction()
#         (spline_linear): SplineLinear(in_features=24, out_features=256, bias=False)
#         (base_linear): Linear(in_features=3, out_features=256, bias=True)
#       )
#       (fasterkanlayer2): FastKANLayer(
#         (layernorm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
#         (rbf): RadialBasisFunction()
#         (spline_linear): SplineLinear(in_features=2048, out_features=256, bias=False)
#         (base_linear): Linear(in_features=256, out_features=256, bias=True)
#       )
#       (residual_layer): FastKANLayer(
#         (layernorm): LayerNorm((3,), eps=1e-05, elementwise_affine=True)
#         (rbf): RadialBasisFunction()
#         (spline_linear): SplineLinear(in_features=24, out_features=256, bias=False)
#         (base_linear): Linear(in_features=3, out_features=256, bias=True)
#       )
#     )
#     (1): FastKANResBlock(
#       (fasterkanlayer1): FastKANLayer(
#         (layernorm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
#         (rbf): RadialBasisFunction()
#         (spline_linear): SplineLinear(in_features=2048, out_features=256, bias=False)
#         (base_linear): Linear(in_features=256, out_features=256, bias=True)
#       )
#       (fasterkanlayer2): FastKANLayer(
#         (layernorm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
#         (rbf): RadialBasisFunction()
#         (spline_linear): SplineLinear(in_features=2048, out_features=256, bias=False)
#         (base_linear): Linear(in_features=256, out_features=256, bias=True)
#       )
#     )
#     (2): FastKANLayer(
#       (layernorm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
#       (rbf): RadialBasisFunction()
#       (spline_linear): SplineLinear(in_features=2048, out_features=3, bias=False)
#       (base_linear): Linear(in_features=256, out_features=3, bias=True)
#     )
#   )
# )