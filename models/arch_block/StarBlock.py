import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath

"""
The code comes from https://github.com/ma-xu/Rewrite-the-Stars/blob/main/imagenet/demonet.py
一种结合了"星星"(乘法)操作的模块,用于高效紧凑网络设计(DemoNet)
"""
class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, mode="sum"):
        super().__init__()
        self.mode = mode # 决定是乘法还是加法
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.f = nn.Linear(dim, 6 * dim)
        self.act = nn.GELU()
        self.g = nn.Linear(3 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else 1.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.norm(x)
        x = self.dwconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.f(x)
        B, H, W, C = x.size()
        x1, x2 = x.reshape(B, H, W, 2, int(C // 2)).unbind(3)
        x = self.act(x1) + x2 if self.mode == "sum" else self.act(x1) * x2
        x = self.g(x)
        x = input + self.drop_path(self.gamma * x)
        return x


class DemoNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depth=12, dim=384, drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 mode="mul", **kwargs):
        super().__init__()
        assert mode in ["sum", "mul"]
        self.num_classes = num_classes
        self.stem = nn.Conv2d(in_chans, dim, kernel_size=16, stride=16)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[Block(dim=dim, drop_path=dp_rates[i],
                                            layer_scale_init_value=layer_scale_init_value, mode=mode)
                                      for i in range(depth)])

        self.norm = nn.LayerNorm(dim)  # final norm layer
        self.head = nn.Linear(dim, self.num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x).permute(0, 2, 3, 1)
        x = self.blocks(x)
        x = self.norm(x.mean([1, 2]))
        x = self.head(x)
        return x




"""
The code comes from https://github.com/ma-xu/Rewrite-the-Stars/blob/main/imagenet/starnet.py
一种结合了"星星"(乘法)操作的模块,用于高效紧凑网络设计(StarNet)
"""

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)


