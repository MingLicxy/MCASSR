# Enhanced Deep Residual Networks for Single Image Super-Resolution
# https://arxiv.org/abs/1707.02921
# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register

################################ EDSR编码器 ##################################
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

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

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
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


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        #x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

# if __name__== '__main__':
#     #############Test Model Complexity #############
#     from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    
#     # x = torch.randn(1, 3, 640, 360)
#     # x = torch.randn(1, 3, 427, 240)
#     x = torch.randn(1, 3, 48, 48)
#     # x = torch.randn(1, 3, 256, 256)
#     args = Namespace()
#     args.n_resblocks = 16
#     args.n_feats = 64
#     args.res_scale = 1
#     args.scale = [2]
#     args.no_upsampling = True
#     args.rgb_range = 1
#     args.n_colors = 3

#     model = EDSR(args)
#     # model = SAFMN(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
#     print(model)
#     print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
#     print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
#     output = model(x)
#     print(output.shape)


if __name__== '__main__':
    #############Test Model Complexity #############
    x = torch.randn(1, 3, 128, 128)

    args = Namespace()
    args.n_resblocks = 16
    args.n_feats = 64
    args.res_scale = 1
    args.scale = [2]
    args.no_upsampling = True
    args.rgb_range = 1
    args.n_colors = 3

    
    # #############Test Model Complexity #############
    # from thop import profile
    # model = EDSR(args).cuda()
    # inputs = torch.randn(1, 3, 128, 128).cuda()  # 适配你的输入尺寸
    # flops, params = profile(model, inputs=(inputs,))
    # # 转换单位
    # flops_g = flops / 1e9  # GFLOPs
    # params_m = params / 1e6  # MParams

    # print(f"FLOPs: {flops_g:.2f}G, Params: {params_m:.2f}M")

    # #############Test Model Running time #############
    # model = EDSR(args).eval().cuda()
    # inputs = torch.randn(1, 3, 128, 128).cuda()

    # # 预热
    # for _ in range(10):
    #     _ = model(inputs)

    # # 创建 CUDA 事件
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)

    # # 计时
    # start_event.record()
    # with torch.no_grad():
    #     for _ in range(100):
    #         _ = model(inputs)
    # end_event.record()

    # # 等待计算完成
    # torch.cuda.synchronize()

    # # 计算时间（毫秒）
    # avg_time = start_event.elapsed_time(end_event) / 100
    # print(f"Avg inference time: {avg_time:.2f} ms") 
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    import time

    # -------------------------------
    # 1. 初始化模型与输入
    # -------------------------------
    model = EDSR(args).eval().cuda()
    inputs = torch.randn(1, 3, 128, 128).cuda()  # 输入尺寸可按需求修改

    # -------------------------------
    # 2. 参数量统计
    # -------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {total_params/1e6:.2f} M")
    print(f"Trainable Params: {trainable_params/1e6:.2f} M")

    # -------------------------------
    # 3. FLOPs 统计（仅卷积/线性层）
    # -------------------------------
    flops = FlopCountAnalysis(model, inputs)
    print(f"FLOPs: {flops.total()/1e9:.2f} G")  # GFLOPs
    print(parameter_count_table(model))  # 可选，输出更详细表格

    # -------------------------------
    # 4. 推理时间测试
    # -------------------------------
    # 预热
    for _ in range(10):
        _ = model(inputs)

    # 创建 CUDA 事件
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 计时
    start_event.record()
    with torch.no_grad():
        for _ in range(100):
            _ = model(inputs)
    end_event.record()
    torch.cuda.synchronize()

    avg_time = start_event.elapsed_time(end_event) / 100  # 毫秒
    print(f"Avg inference time: {avg_time:.2f} ms")













@register('edsr-baseline')
def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1,
                       scale=1, no_upsampling=True, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3 # 适应.mat 4
    return EDSR(args)


@register('edsr')
def make_edsr(n_resblocks=32, n_feats=256, res_scale=0.1,
              scale=1, no_upsampling=True, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSR(args)
