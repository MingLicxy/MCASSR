# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

from argparse import Namespace

import torch
import torch.nn as nn

from models import register

#BUG RDN之所以显存占用很高是因为，密集残差连接（Cat）导致通道数大幅增加
################################ RDN编码器 ##################################
class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate # 每次残差连接增加的通道数
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        self.args = args
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        #TODO 定义模型超参数
        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            # Up-sampling net
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            elif r == 4:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        if self.args.no_upsampling:
            return x
        else:
            return self.UPNet(x)
        

# if __name__== '__main__':
#     #############Test Model Complexity #############
#     from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    
#     # x = torch.randn(1, 3, 640, 360)
#     # x = torch.randn(1, 3, 427, 240)
#     x = torch.randn(1, 3, 48, 48)
#     # x = torch.randn(1, 3, 256, 256)

#     args = Namespace()
#     args.G0 = 64
#     args.RDNkSize = 3
#     args.RDNconfig = 'B'
#     args.scale = [2]
#     args.no_upsampling = True
#     args.n_colors = 3

#     model = RDN(args)
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
    args.G0 = 64
    args.RDNkSize = 3
    args.RDNconfig = 'B'
    args.scale = [2]
    args.no_upsampling = True
    args.n_colors = 3

    
    

    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    import time

    # -------------------------------
    # 1. 初始化模型与输入
    # -------------------------------
    model = RDN(args).eval().cuda()
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
    



@register('rdn')
def make_rdn(G0=64, RDNkSize=3, RDNconfig='B',
             scale=1, no_upsampling=True):
    args = Namespace()
    args.G0 = G0
    args.RDNkSize = RDNkSize
    args.RDNconfig = RDNconfig

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.n_colors = 3
    return RDN(args)
