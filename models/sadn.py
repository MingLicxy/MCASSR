import torch
import os
import time
from os.path import exists
import torch.nn as nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from torch.nn.utils import weight_norm

import models
from models import register


#from litsr.data import *



################################ SADN编码器（SADN） ##################################


#########################################-----常用子模块(common)------#########################################
class WeightNormedConv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        act=nn.ReLU(True),
    ):
        conv = weight_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                bias=bias,
            )
        )
        m = [conv]
        if act:
            m.append(act)
        super().__init__(*m)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        if len(rgb_std) != len(rgb_mean):
            assert len(rgb_std) == 1
            rgb_std = rgb_std * len(rgb_mean)
        channel = len(rgb_mean)
        super(MeanShift, self).__init__(channel, channel, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(channel).view(channel, channel, 1, 1)
        self.weight.data.div_(std.view(channel, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False,
        bn=True,
        act=nn.ReLU(True),
    ):

        m = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=(kernel_size // 2),
                stride=stride,
                bias=bias,
            )
        ]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):

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
#########################################-----常用子模块(common)------#########################################





#########################################-----定义尺度敏感动态卷积(SAD-Conv)------#########################################
class ScaleAwareAttention2d(nn.Module):
    def __init__(self, in_channels, ratios, K, temperature, init_weight=True):
        super().__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_channels != 3:
            hidden_channels = int(in_channels * ratios) + 1
        else:
            hidden_channels = K
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_channels)
        self.fc2 = nn.Conv2d(hidden_channels + 2, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            # print('Change temperature to:', str(self.temperature))

    def forward(self, x, scale):
        if not self.training:
            temperature = 1
        else:
            temperature = self.temperature

        batch_size = x.shape[0]
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = torch.cat(
            [x, torch.ones([batch_size, 2, 1, 1], device=x.device) * scale], dim=1
        )
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / temperature, 1)


class ScaleAwareDynamicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio=0.25,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        K=4,
        temperature=34,
        init_weight=True,
    ):
        super().__init__()
        assert in_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = ScaleAwareAttention2d(in_channels, ratio, K, temperature)

        self.weight = nn.Parameter(
            torch.randn(
                K, out_channels, in_channels // groups, kernel_size, kernel_size
            ),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_channels))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x, scale):
        softmax_attention = self.attention(x, scale)
        batch_size, _, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(
            -1, self.in_channels, self.kernel_size, self.kernel_size
        )
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
        else:
            aggregate_bias = None
        output = F.conv2d(
            x,
            weight=aggregate_weight,
            bias=aggregate_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * batch_size,
        )
        output = output.view(
            batch_size, self.out_channels, output.size(-2), output.size(-1)
        )
        return output
#########################################-----定义尺度敏感动态卷积(SAD-Conv)------#########################################



#########################################-----LDG的基本模块(SARB)------#########################################
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class WideConvBlock(nn.Module):
    def __init__(self, num_features, kernel_size, width_multiplier=4, reduction=4):
        super().__init__()

        self.body = nn.Sequential(
            *[
                WeightNormedConv(
                    num_features, int(num_features * width_multiplier), 3
                ),
                WeightNormedConv(
                    int(num_features * width_multiplier), num_features, 3, act=None
                ),
                WeightNormedConv(
                    num_features,
                    num_features,
                    kernel_size,
                    act=None,
                    # res_scale=res_scale,
                ),
                SEBlock(num_features, reduction),
            ]
        )

    def forward(self, x, scale):
        return x + self.body(x)


class DynamicWideConvBlock(nn.Module):
    def __init__(
        self,
        num_features,
        kernel_size,
        width_multiplier=4,
        dynamic_K=4,
        reduction=4,
    ):
        super().__init__()

        self.body = nn.Sequential(
            *[
                WeightNormedConv(
                    num_features,
                    int(num_features * width_multiplier),
                    kernel_size,
                    # res_scale=2.0,
                ),
                WeightNormedConv(
                    int(num_features * width_multiplier),
                    num_features,
                    kernel_size,
                    act=None,
                ),
            ]
        )
        self.d_conv = weight_norm(
            ScaleAwareDynamicConv2d(
                num_features,
                num_features,
                kernel_size,
                padding=kernel_size // 2,
                K=dynamic_K,
            )
        )
        self.se_block = SEBlock(num_features, reduction)

    def forward(self, x, scale):
        r = self.body(x)
        r = self.d_conv(r, scale)
        r = self.se_block(r)
        return x + r
#########################################-----LDG的基本模块(SARB)------#########################################



#########################################-----SADN基础模块(LDG)------#########################################
class LocalDenseGroup(nn.Module):
    def __init__(
        self,
        num_features,
        width_multiplier,
        num_layers,
        reduction,
        use_dynamic_conv,
        dynamic_K,
    ):
        super().__init__()
        kSize = 3
        self.num_layers = num_layers

        self.ConvBlockList = nn.ModuleList()
        self.compressList = nn.ModuleList()
        self.use_dynamic_conv = use_dynamic_conv
        for idx in range(num_layers):
            if use_dynamic_conv:
                self.ConvBlockList.append(
                    DynamicWideConvBlock(
                        num_features,
                        kSize,
                        width_multiplier=width_multiplier,
                        # res_scale=1 / math.sqrt(num_layers),
                        dynamic_K=dynamic_K,
                        reduction=reduction,
                    )
                )
            else:
                self.ConvBlockList.append(
                    WideConvBlock(
                        num_features,
                        kSize,
                        width_multiplier=width_multiplier,
                        # res_scale=1 / math.sqrt(num_layers),
                        reduction=reduction,
                    )
                )
        for idx in range(1, num_layers):
            self.compressList.append(
                WeightNormedConv(
                    (idx + 1) * num_features, num_features, 1, act=None
                )
            )

    def forward(self, x, scale):
        concat = x
        for l in range(self.num_layers):
            if l == 0:
                out = self.ConvBlockList[l](concat, scale)
            else:
                concat = torch.cat([concat, out], dim=1)
                out = self.compressList[l - 1](concat)
                out = self.ConvBlockList[l](out, scale)
        return out
#########################################-----SADN基础模块(LDG)------#########################################



#########################################-----多个LDG组成反馈模块------#########################################
class FeedbackBlock(nn.Module):
    def __init__(
        self,
        num_features,
        width_multiplier,
        num_layers,
        num_groups,
        reduction,
        use_dynamic_conv,
        dynamic_K,
    ):
        super().__init__()
        kSize = 3
        self.num_groups = num_groups

        self.LDGList = nn.ModuleList()
        for _ in range(num_groups):
            self.LDGList.append(
                LocalDenseGroup(
                    num_features,
                    width_multiplier,
                    num_layers,
                    reduction,
                    use_dynamic_conv,
                    dynamic_K,
                )
            )

        self.compressList = nn.ModuleList()
        for idx in range(1, num_groups):
            self.compressList.append(
                WeightNormedConv(
                    (idx + 1) * num_features, num_features, 1, act=None
                )
            )

        self.compress_in = WeightNormedConv(
            2 * num_features, num_features, kSize
        )

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x, scale):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size(), device=x.device)
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), 1)

        concat = self.compress_in(x)
        for l in range(self.num_groups):
            if l == 0:
                out = self.LDGList[l](concat, scale)
            else:
                concat = torch.cat([concat, out], dim=1)
                out = self.compressList[l - 1](concat)
                out = self.LDGList[l](out, scale)

        self.last_hidden = out
        return out

    def reset_state(self):
        self.should_reset = True
#########################################-----多个LDG组成反馈模块------#########################################



class SADN(nn.Module):
    def __init__(
        self,
        in_channels=3, # 输入维度
        #out_channels, # 输出维度没必要
        num_features=96, # 通道数
        num_layers=4, # FeedBackBlock中LDG数
        num_groups=4, # LDG中SARB数
        reduction=4, # SEBlock中有关通道变换参数
        width_multiplier=4, 
        #interpolate_mode,
        levels=4, # 特征网格
        use_dynamic_conv=True,
        dynamic_K=3,
        #which_uplayer, #TODO 决定采用的上采样模块
        #uplayer_ksize,
        #rgb_range,
        #rgb_mean,
        #rgb_std,
    ):
        super().__init__()
        kernel_size = 3
        skip_kernel_size = 5
        num_inputs = in_channels
        #self.interpolate_mode = interpolate_mode
        self.levels = levels

        num_feats = num_features # 96
        self.out_dim = num_feats


        # 有关图像标准化
        #self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        #self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(
            *[WeightNormedConv(num_inputs, num_features, kernel_size)]
        )

        self.body = FeedbackBlock(
            num_features,
            width_multiplier,
            num_layers,
            num_groups,
            reduction,
            use_dynamic_conv,
            dynamic_K,
        )

        self.tail = nn.Sequential(
            *[
                WeightNormedConv(
                    num_features, num_features, kernel_size, act=None
                )
            ]
        )

        self.skip = WeightNormedConv(
            num_inputs, num_features, skip_kernel_size, act=None
        )

        #TODO 上采样部分逻辑
        # UpLayer = getattr(upsampler, which_uplayer)
        # self.uplayer = UpLayer(
        #     n_feats,
        #     uplayer_ksize,
        #     out_channels,
        #     interpolate_mode,
        #     levels,
        # )

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, ScaleAwareDynamicConv2d):
                m.update_temperature()


    #TODO 这个out_size如何获取
    def forward(self, x, out_size):

        self.body.reset_state()
        if isinstance(out_size, int):
            out_size = [out_size, out_size]
        scale = torch.tensor([x.shape[2] / out_size[0]], device=x.device)

        #x = self.sub_mean(x)

        skip = self.skip(x)

        x = self.head(x)
        h_list = []

        #TODO 此处体现了SAFL的多次迭代用于获取不同尺度特征网格
        for _ in range(self.levels):
            h = self.body(x, scale) # 尺度感知特征提取主干
            h = self.tail(h)
            h = h + skip
            h_list.append(h)

        #x = self.uplayer(h_list, out_size)

        #x = self.add_mean(x)

        return h_list



# 相比于SADN增加了可视化部分代码
# class SADN_vis(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         num_features,
#         num_layers,
#         num_groups,
#         reduction,
#         width_multiplier,
#         interpolate_mode,
#         levels,
#         use_dynamic_conv,
#         dynamic_K,
#         which_uplayer,
#         uplayer_ksize,
#         rgb_range,
#         rgb_mean,
#         rgb_std,
#     ):
#         super().__init__()
#         kernel_size = 3
#         skip_kernel_size = 5
#         num_inputs = in_channels
#         n_feats = num_features
#         self.interpolate_mode = interpolate_mode
#         self.levels = levels

#         self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
#         self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

#         self.head = nn.Sequential(
#             *[WeightNormedConv(num_inputs, num_features, kernel_size)]
#         )

#         self.use_dynamic_conv = use_dynamic_conv
#         self.body = FeedbackBlock(
#             num_features,
#             width_multiplier,
#             num_layers,
#             num_groups,
#             reduction,
#             use_dynamic_conv,
#             dynamic_K,
#         )

#         self.tail = nn.Sequential(
#             *[
#                 WeightNormedConv(
#                     num_features, num_features, kernel_size, act=None
#                 )
#             ]
#         )

#         self.skip = WeightNormedConv(
#             num_inputs, num_features, skip_kernel_size, act=None
#         )

#         UpLayer = getattr(upsampler, which_uplayer)
#         self.uplayer = UpLayer(
#             n_feats,
#             uplayer_ksize,
#             out_channels,
#             interpolate_mode,
#             levels,
#         )

#     def update_temperature(self):
#         for m in self.modules():
#             if isinstance(m, ScaleAwareDynamicConv2d):
#                 m.update_temperature()

#     def forward(self, x, out_size):
#         self.body.reset_state()
#         if isinstance(out_size, int):
#             out_size = [out_size, out_size]
#         scale = torch.tensor([x.shape[2] / out_size[0]], device=x.device)
#         x = self.sub_mean(x)
#         skip = self.skip(x)

#         x = self.head(x)
#         h_list = []

#         for _ in range(self.levels):
#             h = self.body(x, scale)
#             h = self.tail(h)
#             h = h + skip
#             h_list.append(h)
#         vis = torch.mean(h_list[-1], dim=1)
#         vis = (vis - vis.min()) / (vis.max() - vis.min())
#         vis = vis[..., 88:217, 32:161]
#         # vis = vis + 0.2
#         # vis.clamp_max_(1)
#         print(torch.min(vis), torch.max(vis))
#         # print(vis.shape)

#         savepath = "logs/vis"
#         filename = "geo_residential_t7.png"

#         if self.use_dynamic_conv:
#             savepath = os.path.join(savepath, "dy" + filename.replace(".png", ""))
#         else:
#             savepath = os.path.join(savepath, "wo_dy" + filename.replace(".png", ""))
#         if not exists(savepath):
#             os.mkdir(savepath)

#         savepath = os.path.join(savepath, "x{0}.png".format(int((1 / scale).item())))

#         plt.imsave(savepath, vis.cpu().numpy()[0], cmap="hsv")

#         x = self.uplayer(h_list, out_size)

#         x = self.add_mean(x)

#         return x


@register('sadn')
def make_sadn(no_upsampling=True):
    return SADN()



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
    out_size = 256
    x=torch.zeros((1, 3, 128, 128)).type(torch.FloatTensor).cuda() 
    model = SADN() 
    model.cuda() 
    
    since = time.time()
    y=model(x, out_size)
    print(len(y)) # 4
    print(y[0].shape) # [1, 96, 128, 128]
    print("time", time.time()-since) 
    
    flops, params = profile(model, inputs=(x, out_size))  
    flops, params = clever_format([flops, params], '%.6f') 
    print('flops',flops)
    print('params', params) 
    print(count_parameters(model)/1e6)
    # print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    # print("Params=", str(params/1e6)+'{}'.format("M"))
