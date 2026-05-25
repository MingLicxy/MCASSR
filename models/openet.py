import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import models
from models import register
import time

################################################### OPE-net(OPE-SR) ###################################################

# 生成坐标网格
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    follow the coord system as:
          +
          +
    +  +  +  +  + >
          +        y
          +
          V x
    need to use flip(-1) when use grid_sample to change to
          +
          +
    +  +  +  +  + >
          +        x
          +
          V y
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:  # H,W,2 ---> H*W,2
        ret = ret.view(-1, ret.shape[-1])
    return ret

#TODO 生成位置嵌入函数（可以看作一种位置编码方式）
def get_embed_fns(max_freq):
    """
    N,bsize,1 ---> N,bsize,2n+1
    """
    embed_fns = []
    embed_fns.append(lambda x: torch.ones((x.shape[0], x.shape[1], 1)))  # x: N,bsize,1
    for i in range(1, max_freq + 1):
        embed_fns.append(lambda x, freq=i: math.sqrt(2) * torch.cos(x[:, :, 0] * freq).unsqueeze(-1))
        embed_fns.append(lambda x, freq=i: math.sqrt(2) * torch.sin(x[:, :, 0] * freq).unsqueeze(-1))
    return embed_fns 


class OPE(nn.Module):
    def __init__(self, max_freq, omega):
        super(OPE, self).__init__()
        self.max_freq = max_freq
        self.omega = omega # 缩放因子
        self.embed_fns = get_embed_fns(self.max_freq) # 获取嵌入函数列表

    def embed(self, inputs):
        """
        N,bsize,1 ---> N,bsize,1,2n+1
        """
        res = torch.cat([fn(inputs * self.omega).to(inputs.device) for fn in self.embed_fns], -1)
        return res.unsqueeze(-2) # (N, bsize, 1, 2n+1)

    def forward(self, coords):
        """
        N,bsize,2 ---> N,bsize,(2n+1)^2
        """
        # 分别计算x和y的嵌入
        x_coord = coords[:, :, 0].unsqueeze(-1) # (N, bsize, 1)
        y_coord = coords[:, :, 1].unsqueeze(-1)
        X = self.embed(x_coord) # (N, bsize, 1, 2n+1)
        Y = self.embed(y_coord)

        # 计算X和Y的外积
        ope_mat = torch.matmul(X.transpose(2, 3), Y) # # (N, bsize, 2n+1, 2n+1)
        ope_flat = ope_mat.view(ope_mat.shape[0], ope_mat.shape[1], -1) # (N, bsize, (2n+1)^2)
        return ope_flat


class LC_OPE(nn.Module):
    """
    linear combination of OPE with 3 channels
    """

    def __init__(self, max_freq, omega):
        super(LC_OPE, self).__init__()
        self.max_freq = max_freq
        self.omega = omega
        self.c = (2 * max_freq + 1) ** 2 # c=(2n+1)^2
        self.ope = OPE(max_freq=max_freq, omega=omega) 

    def forward(self, latent, rel_coord):
        """
        N,bsize,ccc N,bsize,ccc ---> N,bsize,3
        """
        c = int(latent.shape[-1] // 3) # latent: (N, bsize, 3*(2n+1)^2)
        assert c == self.c

        #TODO ope处理相对坐标
        ope_flat = self.ope(rel_coord).unsqueeze(-2) # (N, bsize, 1, (2n+1)^2)

        # 将latent按R、G、B通道分割
        latent_R = latent[:, :, :c].unsqueeze(-1) # (N, bsize, (2n+1)^2, 1)
        latent_G = latent[:, :, c:c * 2].unsqueeze(-1)
        latent_B = latent[:, :, c * 2:].unsqueeze(-1)

        R = torch.matmul(ope_flat, latent_R).squeeze(-1) # (N, bsize, 1)
        G = torch.matmul(ope_flat, latent_G).squeeze(-1)
        B = torch.matmul(ope_flat, latent_B).squeeze(-1)
        ans = torch.cat([R, G, B], dim=-1) # (N, bsize, 3)
        return ans


class OPE_render_intp(nn.Module):
    """
    omega=0.5*pi
    use local ensemble
    """

    def __init__(self, max_freq, omega=0.5 * math.pi):
        super(OPE_render_intp, self).__init__()
        self.max_freq = max_freq
        self.omega = omega

        self.lc_ope = LC_OPE(max_freq=self.max_freq, omega=self.omega)

    def query_rgb(self, feat, coord):
        # feat: N,3C,h,w / coord:  N,bsize,2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = 2 / feat.shape[-2] / 2  # half pixel of feature map
        ry = 2 / feat.shape[-1] / 2

        # 获取LR对应坐标网格
        # get the center point coord of feature map
        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
            .permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])  # N,2,h,w

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()  # N,bsize,2
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # 对坐标网格与特征进行最邻近插值
                fourier_projection_tmp = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)  # N,bsize,3C
                q_coord = F.grid_sample(  # get feature map 'pixel'
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                # 相对坐标
                rel_coord = coord - q_coord  # N,bsize,2
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]  # scale to -2,2; notice omega=0.5
                
                #TODO 利用lc_ope张量内积代替MLP解码器
                pred = self.lc_ope(fourier_projection_tmp, rel_coord)  # N,bsize,3

                preds.append(pred)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)  # area: N,30000

        tot_area = torch.stack(areas).sum(dim=0)  # areas: 4*N,bsize tot_area: 1*N,30000

        t = areas[0];
        areas[0] = areas[3];
        areas[3] = t
        t = areas[1];
        areas[1] = areas[2];
        areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret  # N,bsize,3

    def batched_predict(self, inp, coord, bsize):
        n = coord.shape[1]  # pixels
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.query_rgb(inp, coord[:, ql: qr, :])  # query_rgb : N,bsize,2 ---> N,bsize,3
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)

        return pred

    def forward(self, img_feature, h, w, bsize=30000):
        coord = make_coord((h, w)).to(img_feature.device)  # H*W,2
        N = img_feature.shape[0]
        pred = self.batched_predict(img_feature,
                                    coord.unsqueeze(0).repeat(N, 1, 1),
                                    bsize=bsize)  # input: UV，output: RGB
        pred = pred.view(N, h, w, 3).permute(0, 3, 1, 2)  # N,pixels,3  --->  N,C,H,W

        return pred

################################ OPE-net ##################################
@register('OPE-net')
class OPE_net(nn.Module):
    def __init__(self, max_freq, srnet_spec=None):
        super(OPE_net, self).__init__()
        self.max_freq = max_freq
        self.render = OPE_render_intp(max_freq=max_freq)
        self.srnet = None
        if srnet_spec is not None:
            # 特征提取主干
            self.srnet = models.make(srnet_spec)

    def forward(self, img_lr, sample_coords, bsize=30000):
        if self.srnet is None:
            raise RuntimeError('not init srnet')
        feature = self.srnet(img_lr)
        pixel_sampled = self.render.batched_predict(feature, sample_coords, bsize)
        return pixel_sampled

    def inference(self, img_lr, h, w, bsize=30000):
        if self.srnet is None:
            raise RuntimeError('not init srnet')
        torch.cuda.synchronize()
        time_start = time.time()
        feature = self.srnet(img_lr)
        torch.cuda.synchronize()
        time_encoder = time.time()
        ans = self.render(feature, h, w, bsize)
        torch.cuda.synchronize()
        time_render = time.time()
        return ans, [time_encoder - time_start, time_render - time_encoder, time_render - time_start]


################################ OPE-net-adapt ##################################
@register('OPE-net-adapt')
class OPE_net_adapt(nn.Module):
    def __init__(self, max_freq, srnet_spec=None):
        super(OPE_net_adapt, self).__init__()
        self.max_freq = max_freq
        self.render = OPE_render_intp(max_freq=max_freq)
        self.C = (2 * max_freq + 1) ** 2 # 位置嵌入的最后一维
        self.srnet = None

        #TODO
        if srnet_spec is not None:
            self.srnet = models.make(srnet_spec)
            self.adapt_R = nn.Conv2d(self.srnet.out_dim, self.C, 3, padding=1)
            self.adapt_G = nn.Conv2d(self.srnet.out_dim, self.C, 3, padding=1)
            self.adapt_B = nn.Conv2d(self.srnet.out_dim, self.C, 3, padding=1)


    def forward(self, img_lr, sample_coords, bsize=30000):
        if self.srnet is None:
            raise RuntimeError('not init srnet')
        
        #TODO 利用Conv从latent code中分别提取RGB三通道特征
        feature = self.srnet(img_lr)
        f_R = self.adapt_R(feature)
        f_G = self.adapt_G(feature)
        f_B = self.adapt_B(feature)
        feature = torch.cat([f_R, f_G, f_B], dim=-3)
        pixel_sampled = self.render.batched_predict(feature, sample_coords, bsize)
        return pixel_sampled

    def inference(self, img_lr, h, w, bsize=30000):
        if self.srnet is None:
            raise RuntimeError('not init srnet')
        torch.cuda.synchronize()
        time_start = time.time()
        feature = self.srnet(img_lr)
        f_R = self.adapt_R(feature)
        f_G = self.adapt_G(feature)
        f_B = self.adapt_B(feature)
        feature = torch.cat([f_R, f_G, f_B], dim=-3)
        torch.cuda.synchronize()
        time_encoder = time.time()
        ans = self.render(feature, h, w, bsize)
        torch.cuda.synchronize()
        time_render = time.time()
        return ans, [time_encoder - time_start, time_render - time_encoder, time_render - time_start]


if __name__ == "__main__":
    pass
