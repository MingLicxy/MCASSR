import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


import models
from models import register
from utils import make_coord

################################ LIT(non-cascaded) ##################################
@register('lit')
class LIT(nn.Module):
    def __init__(
        self,
        encoder_spec,
        imnet_spec=None,
        is_cell=True,
        local_ensemble=False,
        local_attn=True,
        base_dim=192,
        head=8,
        r=3,
        pe_spec=None,
        pb_spec=None
    ):
        super().__init__()
        self.is_cell = is_cell
        self.local_ensemble = local_ensemble
        self.local_attn = local_attn

        # 编码器
        self.encoder = models.make(encoder_spec)

        self.dim = base_dim
        self.head = head
        self.r = r
        self.conv_v = nn.Conv2d(self.encoder.out_dim, self.dim, kernel_size=3, padding=1)
        if self.local_attn:
            self.conv_q = nn.Conv2d(self.encoder.out_dim, self.dim, kernel_size=3, padding=1)
            self.conv_k = nn.Conv2d(self.encoder.out_dim, self.dim, kernel_size=3, padding=1)
            self.is_pb = True if pb_spec else False
            if self.is_pb:
                # 位置编码（head=8）
                self.pb_encoder = models.make(pb_spec, args={'head': self.head}).cuda()
            #self.r = 3
        else:
            self.r = 0
        # 局部区域
        self.r_area = (2 * self.r + 1)**2
        
        # 频域卷积以及频域坐标编码器
        self.pe_encoder = models.make(pe_spec).cuda()
        
        #TODO conv_freq决定feat_freq的最后一维（//2为了配合complex_transform）
        self.conv_freq = nn.Conv2d(
            self.encoder.out_dim, self.pe_encoder.enc_dims // 2, kernel_size=3, padding=1
        )
        
        # 解码器
        if self.is_cell:
            self.imnet = models.make(
                imnet_spec,
                args={'in_dim': (self.dim + self.pe_encoder.enc_dims) * self.r_area + 2}
            )
        else:
            self.imnet = models.make(
                imnet_spec, args={'in_dim': (self.dim + self.pe_encoder.enc_dims) * self.r_area}
            )

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)

        self.feat_v = self.conv_v(self.feat)
        if self.local_attn:
            self.feat_q = self.conv_q(self.feat)
            self.feat_k = self.conv_k(self.feat)

        # 获取频域特征
        self.feat_freq = self.conv_freq(self.feat)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        bs, q = coord.shape[:2]

        # [b, q, 1, 2]
        coord = coord.unsqueeze(2)

        coord_lr = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1). \
                              unsqueeze(0).expand(bs, 2, *feat.shape[-2:])

        # local ensamble - field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # 局部集成（同LIIF和LTE）
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        r = self.r

        #TODO 计算局部注意力的关键
        if self.local_attn:
            dh = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-2]
            dw = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-1]
            # 1, 1, r_area, 2
            delta = torch.stack(torch.meshgrid(dh, dw, indexing='ij'), axis=-1).view(1, 1, -1, 2)

        ############################# LIT与CLIT1的区别在于最终的局部聚合方式 ###############################
        areas = []
        preds = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # [b, 2, h, w] -> [b, 2, q, 1] -> [b, q, 1, 2]
                coord_k = F.grid_sample(
                    coord_lr, coord_.flip(-1), mode='nearest', align_corners=False
                ).permute(0, 2, 3, 1)

                # 计算局部聚合相对坐标和区域面积
                ensamble_coord = coord - coord_k
                ensamble_coord[:, :, :, 0] *= feat.shape[-2]
                ensamble_coord[:, :, :, 1] *= feat.shape[-1]
                area = torch.abs(ensamble_coord[:, :, 0, 0] * ensamble_coord[:, :, 0, 1])
                areas.append(area + 1e-9)

                if self.local_attn:
                    # Q:[b, c, h, w] -> [b, c, q, 1] -> [b, q, 1, c] -> [b, q, 1, head, c] -> [b, q, head, 1, c]
                    feat_q = F.grid_sample(
                        self.feat_q,
                        coord_.flip(-1),
                        mode='nearest' if self.local_ensemble else 'bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                    feat_q = feat_q.reshape(bs, q, 1, self.head,
                                            self.dim // self.head).permute(0, 1, 3, 2, 4)

                    # [b, q, 1, 2] -> [b, q, r_area, 2]
                    coord_k = coord_k + delta

                    # K:[b, c, h, w] -> [b, c, q, r_area] -> [b, q, r_area, c] -> [b, q, r_area, head, c] -> [b, q, head, c, r_area]
                    feat_k = F.grid_sample(
                        self.feat_k, coord_k.flip(-1), mode='nearest', align_corners=False
                    ).permute(0, 2, 3, 1)
                    feat_k = feat_k.reshape(bs, q, self.r_area, self.head,
                                            self.dim // self.head).permute(0, 1, 3, 4, 2)

                    # V:[b, c, h, w] -> [b, c, q, r_area] -> [b, q, r_area, c]
                    feat_v = F.grid_sample(
                        self.feat_v, coord_k.flip(-1), mode='nearest', align_corners=False
                    ).permute(0, 2, 3, 1)

                else:
                    feat_v = F.grid_sample(
                        self.feat_v, coord_.flip(-1), mode='nearest', align_corners=False
                    ).permute(0, 2, 3, 1)
                
                ##################################TODO 采样频域特征(采用最邻近插值) ################################
                feat_freq = F.grid_sample(
                    self.feat_freq, coord_k.flip(-1), mode='nearest', align_corners=False
                ).permute(0, 2, 3, 1)

                # [b, q, r_area, 2] 局部注意力相对坐标
                rel_coord = coord - coord_k # [16, 2304, 49, 2]
                rel_coord[..., 0] *= feat.shape[-2]
                rel_coord[..., 1] *= feat.shape[-1]
                #print('########################## rel_coord #########################', rel_coord.shape)

                # [b, q, 2] 像素尺寸
                rel_cell = cell.clone()
                rel_cell[..., 0] *= feat.shape[-2]
                rel_cell[..., 1] *= feat.shape[-1]
                

                # 计算局部自注意力
                if self.local_attn:
                    # [b, q, head, 1, r_area] -> [b, q, r_area, head]
                    similarity = torch.matmul(feat_q, feat_k).reshape(
                        bs, q, self.head, self.r_area
                    ).permute(0, 1, 3, 2) / np.sqrt(self.dim // self.head)
                    # 注入位置编码
                    if self.is_pb:
                        _, pb = self.pb_encoder(rel_coord) # [16, 2304, 49, 8]
                        #print('########################## pb_encoder #########################', pb.shape)

                        #TODO 注入位置编码的方式（影响注意力图）
                        attn = F.softmax(similarity + pb, dim=-2)
                        #attn = F.softmax(torch.add(similarity, pb), dim=-2)
                    else:
                        attn = F.softmax(similarity, dim=-2)
                    attn = attn.reshape(bs, q, self.r_area, self.head, 1)
                    feat_v = feat_v.reshape(bs, q, self.r_area, self.head, self.dim // self.head)
                    feat_v = torch.mul(feat_v, attn).reshape(bs, q, self.r_area, -1)

                    attn_map = attn[0, 10, :, 0, :].reshape(2 * r + 1, 2 * r + 1, 1)
                
                ################################### 频域特征处理 #####################################
                feat_freq = feat_freq.reshape(bs, q, 2 * r + 1, 2 * r + 1, -1)

                #BUG 频域转换操作对于GPU显存要求非常高
                #TODO 解决方案一（gpu->cpu->gpu）速度太慢
                feat_freq = feat_freq.cpu()
                feat_freq = torch.fft.fft2(feat_freq, dim=(2, 3), norm='ortho').cuda()
                #TODO 解决方案二（numpy）.detach()导致显存不足
                #feat_freq_np = np.fft.fft2(feat_freq.detach().cpu().numpy(), axes=(2, 3), norm='ortho')
                #feat_freq = torch.tensor(feat_freq_np).to(feat_freq.device)
            
                #feat_freq = torch.fft.fft2(feat_freq, dim=(2, 3), norm='ortho')
                feat_freq = feat_freq.reshape(bs, q, self.r_area, -1) # [16, 2304, 49, 64] torch.complex64
                #print('########################## feat_freq.shape #########################', feat_freq.shape)
                #print('########################## feat_freq.dtype #########################', feat_freq.dtype)

                #TODO 利用频域位置编码器输出调制频域信息(project)
                rel_enc, _ = self.pe_encoder(rel_coord) # [16, 2304, 49, 64]
                #print('########################## pe_encoder #########################', rel_enc.shape)
                rel_enc.mul_(feat_freq) # 张量乘法
                # # 将复数表示转化为实部和虚部：1.+4.j => [1,4]
                rel_enc = torch.view_as_real(rel_enc) 
                rel_enc = rel_enc.reshape(bs, q, self.r_area, -1)
                
                #TODO 注入频域特征的方式
                out = torch.cat([feat_v, rel_enc], dim=-1)

                out = out.reshape(bs, q, -1)

                # 像素单元尺寸
                if self.is_cell:
                    out = torch.cat([out, rel_cell], dim=-1)

                pred = self.imnet(out)
                # preds对应局部集成策略中不同偏移位置的pred
                preds.append(pred)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        ret = 0
        # 局部集成
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        # 残差连接
        ret +=  F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
                        padding_mode='border', align_corners=False)[:, :, :, 0].permute(0, 2, 1)
        #return ret, attn_map
        return ret

    def forward(self, inp, coord, cell):
        self.inp = inp
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)