"""
Modified from https://github.com/caojiezhang/CiaoSR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
from models.arch_ciaosr.arch_csnln import CrossScaleAttention


################################ CiaoSR_MC（针对多对比MRI） ##################################
@register('ciaosr_mc_try1')
class CiaoSR_MC_TRY1(nn.Module):
    """
    The subclasses should define `generator` with `encoder` and `imnet`,
        and overwrite the function `gen_feature`.
    If `encoder` does not contain `mid_channels`, `__init__` should be
        overwrite.

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 ref_encoder, # .yaml中要多一个选项
                 imnet_q,
                 imnet_k,
                 imnet_v,
                 imnet_c, 
                 local_size=2, # 与计算资源最相关
                 feat_unfold=True,
                 non_local_attn=True,
                 multi_scale=[2],
                 softmax_scale=1, # 与注意力计算相关
                 ):
        super().__init__()

        self.feat_unfold = feat_unfold
        self.local_size = local_size # 局部特征聚合区域
        self.non_local_attn = non_local_attn # 非局部注意力
        self.multi_scale = multi_scale # 哪几种尺度
        self.softmax_scale = softmax_scale # 用于注意力计算归一化

        # 编码器（LR混合特征编码器与Ref_HR特征编码器）
        self.encoder = models.make(encoder, args={'no_upsampling': True})
        self.ref_encoder = models.make(ref_encoder, args={'no_upsampling': True}) # 即处理ref_hr又处理ref_lr
        
        #TODO 定义各网络输入输出维度
        
        # 两个基础维度一般都设置成64
        imnet_dim = self.encoder.out_dim # self.encoder.embed_dim if hasattr(self.encoder, 'embed_dim') else self.encoder.out_dim
        imnet_dim_ref = self.ref_encoder.out_dim

        # 特征展开
        if self.feat_unfold:
            imnet_q_in_dim = imnet_dim_ref * 9 + imnet_dim_ref # 64x9+64=640
            imnet_q_out_dim = imnet_dim * 9 # 576
            imnet_k_in_dim = imnet_k_out_dim = imnet_dim * 9
            imnet_v_in_dim = imnet_v_out_dim = imnet_dim * 9
            imnet_c_in_dim = imnet_dim * 9
        else:
            imnet_q_in_dim= imnet_dim_ref + imnet_dim_ref # 64+64=128
            imnet_q_out_dim = imnet_dim # 64
            imnet_k_in_dim = imnet_k_out_dim = imnet_dim
            imnet_v_in_dim = imnet_v_out_dim = imnet_dim
            imnet_c_in_dim = imnet_dim

        imnet_k_in_dim += 4 # 576+4=580
        imnet_v_in_dim += 4

        
        # 多尺度特征增强
        if self.non_local_attn:
            imnet_v_in_dim += imnet_dim * len(multi_scale) # 注意力图由qk计算，他们的通道维要对齐
            imnet_v_out_dim += imnet_dim * len(multi_scale)
            imnet_c_in_dim += imnet_dim * len(multi_scale)

        # Q,K,V编码器
        self.imnet_q = models.make(imnet_q, args={'in_dim': imnet_q_in_dim, 'out_dim': imnet_q_out_dim}) 
        self.imnet_k = models.make(imnet_k, args={'in_dim': imnet_k_in_dim, 'out_dim': imnet_k_out_dim})
        self.imnet_v = models.make(imnet_v, args={'in_dim': imnet_v_in_dim, 'out_dim': imnet_v_out_dim})
        self.imnet_c = models.make(imnet_c, args={'in_dim': imnet_c_in_dim}) # 最终的解码器，输出维度为3

        if self.non_local_attn:
            self.non_local_attn_dim = imnet_dim * len(multi_scale)
            # 多尺度非局部注意力（Scale-aware Attention）
            self.cs_attn = CrossScaleAttention(channel=imnet_dim, scale=multi_scale)

        self.feat_coord = None


    #TODO 适应多输入（如何充分利用Ref图像对应的HR与LR）
    def gen_feat(self, inp, ref_lr, ref_hr):
        self.inp = inp # tar_lr
        self.ref_lr = ref_lr
        self.ref_hr = ref_hr
        
        #TODO 特征提取主干应当接收同尺度的tar_lr与ref_lr作为输入（输出的特征图尺度与LR保持一致）
        # feat = self.encoder(inp, ref_lr)

        #TODO 这里ref_hr不发挥作用，利用通道连接作为模态融合方式
        inp_fus = torch.cat((inp, ref_lr), dim=1) # [1, 4, 135, 180]
        feat = self.encoder(inp_fus)

        # self.feat_ref_hr = self.ref_encoder(ref_hr)
        # 方案一：
        # self.feat_ref_lr = self.ref_encoder(ref_lr)
        # 方案二；共享参数
        # with torch.no_grad():  
        #     self.feat_ref_lr = self.ref_encoder(ref_lr)
        '''
        if hasattr(self.encoder, 'embed_dim'):
            # SwinIR
            feat = self.encoder.check_image_size(inp)
            feat = self.encoder.conv_first(feat)
            feat = self.encoder.conv_after_body(self.encoder.forward_features(feat)) + feat
        else:
            feat = self.encoder(inp)
        '''

        # 获取LR对应坐标网格
        if self.training or self.feat_coord is None or self.feat_coord.shape[-2] != feat.shape[-2] \
                or self.feat_coord.shape[-1] != feat.shape[-1]:
            self.feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
                .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        B, C, H, W = feat.shape

        #TODO Scale-aware Attention聚合非局部多尺度特征（操作对象是feat）
        if self.non_local_attn:
            crop_h, crop_w = 48, 48 #TODO patch-size
            if H * W > crop_h * crop_w:
                # Fixme: generate cross attention by image patches
                self.non_local_feat = torch.zeros(B, self.non_local_attn_dim, H, W).cuda()
                for i in range(H // crop_h):
                    for j in range(W // crop_w):
                        i1, i2 = i * crop_h, ((i + 1) * crop_h if i < H // crop_h - 1 else H)
                        j1, j2 = j * crop_w, ((j + 1) * crop_w if j < W // crop_w - 1 else W)

                        padding = 3 // 2
                        pad_i1, pad_i2 = (padding if i1 - padding >= 0 else 0), (
                            padding if i2 + padding <= H else 0)
                        pad_j1, pad_j2 = (padding if j1 - padding >= 0 else 0), (
                            padding if j2 + padding <= W else 0)

                        crop_feat = feat[:, :, i1 - pad_i1:i2 + pad_i2, j1 - pad_j1:j2 + pad_j2]

                        # 计算patch的非局部特征，并将结果存入self.non_local_feat_v的对应位置
                        crop_non_local_feat = self.cs_attn(crop_feat)
                        self.non_local_feat[:, :, i1:i2, j1:j2] = crop_non_local_feat[:, :,
                                                               pad_i1:crop_non_local_feat.shape[-2] - pad_i2,
                                                               pad_j1:crop_non_local_feat.shape[-1] - pad_j2]
            else:
                self.non_local_feat = self.cs_attn(feat)  # [16, 64, 48, 48]

        self.feats = [feat]
        return self.feats
    
    # coord: inp_hr_coord; scale: inp_cell;
    def query_rgb(self, coord, scale):
        """Query RGB value of GT.

        Copyright (c) 2020, Yinbo Chen, under BSD 3-Clause License.

        Args:
            feature (Tensor): encoded feature.
            coord (Tensor): coord tensor, shape (BHW, 2).

        Returns:
            result (Tensor): (part of) output.
        """
        # feat_ref_hr = self.feat_ref_hr # 想办法将其与Q联系起来
        # feat_ref_lr = self.feat_ref_lr

        res_features = []
        for feature in self.feats: #TODO 遍历的目的？
            B, C, H, W = feature.shape  # [16, 64, 48, 48]

            if self.feat_unfold:
                feat_q = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)  # [16, 576, 48, 48]
                feat_k = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)  # [16, 576, 48, 48]
                
                #TODO 这里只有feat_v集成了非局部多尺度特征
                if self.non_local_attn:
                    feat_v = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)  # [16, 576, 48, 48]
                    # 集成非局部多尺度特征
                    feat_v = torch.cat([feat_v, self.non_local_feat], dim=1)  # [16, 576+64, 48, 48]
                else:
                    feat_v = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)  # [16, 576, 48, 48]
            else:
                feat_q = feat_k = feat_v = feature

            ###################################TODO 有待改进 TODO####################################
            # Q: 最邻近采样查询点特征/双线性插值更合适？ ['nearest', 'bilinear']
            # 利用(Ref_HR与Ref_LR的残差/直接Ref_HR)作为Q来引导局部特征聚合 / 将Ref_HR作为V的一部分
            #query = F.grid_sample(feat_q, coord_.flip(-1).unsqueeze(1), mode='nearest',
            #                        align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous() # # [16, 2304, 576]
            query = F.grid_sample(feat_q, coord.flip(-1).unsqueeze(1), mode='nearest',
                                  align_corners=False).permute(0, 3, 2, 1).contiguous()  # [16, 2304, 1, 576]
            # query_ref_hr = F.grid_sample(feat_ref_hr, coord.flip(-1).unsqueeze(1), mode='nearest',
            #                       align_corners=False).permute(0, 3, 2, 1).contiguous()  # [16, 2304, 1, 64]
            # query_ref_lr = F.grid_sample(feat_ref_lr, coord.flip(-1).unsqueeze(1), mode='nearest',
            #                       align_corners=False).permute(0, 3, 2, 1).contiguous()  # [16, 2304, 1, 64]
            # query_ref_res = query_ref_hr - query_ref_lr

            #TODO 方案一：query投影降维
            # query = self.imnet_q(torch.cat((query, query_ref_res), dim=-1)) # [16, 2304, 1, 640]->[16, 2304, 1, 576]
            
            #TODO 方案二：采用与kv一样的meta-learning的思想
            # bs, q = coord.shape[:2] # bs=16, q=2304
            # inp_q = torch.cat([query, query_ref_res], dim=-1)  # [16, 2304, 1, 640]
            # inp_q = inp_q.contiguous().view(bs * q, -1) # [16*2304, 640]
            # weight_q = self.imnet_q(inp_q).view(bs, q, -1).contiguous()  # [16, 2304, 576]
            # query = query.contiguous().view(bs, q, -1) # [16, 2304, 1, 576]->[16, 2304, 576]
            # pred_q = (query * weight_q).view(bs, q, 1, -1)  # [16, 2304, 1, 576]

            #TODO 方案三：query_ref_res也许用于处理KV的效果更好？


            #feat_coord = make_coord(feature.shape[-2:], flatten=False).permute(2, 0, 1) \
            #    .unsqueeze(0).expand(B, 2, *feature.shape[-2:])  # [16, 2, 48, 48]
            #feat_coord = feat_coord.to(coord)
            feat_coord = self.feat_coord

            # 根据局部区域的尺度计算偏移量（同LIIF）
            if self.local_size == 1:
                v_lst = [(0, 0)]
            else:
                v_lst = [(i, j) for i in range(-1, 2, 4 - self.local_size) for j in range(-1, 2, 4 - self.local_size)]
            eps_shift = 1e-6
            preds_k, preds_v = [], []

            for v in v_lst:
                vx, vy = v[0], v[1]
                # project to LR field
                tx = ((H - 1) / (1 - scale[:, 0, 0])).view(B, 1)  # [16, 1]
                ty = ((W - 1) / (1 - scale[:, 0, 1])).view(B, 1)  # [16, 1]
                rx = (2 * abs(vx) - 1) / tx if vx != 0 else 0  # [16, 1]
                ry = (2 * abs(vy) - 1) / ty if vy != 0 else 0  # [16, 1]

                bs, q = coord.shape[:2]
                coord_ = coord.clone()  # [16, 2304, 2]
                if vx != 0:
                    coord_[:, :, 0] += vx / abs(vx) * rx + eps_shift  # [16, 2304]
                if vy != 0:
                    coord_[:, :, 1] += vy / abs(vy) * ry + eps_shift  # [16, 2304]
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # K and V
                key = F.grid_sample(feat_k, coord_.flip(-1).unsqueeze(1), mode='nearest',
                                    align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()  # [16, 2304, 576]
                value = F.grid_sample(feat_v, coord_.flip(-1).unsqueeze(1), mode='nearest',
                                      align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()  # [16, 2304, 576]

                #print("################################################", feat_coord.shape) # [12, 2, 48, 48]
                #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", coord_.shape) # [16, 2304, 2]
                #BUG Interpolate K to HR resolution（上采样LR坐标网格到HR尺度） 批量不能被整除导致的
                coord_k = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1),
                                        mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2,
                                                                                                 1)  # [16, 2304, 2]

                Q, K = coord, coord_k  # [16, 2304, 2]
                # 相对坐标（缩放到特征图尺寸一致）
                rel = Q - K  # [16, 2304, 2] 
                rel[:, :, 0] *= feature.shape[-2]  # without mul
                rel[:, :, 1] *= feature.shape[-1]
                inp = rel  # [16, 2304, 2]
                
                # 上采样尺度
                scale_ = scale.clone()  # [16, 2304, 2] 
                scale_[:, :, 0] *= feature.shape[-2]
                scale_[:, :, 1] *= feature.shape[-1]

                #TODO 相对坐标用于计算K,V
                inp_v = torch.cat([value, inp, scale_], dim=-1)  # [16, 2304, 580]
                inp_k = torch.cat([key, inp, scale_], dim=-1)  # [16, 2304, 580]

                inp_k = inp_k.contiguous().view(bs * q, -1) # [36864, 580]
                inp_v = inp_v.contiguous().view(bs * q, -1) # [36864, 580]

                # 利用meta-learning的思想（project）
                weight_k = self.imnet_k(inp_k).view(bs, q, -1).contiguous()  # [16, 2304, 576]
                pred_k = (key * weight_k).view(bs, q, -1)  # [16, 2304, 576]

                weight_v = self.imnet_v(inp_v).view(bs, q, -1).contiguous()  # [16, 2304, 576]
                pred_v = (value * weight_v).view(bs, q, -1)  # [16, 2304, 576]

                preds_v.append(pred_v)
                preds_k.append(pred_k)

            # batch-size=16, local_size=2（四个采样点）
            preds_k = torch.stack(preds_k, dim=-1)  # [16, 2304, 576, 4]
            preds_v = torch.stack(preds_v, dim=-2)  # [16, 2304, 4, 576]

            #TODO 利用注意力机制进行特征聚合
            attn = (query @ preds_k)  # [16, 2304, 1, 4]
            x = ((attn / self.softmax_scale).softmax(dim=-1) @ preds_v)  # [16, 2304, 1, 576]
            x = x.view(bs * q, -1)  # [16*2304, 576]

            res_features.append(x) # 对应前面的遍历

        result = torch.cat(res_features, dim=-1)  # [16, 2304, 576x2]

        #TODO 最后经过out_dim=3的imnet_c
        result = self.imnet_c(result)  # [16, 2304, 3]
        result = result.view(bs, q, -1) 
        #print("################################################", result.shape)

        # 残差连接
        result += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',
                                padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        return result

    # 批量预测
    def batched_predict(self, x, coord, cell, ref, ref_hr, eval_bsize):
        """Batched predict.

        Args:
            x (Tensor): Input tensor.
            coord (Tensor): coord tensor.
            cell (Tensor): cell tensor.

        Returns:
            pred (Tensor): output of model.
        """
        with torch.no_grad():
            if coord is None and cell is None:
                # Evaluate encoder efficiency
                feat = self.encoder(x)
                return None

            self.gen_feat(x, ref, ref_hr)
            n = coord.shape[1]
            left = 0
            preds = []
            while left < n:
                right = min(left + eval_bsize, n)
                pred = self.query_rgb(coord[:, left:right, :], cell[:, left:right, :])
                preds.append(pred)
                left = right
            pred = torch.cat(preds, dim=1)
        return pred

    def forward(self, x, coord, cell, ref, ref_hr, bsize=None):
        """Forward function.

        Args:
            x: input tensor.
            coord (Tensor): coordinates tensor.
            cell (Tensor): cell tensor.
            test_mode (bool): Whether in test mode or not. Default: False.

        Returns:
            pred (Tensor): output of model.
        """
        if bsize is not None:
            pred = self.batched_predict(x, coord, cell, ref, ref_hr, bsize)
        else:
            self.gen_feat(x, ref, ref_hr)
            pred = self.query_rgb(coord, cell)

        return pred