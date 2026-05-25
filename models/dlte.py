import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register


def make_cell_info(out_shape, in_shape, v_range=None, device=None):
    """ Make cell info of each latent code. A cell contains the nearest output pixels of an latent code.
            out_shape: [s*H,s*W]
            in_shape: [H,W]
            stack_unit: stack coordinate nums
            cell_decode: True or False
            v_range: coord range[-1, 1]
        return:
            cell_info: (top_coord, left_coord, bottom_coord, right_coord, h_cell, w_cell, h_num. w_num, h_max_num. w_max_num, startIndex, endIndex) * H * W
            coord_num_seqs: [h axis coordinate nums, w axis coordinate nums]
    """

    device = device if device is not None else torch.device('cpu')  # use gpu or cpu

    if v_range is None:
        v0, v1 = -1, 1
    else:
        v0, v1 = v_range
    [in_h, in_w] = in_shape
    [out_h, out_w] = out_shape
    in_radius_h, in_radius_w = (v1 - v0) / (2 * in_h), (v1 - v0) / (2 * in_w)
    out_radius_h, out_radius_w = (v1 - v0) / (2 * out_h), (v1 - v0) / (2 * out_w)
    h_ratio, w_ratio = out_h / in_h, out_w / in_w

    # input index sequences on each axis
    in_h_axis_seq, in_w_axis_seq = torch.arange(in_h).float(), torch.arange(in_w).float()

    # output index sequences on each axis
    start_index_seqs = [
        (torch.arange(in_h).float() * h_ratio - 0.5).ceil(),
        (torch.arange(in_w).float() * w_ratio - 0.5).ceil()
    ]
    end_index_seqs = [
        ((torch.arange(in_h).float() + 1) * h_ratio - 0.5).ceil(),
        ((torch.arange(in_w).float() + 1) * w_ratio - 0.5).ceil()
    ]
    # coordinate numbers of each cell, (h_num, w_num)
    coord_num_seqs = [
        end_index_seqs[0] - start_index_seqs[0],
        end_index_seqs[1] - start_index_seqs[1]
    ]

    # cell center coordinates
    in_h_coord_seq, in_w_coord_seq = in_radius_h + 2 * in_radius_h * in_h_axis_seq, in_radius_w + 2 * in_radius_w * in_w_axis_seq
    # output border coordinates relative to cell center, (top, left, bottom, right)
    start_coord_seqs = [
        in_h * (out_radius_h + 2 * out_radius_h * start_index_seqs[0] - in_h_coord_seq),
        in_w * (out_radius_w + 2 * out_radius_w * start_index_seqs[1] - in_w_coord_seq)
    ]
    end_coord_seqs = [
        in_h * (out_radius_h + 2 * out_radius_h * (end_index_seqs[0] - 1) - in_h_coord_seq),
        in_w * (out_radius_w + 2 * out_radius_w * (end_index_seqs[1] - 1) - in_w_coord_seq),
    ]

    start_borders = torch.stack(torch.meshgrid(*start_coord_seqs, indexing='ij'), dim=-1).to(device)
    end_borders = torch.stack(torch.meshgrid(*end_coord_seqs, indexing='ij'), dim=-1).to(device)
    #num_grids = torch.stack(torch.meshgrid(*coord_num_seqs, indexing='ij'), dim=-1).to(device)

    # area of coordinate, (h_area, w_area)
    cell_areas = torch.ones(in_h, in_w, 2).to(device)
    cell_areas[:, :, 0] = 2 * in_h / out_h
    cell_areas[:, :, 1] = 2 * in_w / out_w

    return cell_areas.permute(2, 0, 1), start_borders.permute(2, 0, 1), end_borders.permute(2, 0, 1), coord_num_seqs


################################ DLTE（DIIF） ##################################
@register('dlte')
class DLTE(nn.Module):
    def __init__(self,
                 encoder_spec,
                 imnet_spec,
                 single_slice=False,
                 coarse_cell_decode=True,
                 fine_cell_decode=False,
                 local_ensemble=False,
                 ensemble_num=1,
                 feat_unfold=False,
                 unfold_num=3,
                 cell_decode=True,
                 batch_unit=512 * 384,
                 fixed_area=False,
                 hidden_dim=256 #TODO Conv的输出维度相关，imnet的输入维度相关
                 ):
        super(DLTE, self).__init__()

        # slicing mode
        self.single_slice = single_slice
        # cell docoding
        self.fine_cell_decode = cell_decode and fine_cell_decode
        self.coarse_cell_decode = cell_decode and coarse_cell_decode
        self.cell_decode = cell_decode #TODO
        # slice ensemble
        self.local_ensemble = local_ensemble
        self.ensemble_num = ensemble_num
        self.ensemble_num2 = ensemble_num * ensemble_num
        # feature unfolding
        self.feat_unfold = feat_unfold
        self.unfold_num = unfold_num
        self.unfold_num2 = unfold_num * unfold_num

        # testing config
        self.batch_unit = batch_unit   # 256 * 256 | 512 * 384
        self.fixed_area = fixed_area
        self.max_scale = 4
        self.fixed_border = False
        # self.device = torch.device(('cuda:' + str(gpu_ids[0])) if gpu_ids is not None else 'cpu')  # use gpu or cpu

        
        #TODO 创建编码器
        self.encoder = models.make(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim // (ensemble_num * ensemble_num), 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim // (ensemble_num * ensemble_num), 3, padding=1)

        # conv_in_dim = self.encoder.out_dim # 64
        # if self.local_ensemble:
        #     conv_in_dim *= self.ensemble_num2
        # elif self.feat_unfold:
        #     conv_in_dim *= self.unfold_num2
        
        # #TODO 有关输入特征维度的计算LIIF与LTE是统一的
        # # attach border coordinates
        # conv_in_dim += 4
        # # attach cell areas
        # if self.coarse_cell_decode:
        #     conv_in_dim += 2

        #TODO 定义imnet_spec的参数字典
        args = {}
        if self.fine_cell_decode:
            args['c_dim'] = 4
        
        #TODO 目前尚无法确定解码器的输入维度，要看self.imnet()的输入
        args['in_dim'] = hidden_dim 
        args['local_ensemble'] = self.local_ensemble
        args['unfold_num'] = self.unfold_num if self.local_ensemble else 0
        args['ensemble_num'] = self.ensemble_num if self.local_ensemble else 0

        #TODO 创建解码器
        self.imnet = models.make(imnet_spec, args)

    def unfold_feat(self, feat, coef, freq, w_seq=None):
        """
        Unfold feature maps

        :param feat:
        :param coef:
        :param freq:
        :param w_seq:
        :return:
        """

        #[in_b, in_c, in_h, in_w] = feat.shape # [2, 64, 48, 48]
        [in_b, in_c, in_h, in_w] = coef.shape # [2, 256, 48, 48]

        if w_seq is None:
            w_start, w_end = 0, in_w
        else:
            padding = self.unfold_num // 2
            w_start = w_seq[0] - padding if w_seq[0] - padding >= 0 else 0
            w_end = w_seq[1] + padding if w_seq[1] + padding <= in_w else in_w
            in_w = w_end - w_start
        
        #TODO 特征展开是在feat通过coef/freq后采用
        #print('#####################################################', coef.shape) # [2, 256, 48, 48]
        #print('#####################################################', freq.shape) # [2, 256, 48, 48]
        if self.feat_unfold:
            coef = F.unfold(coef[:, :, :, w_start: w_end], self.unfold_num, padding=self.unfold_num // 2).view(
                in_b, in_c, self.unfold_num, self.unfold_num, in_h, in_w)
            freq = F.unfold(freq[:, :, :, w_start: w_end], self.unfold_num, padding=self.unfold_num // 2).view(
                in_b, in_c, self.unfold_num, self.unfold_num, in_h, in_w)
            

        if w_seq is None:
            return feat, coef, freq
        else:
            start = padding if w_seq[0] - padding > 0 else 0
            end = -padding if w_seq[1] + padding <= feat.shape[-1] else feat.shape[-1]
            return feat[:, :, :, w_seq[0]:w_seq[1]], coef[:, :, :, :, :, start:end], freq[:, :, :, :, :, start:end]

    def query_rgb(self,
                  feat,
                  out_size,
                  cell_areas=None,
                  start_borders=None,
                  end_borders=None,
                  coord_num_seqs=None
                  ):
        """
        Upscale with mlp network

        feat: input feature map, (B, C, H, W)
        out_size: output size, (B, 3, sH, sW)
        cell_info: cell info, (stack_unit*2 (+2), H, W)
        coord_num_seqs: coordinate nums of h and w axis, list
        """

        # sample upscale without mlp network
        if self.imnet is None:
            # coord: (B, sH * sW, 2)
            coord = None
            out = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
            return out

        coef = self.coeff
        freq = self.freqq

        # get input and output size
        [in_b, in_c, in_h, in_w] = feat.shape
        [out_c, out_h, out_w]  = out_size[-3:]

        [h_num_seq, w_num_seq] = coord_num_seqs
        h_max_num, w_max_num = int(h_num_seq.max()), int(w_num_seq.max())
        h_min_num, w_min_num = int(h_num_seq.min()), int(w_num_seq.min())

        # determine coordinate slice direction, row or column
        slice_by_w = True
        if slice_by_w:
            main_max_num, cross_max_num = h_max_num, w_max_num
            main_dim_index, cross_dim_index = 0, 1
        else:
            main_max_num, cross_max_num = w_max_num, h_max_num
            main_dim_index, cross_dim_index = 1, 0

        # coordinate shift relative to slice start border (top, left)
        coord_shifts = torch.ones(2, in_h, in_w).to(feat.device)

        # cell info, shape: ((area) + start coord + end coord, H, W)
        if self.coarse_cell_decode:
            cell_info = torch.ones(6, in_h, in_w).to(feat.device)
            cell_info[:2, :, :] = cell_areas.clone()

            # TO DO: supporting training with x4 only model
            if self.fixed_area and out_h > in_h * self.max_scale:
                cell_info[:1, :, :] = 2 / self.max_scale
            if self.fixed_area and out_w > in_w * self.max_scale:
                cell_info[1:2, :, :] = 2 / self.max_scale
        else:
            cell_info = torch.ones(4, in_h, in_w).to(feat.device)

        # determine slice mode
        if self.single_slice:
            main_borders = [[0, main_max_num]]
            cross_borders = [[0, cross_max_num]]
        else:
            main_borders = []
            for main_index in range(main_max_num):
                main_borders.append([main_index, main_index + 1])
            cross_borders = [[0, cross_max_num]]

        # merge slices based on the input size
        #batched_interval = 1 if self.single_slice or self.training else int(256 * 256 / (in_w * in_h)) # int(self.batch_unit / (in_w * in_h))
        batched_interval = 1
        batched_index = 0
        batched_coef, batched_freq, batched_slice_coords, batched_slice_info = [], [], [], []

        # get pixels by coordinate slice order
        preds = []
        for main_border in main_borders:
            for cross_border in cross_borders:

                if batched_interval > 1:
                    batched_index += 1

                if main_border[0] == main_border[1] or cross_border[0] == cross_border[1]:
                    continue

                slice_info = cell_info.clone()
                # slice start coordinate (top, left)
                coord_shifts[main_dim_index, :, :] = main_border[0]
                coord_shifts[cross_dim_index, :, :] = cross_border[0]
                slice_info[-4:-2, :, :] = start_borders + coord_shifts * cell_areas
                # slice end coordinate (bottom, right)
                coord_shifts[main_dim_index, :, :] = main_border[1] - 1
                coord_shifts[cross_dim_index, :, :] = cross_border[1] - 1
                slice_info[-2:, :, :] = start_borders + coord_shifts * cell_areas

                # fixed borders with scale 4
                if self.fixed_border:
                    slice_info[-4 + cross_dim_index, :, :] = -0.75
                    slice_info[-2 + cross_dim_index, :, :] = 0.75
                    if self.single_slice:
                        slice_info[-4 + main_dim_index, :, :] = -0.75
                        slice_info[-2 + main_dim_index, :, :] = 0.75

                # get slice coordinates
                slice_coords = []
                for main_index in range(main_border[0], main_border[1]):
                    coord_shifts[main_dim_index, :, :] = main_index
                    for cross_index in range(cross_border[0], cross_border[1]):
                        coord_shifts[cross_dim_index, :, :] = cross_index
                        slice_coords.append(start_borders + coord_shifts * cell_areas)
                        if self.fine_cell_decode:
                            this_cell_area = cell_areas.clone()

                            # TO DO: supporting training with x4 only model
                            if self.fixed_area and out_h > in_h * self.max_scale:
                                this_cell_area[:1, :, :] = 2 / self.max_scale
                            if self.fixed_area and out_w > in_w * self.max_scale:
                                this_cell_area[1:2, :, :] = 2 / self.max_scale
                            slice_coords.append(this_cell_area)
                slice_coords = torch.cat(slice_coords, dim=0)

                # repeat input by batch size
                slice_info = slice_info.unsqueeze(0).repeat(in_b, 1, 1, 1)
                slice_coords = slice_coords.unsqueeze(0).repeat(in_b, 1, 1, 1)
                slice_coords = slice_coords.permute(0, 2, 3, 1).contiguous().view(in_b * in_h * in_w, -1)

                #TODO local_encemble也是对coef/freq起作用
                if self.local_ensemble:
                    # reshape mlp input to (B * H * W, C')
                    #inp = feat.permute(0, 2, 3, 1).contiguous().view(in_b * in_h * in_w, -1)
                    slice_info = slice_info.permute(0, 2, 3, 1).contiguous().view(in_b * in_h * in_w, -1)

                    if batched_interval <= 1:
                        # use mlp to predict pixels, shape: (B * H * W, S * 3)
                        pred = self.imnet(coef, freq, slice_coords, slice_info)
                        preds.append(pred)
                    else:
                        batched_coef.append(coef)
                        batched_freq.append(freq)
                        batched_slice_coords.append(slice_coords)
                        batched_slice_info.append(slice_info)

                        if batched_index == batched_interval:
                            # use mlp to predict pixels, shape: (B * H * W, bi * S * 3)
                            pred = self.imnet(torch.cat(batched_coef, dim=0),
                                              torch.cat(batched_freq, dim=0),
                                              torch.cat(batched_slice_coords, dim=0),
                                              torch.cat(batched_slice_info, dim=0))
                            for pred_index in range(batched_interval):
                                preds.append(pred[pred_index * in_b * in_h * in_w : (pred_index + 1) * in_b * in_h * in_w])
                            batched_index = 0
                            batched_coef, batched_freq, batched_slice_coords, batched_slice_info = [], [], [], []
                else:
                    # concat feature map and slice info as mlp input, shape: (B, C', H, W)
                    #inp = torch.cat([feat, slice_info], dim=1)
                    # reshape mlp input to (B * H * W, C')
                    #inp = inp.permute(0, 2, 3, 1).contiguous().view(in_b * in_h * in_w, -1)

                    # use mlp to predict pixels, shape: (B * H * W, S * 3)
                    pred = self.imnet(coef, freq, slice_coords)
                    preds.append(pred)

        if len(batched_coef) > 0:
            pred = self.imnet(torch.cat(batched_coef, dim=0),
                              torch.cat(batched_freq, dim=0),
                              torch.cat(batched_slice_coords, dim=0),
                              torch.cat(batched_slice_info, dim=0))
            for pred_index in range(len(batched_coef)):
                preds.append(pred[pred_index * in_b * in_h * in_w : (pred_index + 1) * in_b * in_h * in_w])

        # reshape predicted pixels to (B. H, W, main axis coord num, cross axis coord num, channel)
        pred_rgbs = torch.cat(preds, dim=1).view(in_b, in_h, in_w, main_max_num, cross_max_num, out_c)

        # reshape predicted pixels to (B. C, H, h_num, W, w_num)
        if slice_by_w:
            pred_rgbs = pred_rgbs.permute(0, -1, 1, 3, 2, 4).contiguous()
        else:
            pred_rgbs = pred_rgbs.permute(0, -1, 1, 4, 2, 3).contiguous()
        # reshape predicted pixels to (B. C, H * h_num, W * w_num)
        pred_rgbs = pred_rgbs.view(in_b, out_c, in_h * h_max_num, in_w * w_max_num)

        if h_max_num > h_min_num or w_max_num > w_min_num:
            # remove unnecessary pixels in h axis
            pred_img_h = []
            for in_h_index in range(in_h):
                h_num = int(h_num_seq[in_h_index])
                h_start_index = in_h_index * h_max_num
                pred_img_h.append(pred_rgbs[:, :, h_start_index: h_start_index + h_num, :])
            pred_img_h = torch.cat(pred_img_h, dim=-2)

            # remove unnecessary pixels in w axis
            pred_img_w = []
            for in_w_index in range(in_w):
                w_num = int(w_num_seq[in_w_index])
                w_start_index = in_w_index * w_max_num
                pred_img_w.append(pred_img_h[:, :, :, w_start_index: w_start_index + w_num])
            pred_img = torch.cat(pred_img_w, dim=-1)
        else:
            pred_img = pred_rgbs

        return pred_img

    def forward(self, x, coord, cell):
        """ x: input image
            coord: output grid [b, h, w, 2]
            cell: 多余输入
        """
        #TODO 输出的h和w不用直接输入，而是从coord中获取
        h = coord.shape[1]
        w = coord.shape[2] 

        if h == 0 or w == 0:
            out_shape = [x.shape[0], x.shape[1], x.shape[-2], x.shape[-1]]  # B, C, H, W
        else:
            out_shape = [x.shape[0], x.shape[1], h, w]  # B, C, H, W

        #TODO 将编码器得到的feat直接输入卷积
        feat = self.encoder(x)
        coef = self.coef(feat)
        freq = self.freq(feat)
        #print('#####################################################', coef.shape) # [2, 256, 48, 48]
        #print('#####################################################', freq.shape) # [2, 256, 48, 48]

        cell_areas, start_borders, end_borders, coord_num_seqs = make_cell_info(
            out_shape[-2:], feat.shape[-2:], device=feat.device)

        if self.training:
            feat, coef, freq = self.unfold_feat(feat, coef, freq)
            self.coeff = coef
            self.freqq = freq

            out = self.query_rgb(feat, out_shape, cell_areas=cell_areas, start_borders=start_borders,
                                 end_borders=end_borders, coord_num_seqs=coord_num_seqs)
        else:
            # query rgb values by batches in w axis
            batch_num = math.ceil(feat.shape[-2] * feat.shape[-1] / self.batch_unit)
            out_batches = []
            for batch_index in range(batch_num):
                w_start = int(feat.shape[-1] * batch_index / batch_num)
                w_end = feat.shape[-1] if batch_index == batch_num - 1 else int(feat.shape[-1] * (batch_index + 1) / batch_num)
                out_w = sum(coord_num_seqs[1][w_start: w_end])

                feat_batch, coef_batch, freq_batch = self.unfold_feat(feat, coef, freq, [w_start, w_end])
                self.coeff = coef_batch
                self.freqq = freq_batch
                out_batch = self.query_rgb(feat_batch, [out_shape[0], out_shape[1], out_shape[-2], out_w],
                                           cell_areas=cell_areas[:, :, w_start: w_end],
                                           start_borders=start_borders[:, :, w_start: w_end],
                                           end_borders=end_borders[:, :, w_start: w_end],
                                           coord_num_seqs=[coord_num_seqs[0], coord_num_seqs[1][w_start: w_end]])
                out_batches.append(out_batch)
            out = torch.cat(out_batches, dim=-1)

        out += F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False, antialias=True)
        return out


