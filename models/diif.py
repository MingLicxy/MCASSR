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
            v_range: coordinate range [-1, 1]
            device: device info
        return:
            cell_areas: (cell h, cell w) * H * W
            start_borders: (cell top, cell left) * H * W
            end_borders: (cell bottom, cell right) * H * W
            coord_num_seqs: [h axis coordinate nums, w axis coordinate nums]
    """
    device = device if device is not None else torch.device('cpu')  # use gpu or cpu
    # 坐标范围
    if v_range is None:
        v0, v1 = -1, 1
    else:
        v0, v1 = v_range
    
    [in_h, in_w] = in_shape
    [out_h, out_w] = out_shape
    # 计算输入和输出的半径，作为计算坐标位置的基础
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

    # area of each coordinate, (h_area, w_area)
    cell_areas = torch.ones(in_h, in_w, 2).to(device)
    cell_areas[:, :, 0] = 2 * in_h / out_h
    cell_areas[:, :, 1] = 2 * in_w / out_w

    return cell_areas.permute(2, 0, 1), start_borders.permute(2, 0, 1), end_borders.permute(2, 0, 1), coord_num_seqs

################################ DIIF（DIIF） ##################################
@register('diif')
class DIIF(nn.Module):
    """ Diif framework """
    def __init__(self,
                 encoder_spec, # 编码器
                 imnet_spec=None, # 解码器
                 single_slice=False,
                 coarse_cell_decode=True,
                 fine_cell_decode=False,
                 local_ensemble=False,
                 ensemble_num=4,
                 feat_unfold=True,
                 unfold_num=3,
                 batch_unit=384 * 384, # 
                 fixed_area=True
                 ):
        super(DIIF, self).__init__()

        # slicing mode
        self.single_slice = single_slice
        # cell decoding
        self.fine_cell_decode = fine_cell_decode
        self.coarse_cell_decode = coarse_cell_decode
        # slice ensemble
        self.local_ensemble = local_ensemble
        self.ensemble_num = ensemble_num
        self.ensemble_num2 = ensemble_num * ensemble_num
        # feature unfolding
        self.feat_unfold = feat_unfold
        self.unfold_num = unfold_num # 3
        self.unfold_num2 = unfold_num * unfold_num # 9

        # testing config
        self.batch_unit = batch_unit
        self.fixed_area = fixed_area
        self.max_scale = 4
        # self.device = torch.device(('cuda:' + str(gpu_ids[0])) if gpu_ids is not None else 'cpu')  # use gpu or cpu
    

        #TODO 创建编码器
        self.encoder = models.make(encoder_spec)
        # 为SwinIR的窗口策略做准备
        self.window_size = 8 if encoder_spec["name"] == "swinir" else 0 
        
        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim # 64
            #TODO
            if self.local_ensemble:
                imnet_in_dim *= self.ensemble_num2
            elif self.feat_unfold:
                imnet_in_dim *= self.unfold_num2

            # attach border coordinates
            imnet_in_dim += 4
            # attach cell areas
            if self.coarse_cell_decode:
                imnet_in_dim += 2

            #TODO 定义imnet_spec的参数字典
            args = {}
            if self.fine_cell_decode:
                args['c_dim'] = 4

            #print('#####################################', imnet_in_dim) #1606

            #BUG 这种方式无法传入参数
            args['in_dim'] = imnet_in_dim
            args['local_ensemble'] = self.local_ensemble
            args['unfold_num'] = self.unfold_num if self.local_ensemble else 0
            args['ensemble_num'] = self.ensemble_num if self.local_ensemble else 0
            #print('#####################################', args)

            #TODO 创建解码器
            self.imnet = models.make(imnet_spec, args)
        else:
            self.imnet = None

    # 特征展开函数（指定范围）提取局部感受野的信息来丰富特征
    def unfold_feat(self, feat, w_seq=None):
        """
        Unfold feature maps

        feat: feature map
        w_seq: start index and end index in w axis
        注意: w_seq表示宽度w维度上的开始与结束索引,用于选择特定区域
        """

        if self.imnet is None:
            return feat

        if self.feat_unfold:
            # unfold nearby features to enrich info
            [in_b, in_c, in_h, in_w] = feat.shape # [b, c, h, w]
            if w_seq is None: 
                # 对整个特征图进行展开，unfold_num是展开核大小
                return F.unfold(feat, self.unfold_num, padding=(self.unfold_num // 2)).view(
                    in_b, in_c * self.unfold_num2, in_h, in_w)

            padding = self.unfold_num // 2
            w_start = w_seq[0] - padding if w_seq[0] - padding >= 0 else 0
            w_end = w_seq[1] + padding if w_seq[1] + padding <= in_w else in_w
            # 在w_seq指定的范围内进行特征展开
            out = F.unfold(feat[:, :, :, w_start: w_end], self.unfold_num, padding=padding).view(
                in_b, in_c * self.unfold_num2, in_h, w_end - w_start)
            return out[:, :, :, padding if w_seq[0] - padding > 0 else 0:
                                -padding if w_seq[1] + padding <= in_w else in_w]
        else:
            if w_seq is None:
                return feat
            return feat[:, :, :, w_seq[0]: w_seq[1]]

    def query_rgb(self,
                  feat,
                  out_size, # 在forword中传入
                  cell_areas=None, # 像素单元相关
                  start_borders=None,
                  end_borders=None,
                  coord_num_seqs=None
                  ):
        """
        Upscale with mlp network

        feat: feature map, (B, C, H, W)
        out_size: output size, (B, 3, sH, sW)
        cell_areas: cell area, (2, H, W)
        start_borders: cell top and left coord, (2, H, W)
        end_borders: cell bottom and right coord, (2, H, W)
        coord_num_seqs: coordinate nums of each cell in h and w axis, list
        """

        # sample upscale without mlp network
        if self.imnet is None:
            # coord: (B, sH * sW, 2)
            coord = None #TODO 并未传入coord
            out = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                                mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
            return out

        # get input and output size
        [in_b, in_c, in_h, in_w] = feat.shape
        [out_c, out_h, out_w]  = out_size[-3:]

        #TODO cell在高度h和宽度w方向上的坐标数
        [h_num_seq, w_num_seq] = coord_num_seqs
        h_max_num, w_max_num = int(h_num_seq.max()), int(w_num_seq.max())
        h_min_num, w_min_num = int(h_num_seq.min()), int(w_num_seq.min())

        # determine coordinate slice direction, row or column
        #TODO 决定坐标切片的方向
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
            cell_info[:2, :, :] = cell_areas

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

        # get pixels by coordinate slice order
        preds = []
        for main_border in main_borders:
            for cross_border in cross_borders:
                if main_border[0] == main_border[1] or cross_border[0] == cross_border[1]:
                    continue

                slice_info = cell_info.clone()

                # slice start coordinate (top, left) 切片开始坐标
                coord_shifts[main_dim_index, :, :] = main_border[0]
                coord_shifts[cross_dim_index, :, :] = cross_border[0]
                slice_info[-4:-2, :, :] = start_borders + coord_shifts * cell_areas

                # slice end coordinate (bottom, right) 切片结束坐标
                coord_shifts[main_dim_index, :, :] = main_border[1] - 1
                coord_shifts[cross_dim_index, :, :] = cross_border[1] - 1
                slice_info[-2:, :, :] = start_borders + coord_shifts * cell_areas

                # get slice coordinates 获取切片坐标
                slice_coords = []
                for main_index in range(main_border[0], main_border[1]):
                    coord_shifts[main_dim_index, :, :] = main_index
                    for cross_index in range(cross_border[0], cross_border[1]):
                        coord_shifts[cross_dim_index, :, :] = cross_index
                        slice_coords.append(start_borders + coord_shifts * cell_areas)
                        if self.fine_cell_decode:
                            slice_coords.append(cell_areas.clone())

                slice_coords = torch.cat(slice_coords, dim=0)

                # repeat by batch size
                slice_info = slice_info.unsqueeze(0).repeat(in_b, 1, 1, 1)
                slice_coords = slice_coords.unsqueeze(0).repeat(in_b, 1, 1, 1)
                slice_coords = slice_coords.permute(0, 2, 3, 1).contiguous().view(in_b * in_h * in_w, -1)

                #TODO 局部聚合
                if self.local_ensemble:
                    # reshape mlp input to (B * H * W, C')
                    inp = feat.permute(0, 2, 3, 1).contiguous().view(in_b * in_h * in_w, -1)
                    slice_info = slice_info.permute(0, 2, 3, 1).contiguous().view(in_b * in_h * in_w, -1)

                    # use mlp to predict pixels, shape: (B * H * W, S * 3)
                    #BUG c2fmlp()接收的参数数量没对应
                    pred = self.imnet(inp, slice_coords, slice_info, None)
                else:
                    # concat feature map and slice info as mlp input, shape: (B, C', H, W)
                    inp = torch.cat([feat, slice_info], dim=1)
                    # reshape mlp input to (B * H * W, C')
                    inp = inp.permute(0, 2, 3, 1).contiguous().view(in_b * in_h * in_w, -1)
                    # use mlp to predict pixels, shape: (B * H * W, S * 3)
                    pred = self.imnet(inp, slice_coords)
                preds.append(pred)

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


        #TODO 当且仅当encoder是SwinIR时，根据窗口尺寸调整输入输出
        if self.window_size != 0:
            h_scale, w_scale = h / x.shape[-2], w / x.shape[-1]
            h_pad = (x.shape[-2] // self.window_size + 1) * self.window_size - x.shape[-2]
            w_pad = (x.shape[-1] // self.window_size + 1) * self.window_size - x.shape[-1]
            x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :x.shape[-2] + h_pad, :]
            x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :x.shape[-1] + w_pad]

            old_h, old_w = h, w # 保存原有的输出尺度
            h, w = int(h_scale * x.shape[-2]), int(w_scale * x.shape[-1])

        if h == 0 or w == 0:
            out_shape = [x.shape[0], x.shape[1], x.shape[-2], x.shape[-1]]  # B, C, H, W
        else:
            out_shape = [x.shape[0], x.shape[1], h, w]  # B, C, H, W

        # 提取特征
        feat = self.encoder(x) 
        #TODO 利用make_cell_info()获取query_rgb()所需的输入
        cell_areas, start_borders, end_borders, coord_num_seqs = make_cell_info(out_shape[-2:], feat.shape[-2:],
                                                                                device=feat.device)
        
        #TODO 训练与测试的前向过程不同
        if self.training:
            feat = self.unfold_feat(feat, None)
            out = self.query_rgb(feat, out_shape, cell_areas=cell_areas, start_borders=start_borders,
                                 end_borders=end_borders, coord_num_seqs=coord_num_seqs)
        else:
            # query rgb values for batches in w axis
            batch_num = math.ceil(feat.shape[-2] * feat.shape[-1] / self.batch_unit)
            out_batches = []
            for batch_index in range(batch_num):
                w_start = int(feat.shape[-1] * batch_index / batch_num)
                w_end = feat.shape[-1] if batch_index == batch_num - 1 else int(feat.shape[-1] * (batch_index + 1) / batch_num)
                out_w = sum(coord_num_seqs[1][w_start: w_end])

                feat_batch = self.unfold_feat(feat, [w_start, w_end])
                out_batch = self.query_rgb(feat_batch, [out_shape[0], out_shape[1], out_shape[-2], out_w],
                                           cell_areas=cell_areas[:, :, w_start : w_end],
                                           start_borders=start_borders[:, :, w_start : w_end],
                                           end_borders=end_borders[:, :, w_start : w_end],
                                           coord_num_seqs=[coord_num_seqs[0], coord_num_seqs[1][w_start: w_end]])
                out_batches.append(out_batch)
            out = torch.cat(out_batches, dim=-1)

        # 裁剪填充
        if self.window_size != 0:
            out = out[..., :old_h, :old_w]

        return out

