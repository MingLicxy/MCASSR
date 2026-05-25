import math
import torch
import numpy as np
import torch.nn as nn

from models import register
####################
# MLP
####################

# MLP network
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super(MLP, self).__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

################################ C2FMLP_LIIF（DIIF） ##################################
# Corse-to-fine MLP network
@register('c2fmlp_liif')
class C2FMLP_LIIF(nn.Module):
    def __init__(self,
                 in_dim, # 由dliif传入
                 out_dim, # 3
                 c_dim, # 由dliif传入 4
                 head_hidden_list,
                 tail_hidden_list,
                 unfold_num=3, ensemble_num=0, local_ensemble=False # 由diff传入
                 ):
        super(C2FMLP_LIIF, self).__init__()
        #self.useReshape = out_dim < in_dim
        self.c_dim = c_dim if c_dim is not None else 2
        self.unfold_num = unfold_num if unfold_num is not None else 3
        self.ensemble_num = ensemble_num if ensemble_num is not None else int(1 + self.unfold_num // 2)
        self.local_ensemble = local_ensemble if local_ensemble is not None else False

        #self.v_unit = 256 * 256
        self.coord_unit = 4 * 3  # 4 * 3

        layers = []
        lastv = in_dim
        for hidden in head_hidden_list: # [256, 256]
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        self.layers = nn.Sequential(*layers)

        tails = []
        lastv = lastv + self.c_dim
        for hidden in tail_hidden_list: # [256, 256]
            tails.append(nn.Linear(lastv, hidden))
            tails.append(nn.ReLU())
            lastv = hidden
        tails.append(nn.Linear(lastv, out_dim))
        self.tail = nn.Sequential(*tails)

    # 第一阶段
    def get_coarse_inp(self, x, slice_coord, vh, vw, v_num, feat_dim):
        this_slice_coord = slice_coord.clone()
        # start border
        this_slice_coord[:, -4] -= vh
        this_slice_coord[:, -3] -= vw
        # end border
        this_slice_coord[:, -2] -= vh
        this_slice_coord[:, -1] -= vw

        h_start = self.unfold_num // 2 + int(vh * 0.5 + 0.5) - self.ensemble_num // 2
        w_start = self.unfold_num // 2 + int(vw * 0.5 + 0.5) - self.ensemble_num // 2

        this_feat = x[:, :, h_start:h_start + self.ensemble_num, w_start:w_start + self.ensemble_num]
        return torch.cat([this_feat.contiguous().view(
            v_num, feat_dim * self.ensemble_num * self.ensemble_num), this_slice_coord], dim=1)
    
    # 第二阶段
    def get_fine_inp(self, vs, coord, coord_index):
        # c_dim: coordinate (2 dim) + area (2 dim)
        this_coord = coord[:, coord_index * self.c_dim:(coord_index + 1) * self.c_dim]

        # get average rgb hidden vector using area sizes
        areas = []
        total_area = 0
        for vh in [-1, 1]:  # top, bottom
            for vw in [-1, 1]:  # left, right
                area = torch.abs((this_coord[:, 0] - vh) * (this_coord[:, 1] - vw))
                areas.append(area + 1e-9)
                total_area += area + 1e-9
        t = areas[0]
        areas[0] = areas[3]
        areas[3] = t
        t = areas[1]
        areas[1] = areas[2]
        areas[2] = t

        v_avg = 0
        for this_v, this_area in zip(vs, areas):
            v_avg = v_avg + this_v * (this_area / total_area).unsqueeze(-1)
        return torch.cat([v_avg, this_coord], dim=-1)

    def query(self, x, coord, slice_coord=None):
        v_num = x.shape[0]
        feat_dim = int(x.shape[-1] // (self.unfold_num * self.unfold_num))
        # shape: (b * h * w, c, h_u, w_u)
        x = x.view(v_num, feat_dim, self.unfold_num, self.unfold_num)

        vs = []
        # get four rgb hidden vectors
        for vh in [-1, 1]:  # top, bottom
            for vw in [-1, 1]:  # left, right
                # get rgb hidden vector
                this_v = self.layers(self.get_coarse_inp(x, slice_coord, vh, vw, v_num, feat_dim))
                vs.append(this_v)

        coord_num = coord.shape[-1] // self.c_dim
        coord_unit_num = int(math.ceil(coord_num/self.coord_unit))
        outs = []
        for coord_unit_index in range(coord_unit_num):
            coord_start = coord_unit_index * self.coord_unit
            coord_end = min((coord_unit_index + 1) * self.coord_unit, coord_num)
            inps = []
            for coord_index in range(coord_start, coord_end):
                inps.append(self.get_fine_inp(vs, coord, coord_index))
            out = self.tail(torch.cat(inps, dim=0))
            outs.append(out)
        outs = torch.cat(outs, dim=0).view(coord_num, v_num, -1).permute(1, 0, 2).contiguous().view(v_num, -1)
        return outs

    #BUG 这里的输入都对应什么
    def forward(self, x, coord, slice_coord=None):
        """
        :param x: shape (v_num, feat_dim)
        :param coord: shape (v_num, coord_num * coord_dim)
        :param slice_coord: shape (v_num, coord_dim * 2 + area_dim)

        :param quartered_center: [-1,-1], [1,-1], [-1,1], [1,1]
        :return: shape (v_num, coord_num * out_dim)
        """

        if self.local_ensemble:
            if self.training:
                v_num = x.shape[0]
                feat_dim = int(x.shape[-1] // (self.unfold_num * self.unfold_num))
                # shape: (b * h * w, c, h_u, w_u)
                x = x.view(v_num, feat_dim, self.unfold_num, self.unfold_num)

                vs = []
                for vh in [-1, 1]:  # top, bottom
                    for vw in [-1, 1]:  # left, right
                        # get rgb hidden vector
                        this_v = self.layers(self.get_coarse_inp(x, slice_coord, vh, vw, v_num, feat_dim))
                        vs.append(this_v)

                coord_num = coord.shape[-1] // self.c_dim
                out = []
                for coord_index in range(coord_num):
                    p = self.tail(self.get_fine_inp(vs, coord, coord_index))
                    out.append(p)
                return torch.cat(out, dim=-1)
            else:
                '''
                v_num = x.shape[0]
                v_unit_num = int(math.ceil(v_num / self.v_unit))
                outs = []
                for v_unit_index in range(v_unit_num):
                    v_start = v_unit_index * self.v_unit
                    v_end = min((v_unit_index + 1) * self.v_unit, v_num)
                    out = self.query(x[v_start:v_end, :], coord[v_start:v_end, :], slice_coord[v_start:v_end, :])
                    outs.append(out)
                return torch.cat(outs, dim=0)
                '''

                return self.query(x, coord, slice_coord)
        else:
            v = self.layers(x)
            # 2 dimension coordinate + (2 dimension area)
            coord_num = coord.shape[-1] // self.c_dim
            out = []
            for coord_index in range(coord_num):
                this_coord = coord[:, coord_index * self.c_dim:(coord_index + 1) * self.c_dim]
                p = self.tail(torch.cat([v, this_coord], dim=-1))
                out.append(p)
            return torch.cat(out, dim=-1)



################################ C2FMLP_LTE（DIIF） ##################################
@register('c2fmlp_lte')
class C2FMLP_LTE(nn.Module):
    """ Corse-to-fine MLP network for LTE """
    def __init__(self,
                 in_dim,
                 out_dim,
                 c_dim,
                 head_hidden_list,
                 tail_hidden_list,
                 unfold_num=3,
                 ensemble_num=0,
                 local_ensemble=False
                 ):
        super(C2FMLP_LTE, self).__init__()
        #self.useReshape = out_dim < in_dim
        self.c_dim = c_dim if c_dim is not None else 2
        self.unfold_num = unfold_num if unfold_num is not None else 3
        self.ensemble_num = ensemble_num if ensemble_num is not None else int(1 + self.unfold_num // 2)
        self.local_ensemble = local_ensemble if local_ensemble is not None else False

        #self.v_unit = 256 * 256
        self.coord_unit = 16 # 4 * 3

        self.phase = nn.Linear(2, in_dim // 2, bias=False)

        layers = []
        lastv = in_dim
        for hidden in head_hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        self.layers = nn.Sequential(*layers)

        tails = []
        lastv = lastv + self.c_dim
        for hidden in tail_hidden_list:
            tails.append(nn.Linear(lastv, hidden))
            tails.append(nn.ReLU())
            lastv = hidden
        tails.append(nn.Linear(lastv, out_dim))
        self.tail = nn.Sequential(*tails)

    def get_coarse_inp(self, coef, freq, slice_coord=None, vh=0, vw=0):
        this_slice_coord = slice_coord.clone()
        # start border
        this_slice_coord[:, -4] -= vh
        this_slice_coord[:, -3] -= vw
        # end border
        this_slice_coord[:, -2] -= vh
        this_slice_coord[:, -1] -= vw

        h_start = self.unfold_num // 2 + int(vh * 0.5 + 0.5) - self.ensemble_num // 2
        w_start = self.unfold_num // 2 + int(vw * 0.5 + 0.5) - self.ensemble_num // 2

        freq_ = freq[:, :, h_start:h_start + self.ensemble_num, w_start:w_start + self.ensemble_num, :, :]
        freq_ = freq_.reshape(freq_.shape[0], freq_.shape[1] * self.ensemble_num * self.ensemble_num, freq_.shape[-2], freq_.shape[-1]).permute(
            0, 2, 3, 1).contiguous()
        freq_ = freq_.view(freq_.shape[0], freq_.shape[1] * freq_.shape[2], -1)

        coef_ = coef[:, :, h_start:h_start + self.ensemble_num, w_start:w_start + self.ensemble_num, :, :]
        coef_ = coef_.reshape(coef_.shape[0], coef_.shape[1] * self.ensemble_num * self.ensemble_num, coef_.shape[-2], coef_.shape[-1]).permute(
            0, 2, 3, 1).contiguous()
        coef_ = coef_.view(coef_.shape[0], coef_.shape[1] * coef_.shape[2], -1)

        bs, q = coef_.shape[0], coef_.shape[1]
        this_slice_coord = this_slice_coord.view(bs, q, this_slice_coord.shape[-1])

        freq_ = torch.stack(torch.split(freq_, 2, dim=-1), dim=-1)
        freq_start = torch.mul(freq_, this_slice_coord[:, :, :2].unsqueeze(-1))
        freq_end = torch.mul(freq_, this_slice_coord[:, :, 2:4].unsqueeze(-1))
        freq_ = torch.sum(torch.cat([freq_start, freq_end], dim=-2), dim=-2)
        freq_ += self.phase(this_slice_coord[:, :, 4:]).view(bs, q, -1)
        freq_ = torch.cat((torch.cos(np.pi * freq_), torch.sin(np.pi * freq_)), dim=-1)

        return torch.mul(coef_, freq_).contiguous().view(bs * q, -1)

    def get_fine_inp(self, vs, coord, coord_index):
        # c_dim: coordinate (2 dim) + area (2 dim)
        this_coord = coord[:, coord_index * self.c_dim:(coord_index + 1) * self.c_dim]

        # get average rgb hidden vector using area sizes
        areas = []
        total_area = 0
        for vh in [-1, 1]:  # top, bottom
            for vw in [-1, 1]:  # left, right
                area = torch.abs((this_coord[:, 0] - vh) * (this_coord[:, 1] - vw))
                areas.append(area + 1e-9)
                total_area += area + 1e-9
        t = areas[0]
        areas[0] = areas[3]
        areas[3] = t
        t = areas[1]
        areas[1] = areas[2]
        areas[2] = t

        v_avg = 0
        for this_v, this_area in zip(vs, areas):
            v_avg = v_avg + this_v * (this_area / total_area).unsqueeze(-1)
        return torch.cat([v_avg, this_coord], dim=-1)

    def query(self, coef, freq, coord, slice_coord=None):
        vs = []
        # get four rgb hidden vectors
        for vh in [-1, 1]:  # top, bottom
            for vw in [-1, 1]:  # left, right
                # get rgb hidden vector
                this_v = self.layers(self.get_coarse_inp(coef, freq, slice_coord, vh, vw))
                vs.append(this_v)

        coord_num = coord.shape[-1] // self.c_dim
        coord_unit_num = int(math.ceil(coord_num/self.coord_unit))
        outs = []
        for coord_unit_index in range(coord_unit_num):
            coord_start = coord_unit_index * self.coord_unit
            coord_end = min((coord_unit_index + 1) * self.coord_unit, coord_num)
            inps = []
            for coord_index in range(coord_start, coord_end):
                inps.append(self.get_fine_inp(vs, coord, coord_index))
            out = self.tail(torch.cat(inps, dim=0))
            outs.append(out)

        v_num = coef.shape[0] * coef.shape[-2] * coef.shape[-1]
        outs = torch.cat(outs, dim=0).view(coord_num, v_num, -1).permute(1, 0, 2).contiguous().view(v_num, -1)
        return outs

    def forward(self, coef, freq, coord, slice_coord=None):
        """
        :param coef: shape (b, c, u, u, h, w)
        :param freq: shape (b, c, u, u, h, w)
        :param coord: shape (v_num, coord_num * coord_dim)
        :param slice_coord: shape (v_num, coord_dim * 2 + area_dim)
        :return: shape (v_num, coord_num * out_dim)
        """

        if self.local_ensemble:
            if self.training:
                vs = []
                for vh in [-1, 1]:  # top, bottom
                    for vw in [-1, 1]:  # left, right
                        # get rgb hidden vector
                        this_v = self.layers(self.get_coarse_inp(coef, freq, slice_coord, vh, vw))
                        vs.append(this_v)

                coord_num = coord.shape[-1] // self.c_dim
                out = []
                for coord_index in range(coord_num):
                    p = self.tail(self.get_fine_inp(vs, coord, coord_index))
                    out.append(p)
                return torch.cat(out, dim=-1)
            else:
                return self.query(coef, freq, coord, slice_coord)
        else:
            #BUG
            x = None
            v = self.layers(x)

            # 2 dimension coordinate + (2 dimension area)
            coord_num = coord.shape[-1] // self.c_dim
            out = []
            for coord_index in range(coord_num):
                this_coord = coord[:, coord_index * self.c_dim:(coord_index + 1) * self.c_dim]
                p = self.tail(torch.cat([v, this_coord], dim=-1))
                out.append(p)
            return torch.cat(out, dim=-1)