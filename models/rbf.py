# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
注意：LIIF与基于INR的2D Image Fitting方法（eg: INGP）的本质区别在于显式特征网格的来源不同
LIIF的特征网格由LR通过特征提取encoder获取，而INGP的特征网格通过在拟合2D图像的训练过程中自行生成
LIIF最终解码器的输入是LR特征+查询坐标; 而INGP的输入仅仅是(x,y)/(x,y,z)坐标，学习到的显式特征表示不会明显表示出来
"""







import os
import sys
import time
import math
import numpy as np
#import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykdtree.kdtree import KDTree

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch_rbf.thirdparty.torch_ngp.encoding import get_encoder
from arch_rbf import util_misc
from arch_rbf import util_clustering
from arch_rbf import util_network


act_dict = {'none': 0, 'relu': 1, 'sine': 2, 'gauss': 3}

#TODO python实现torch.exp2()
def exp2(tensor):
    # 确保 tensor 在 CPU 上，并将其转换为 NumPy 数组
    numpy_array = tensor.cpu().numpy()
    # 使用 NumPy 的 exp2 函数计算结果
    result_numpy = np.exp2(numpy_array)
    # 将结果转换回 PyTorch 张量
    result_tensor = torch.from_numpy(result_numpy).to(tensor.device)
    return result_tensor

# TODO (x,y)->(R,G,B), 输入只有坐标
class IMGNetwork(nn.Module):
    def __init__(self,
                 cmin,
                 cmax,
                 encoding="hashgrid",
                 num_layers=3,
                 skips=[],
                 hidden_dim=64,
                 in_dim=3,   # 2
                 out_dim=3,
                 num_levels=16,
                 level_dim=2,
                 base_resolution=16,
                 log2_hashmap_size=24,
                 desired_resolution=2048,
                 act='relu',
                 lc_act='relu',
                 lc_init=1e-4,
                 lca_init=None,
                 w_init=None,
                 b_init=None,
                 a_init=None,
                 pe_freqs=[],
                 levels_omit=[],
                 ):
        super().__init__()
        if not isinstance(cmin, list):
            cmin = [cmin] * in_dim
        if not isinstance(cmax, list):
            cmax = [cmax] * in_dim
        self.register_buffer('cmin', torch.tensor(cmin))  # ... y x
        self.register_buffer('cmax', torch.tensor(cmax))  # ... y x

        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = act
        self.lc_act = lc_act
        self.lc_init = lc_init
        self.lca_init = lca_init
        self.w_init = w_init
        self.b_init = b_init
        self.a_init = a_init
        
        

        #TODO 获取编码器（编码坐标而不是）
        self.encoder, self.backbone_in_dim = get_encoder(encoding, input_dim=in_dim, num_levels=num_levels, 
            level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, 
            desired_resolution=desired_resolution, levels_omit=levels_omit)
        
        self.encoder.embeddings.data.uniform_(-self.lc_init, self.lc_init)
        self.out_lc0_dim = self.backbone_in_dim

        # PE
        self.pe = None
        if len(pe_freqs) > 0:
            P = torch.cat([torch.eye(2)*2**i for i in pe_freqs], 1)
            self.pe = util_network.PE(P)
            self.backbone_in_dim += self.pe.out_dim
        
        backbone = []
        self.w_dims = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.backbone_in_dim
            elif l in self.skips:
                in_dim = self.hidden_dim + self.backbone_in_dim
            else:
                in_dim = self.hidden_dim
            if l == num_layers - 1:
                out_dim = self.out_dim
            else:
                out_dim = self.hidden_dim
            backbone.append(nn.Linear(in_dim, out_dim, bias=True))
            self.w_dims.append([out_dim, in_dim])
        self.backbone = nn.ModuleList(backbone)

        self.init_w_b()

    def count_params(self):
        params = {}
        params['hg'] = util_misc.count_parameters(self.encoder)
        params['mlp'] = util_misc.count_parameters(self.backbone)
        params['total'] = sum([v for _, v in params.items()])
        for k, v in params.items():
            print(f'{k} params: {v}')
        return params

    def init_w_b(self):
        with torch.no_grad():
            for i in range(self.num_layers):
                wi = self.backbone[i].weight
                bi = self.backbone[i].bias
                out_dim, in_dim = self.w_dims[i]

                if len(self.w_init) > i and self.w_init[i] is not None:
                    val = self.w_init[i]
                    nn.init.uniform_(wi, -val, val)
                elif self.act == 'sine':
                    if i == 0:
                        nn.init.uniform_(wi, -1 / in_dim, 1 / in_dim)
                    else:
                        nn.init.uniform_(wi, -np.sqrt(6 / in_dim) / 30, np.sqrt(6 / in_dim) / 30)
                
                if len(self.b_init) > i and self.b_init[i] is not None:
                    val = self.b_init[i]
                    nn.init.uniform_(bi, -val, val)

    def clip_rbf_params(self):
        pass

    def forward(self, x, **kwargs):
        # x: [B, 3]
        out_other = {}

        #print("#####################################x shape:", x.shape) # [1, 3]
        #print("#############################self.cmax shape:", self.cmax.shape) # [2]

        #TODO 编码器输出只会影响特征维度
        h = self.encoder(x/self.cmax.flip(-1)[None])
        print("#####################################h shape:", h.shape) # [1, 48, 48, 32]

        out_other['out_lc0'] = h

        if self.lc_act == 'none':
            pass
        elif self.lc_act == 'relu':
            h = F.relu(h, inplace=True)
        elif self.lc_act == 'sine':
            h = util_network.scaledsin_activation(h, torch.tensor(self.lca_init).to(h.device))
        else:
            raise NotImplementedError

        if self.pe is not None:
            h = torch.cat((h, self.pe(x)), dim=-1)

        for l in range(self.num_layers):
            if l in self.skips:
                h = torch.cat([h, x], dim=-1)
            h = self.backbone[l](h)
            if l != self.num_layers - 1:
                if self.act is None:
                    pass
                elif self.act == 'relu':
                    h = F.relu(h, inplace=True)
                elif self.act == 'sine':
                    h = util_network.scaledsin_activation(h, torch.tensor(30).to(h.device))
                elif self.act == 'gauss':
                    h = util_network.gaussian_activation(h, self.a_init[l])
                else:
                    raise NotImplementedError

        return h, out_other


class rbf(nn.Module):
    def __init__(self,
                 cmin,
                 cmax,
                 s_dims,
                 in_dim=2, out_dim=3, 
                 num_layers=3, skips=[], hidden_dim=64, n_hidden_fl=20,
                 num_levels_ref=16, level_dim_ref=2, base_resolution_ref=16, log2_hashmap_size_ref=24,
                 num_levels=2, level_dim=2, base_resolution=16, log2_hashmap_size=24, desired_resolution=2048,
                 rbf_type='nlin_s', n_kernel=64, point_nn_kernel=4, ks_alpha=1, 
                 lc_init=[3e-1], lcb_init=None,
                 w_init=None, b_init=None, a_init=None,
                 sparse_embd_grad=True, act='relu', lc_act=None, 
                 rbf_suffixes=None, kc_init_config=None, 
                 rbf_lc0_normalize=True, 
                 pe_freqs=[], pe_lc0_freq=None, pe_hg0_freq=None, pe_lc0_rbf_freq=None, pe_lc0_rbf_keep=None, # 位置编码相关
                 use_train_knn=False, #TODO 该参数额外加入
                 **kwargs):
        
        super().__init__()

        # 确保cmin与cmax为长度in_dim的列表
        if not isinstance(cmin, list):
            cmin = [cmin] * in_dim
        if not isinstance(cmax, list):
            cmax = [cmax] * in_dim
   
        self.cmin = torch.tensor(cmin)  # ... y x
        self.cmax = torch.tensor(cmax)  # ... y x

        # 将cmin和cmax作为缓冲区注册到模型中，表示不会被当作模型参数进行优化，但会随模型一起移动到GPU或CPU上
        self.register_buffer('cmin_gpu', self.cmin.clone())  # ... y x
        self.register_buffer('cmax_gpu', self.cmax.clone())  # ... y x

        self.s_dims = torch.tensor(s_dims)  # ... h w

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_levels = num_levels

        self.rbf_type = rbf_type #TODO 采用rbf的类别
        self.point_nn_kernel = point_nn_kernel

        self.lc_init = lc_init
        self.lcb_init = lcb_init
        self.w_init = w_init
        self.b_init = b_init
        self.a_init = a_init

        self.act = act_dict[act]
        
        self.lc_act = act_dict[lc_act]

        self.rbf_suffixes = rbf_suffixes
        self.rbf_lc0_normalize = rbf_lc0_normalize

        self.pe_lc0_rbf_freq = pe_lc0_rbf_freq
        self.pe_lc0_rbf_keep = pe_lc0_rbf_keep
        self.pe_lc0_freq = pe_lc0_freq
        self.pe_hg0_freq = pe_hg0_freq
        self.kc_mult = 1  # kc_mult=2 means extra copy of initial kc

        #TODO 调试代码
        self.use_train_knn = use_train_knn


        lc0_dim = n_hidden_fl
        hg0_dim = num_levels*level_dim
        lcb0_dim = lc0_dim + hg0_dim
        self.backbone_in_dim = lcb0_dim
        lcb0_dim = self.backbone_in_dim

        if len(self.pe_lc0_freq) >= 2:
            self.pe_lc0_freqs = torch.linspace(np.log2(self.pe_lc0_freq[0]), np.log2(self.pe_lc0_freq[1]), 
                                               hidden_dim, device=0)
            #self.pe_lc0_freqs = torch.exp2(self.pe_lc0_freqs)
            self.pe_lc0_freqs = exp2(self.pe_lc0_freqs)
        else:
            self.pe_lc0_freqs = None
        if len(self.pe_lc0_rbf_freq) >= 2 and self.pe_lc0_rbf_keep < lc0_dim:
            self.pe_lc0_rbf_freqs = torch.linspace(np.log2(self.pe_lc0_rbf_freq[0]), np.log2(self.pe_lc0_rbf_freq[1]), 
                                                   lc0_dim - self.pe_lc0_rbf_keep, device=0)
            #self.pe_lc0_rbf_freqs = torch.exp2(self.pe_lc0_rbf_freqs)
            self.pe_lc0_freqs = exp2(self.pe_lc0_freqs)
        else:
            self.pe_lc0_rbf_freqs = None

        self.kc_init_regular = {}
        for k, v in kc_init_config.items():
            if v['type'] == 'none':
                self.kc_init_regular[k] = True
            else:
                self.kc_init_regular[k] = False

        # Get reference number of params
        # get_encoder()返回一个编码器对象
        temp, _ = get_encoder('hashgrid', input_dim=in_dim, num_levels=num_levels_ref, 
            level_dim=level_dim_ref, base_resolution=base_resolution_ref, log2_hashmap_size=log2_hashmap_size_ref, 
            desired_resolution=desired_resolution, allocate_params=False)
        
        # 编码器与MLP解码器的总参数量
        n_params_ref = temp.n_params+(32+hidden_dim*(num_layers-2)+out_dim)*hidden_dim+(hidden_dim*(num_layers-1)+out_dim)

        # PE 根据频率生成位置编码，并将其维度整合到网络的输入维度中
        self.pe_x = None
        if len(pe_freqs) > 0:
            P = torch.cat([torch.eye(self.in_dim)*2**i for i in pe_freqs], 1)
            self.pe_x = util_network.PE(P)
            self.backbone_in_dim += self.pe_x.out_dim

        # Hash grid 创建并初始化一个hashgrid编码器，并对其嵌入层(embeddings)进行初始化
        if num_levels > 0:
            self.hg0, _ = get_encoder('hashgrid', input_dim=in_dim, num_levels=num_levels, 
                level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, 
                desired_resolution=desired_resolution)
            self.hg0.embeddings.data.uniform_(-self.lc_init[0], self.lc_init[0])

        # Backbone
        backbone = []
        self.w_dims = [] # 维度列表
        for l in range(num_layers):
            if l == 0:
                in_dim_i = self.backbone_in_dim
            else:
                in_dim_i = self.hidden_dim
            if l == num_layers - 1:
                out_dim_i = self.out_dim
            else:
                out_dim_i = self.hidden_dim
            self.w_dims.append([out_dim_i, in_dim_i])
            backbone.append(nn.Linear(in_dim_i, out_dim_i, bias=True))
        self.backbone = nn.ModuleList(backbone)

        # 创建一个名为lcb0的可学习参数
        self.lcb0 = torch.nn.Parameter(torch.zeros(lcb0_dim))

        # Activation parameters 为每一层（包括输入和输出层）创建激活函数相关的可学习参数
        for l in range(num_layers+1):
            setattr(self, f'a{l}', torch.nn.Parameter(torch.ones(1) * self.a_init[l]))

        # RBF parameters 设置径向基函数层的相关参数，自动计算核的数量，初始化RBF层的核参数，并设置对应的RBF激活函数
        if n_kernel == 'auto':
            n_params_rbf = n_params_ref - self.count_params(exclude_rbf_lc0=True, verbose=False)['total'][-1]
            n_kernel = int(n_params_rbf // (lc0_dim + util_misc.get_rbf_params_per_kernel(
                self.rbf_type, in_dim, self.kc_mult)))
        if self.kc_init_regular['0']:
            n_kernel = util_misc.get_lower_int_power(n_kernel, in_dim)
        print('n_kernel:', n_kernel)
        self.n_kernel = n_kernel
        self.lc_dims = [[n_kernel, lc0_dim]]
        self.kc0, self.ks0, self.lc0, self.ks_dims, k_dims, kc_interval = self.create_rbf_params(
            self.rbf_type, n_kernel, in_dim, lc0_dim, sparse_embd_grad, 
            self.cmin, self.cmax, ks_alpha, is_bag=False)
        self.register_buffer(f'k_dims_0', k_dims)
        self.register_buffer(f'kci0', kc_interval)
        
        self.init_lc()
        self.init_w_b()

        # 根据RBF的类型，动态获取对应的RBF函数
        self.rbf_fn = eval(f'util_network.rbf_{self.rbf_type}_fb')


    #TODO 为RBF层生成核中心、核尺度等参数，并根据不同类型的RBF函数对这些参数进行初始化和调整
    def create_rbf_params(self, rbf_type, n_kernel, in_dim, lc_dim, sparse_embd_grad, 
                          cmin, cmax, ks_alpha, scale_grad_by_freq=False, is_bag=True):
        if rbf_type.endswith('_a'):
            ks_dims = [in_dim, in_dim]
        elif rbf_type.endswith('_d'):
            ks_dims = [in_dim]
        elif rbf_type.endswith('_s'):
            ks_dims = [1]
        else:
            raise NotImplementedError
        kc = torch.nn.Embedding(n_kernel, in_dim, scale_grad_by_freq=False, sparse=sparse_embd_grad)
        ks = torch.nn.Embedding(n_kernel, np.prod(ks_dims), scale_grad_by_freq=False, 
            sparse=sparse_embd_grad)
        if rbf_type.endswith('_a'):
            ks.weight.data = torch.eye(in_dim)[None, ...].repeat(n_kernel, 1, 1).reshape(n_kernel, -1)
        elif rbf_type.endswith('_d'):
            ks.weight.data[:] = 1
        elif rbf_type.endswith('_s'):
            ks.weight.data[:] = 1
        else:
            raise NotImplementedError

        if is_bag:
            lc = torch.nn.EmbeddingBag(n_kernel, lc_dim, scale_grad_by_freq=scale_grad_by_freq, 
                                       sparse=sparse_embd_grad, mode='sum')
        else:
            lc = torch.nn.Embedding(n_kernel, lc_dim, scale_grad_by_freq=scale_grad_by_freq, 
                                    sparse=sparse_embd_grad)

        # Initialize
        n_kernel_per_dim = int(math.ceil(n_kernel**(1/in_dim)))
        k_dims = [n_kernel_per_dim] * in_dim
        points, _, side_interval = util_misc.get_grid_points(k_dims, align_corners=True, 
            vmin=cmin.tolist(), vmax=cmax.tolist())
        points = points.flip(-1)  # x y ...
        side_interval = side_interval.flip(-1)  # x y ...
        kc.weight.data = points[:n_kernel]
        if rbf_type.startswith('nlin_'):
            if rbf_type.endswith('_s'):
                ks.weight.data *= 1 / side_interval.mean()
            elif rbf_type.endswith('_d'):
                ks.weight.data *= 1 / side_interval[None]
            else:
                raise NotImplementedError
        else:
            if rbf_type.endswith('_s'):
                ks.weight.data *= 1 / (side_interval.mean() / 2) ** 2
            elif rbf_type.endswith('_d'):
                ks.weight.data *= 1 / (side_interval[None] / 2) ** 2
            elif rbf_type.endswith('_a'):
                ks.weight.data *= (1 / (side_interval / 2) ** 2)[:, None].repeat(1, side_interval.shape[0]).view(1, -1)
            else:
                raise NotImplementedError
        ks.weight.data *= ks_alpha

        return kc, ks, lc, ks_dims, torch.tensor(k_dims).flip(-1), side_interval

    # 根据不同的RBF类型，动态计算RBF层中核中心和核尺度的参数数量，并进行适当的缩放和累加
    def count_params_rbf_i(self, suffix, rbf_type, kc_mult=1):
        params = util_misc.count_parameters(getattr(self, f'lc{suffix}') if suffix=='0' else getattr(self, suffix))
        kc = getattr(self, f'kc{suffix}')
        ks = getattr(self, f'ks{suffix}')
        if suffix[0] == 'w' or suffix[0] == 'b' or suffix[0] == 'a':
            pass
        else:
            params += util_misc.count_parameters(kc) * kc_mult
            if rbf_type.endswith('_a'):
                params += (util_misc.count_parameters(ks)*(self.in_dim+1)/2/self.in_dim).astype(np.int64)
            elif rbf_type.endswith('_s') or rbf_type.endswith('_d'):
                params += (util_misc.count_parameters(ks)).astype(np.int64)
            else:
                raise NotImplementedError
        return params

    # 计算模型的各个部分的参数数量
    def count_params(self, exclude_rbf_lc0=False, verbose=True):
        params = {}
        params['hg0'] = util_misc.count_parameters(self.hg0) if self.num_levels > 0 else np.array([0, 0, 0], dtype=np.int64)
        params['mlp'] = util_misc.count_parameters(self.backbone)
        params['rbf_lcb0'] = util_misc.count_parameters(self.lcb0)

        if not exclude_rbf_lc0:
            params['rbf_lc0'] = self.count_params_rbf_i('0', self.rbf_type, self.kc_mult)

        params['total'] = sum([v for _, v in params.items()])
        if verbose:
            for k, v in params.items():
                print(f'{k} params: {v}')

        return params

    # 初始化RBF层的参数
    def init_lc(self):
        with torch.no_grad():
            for i in range(1):
                lci = getattr(self, f'lc{i}') if hasattr(self, f'lc{i}') else None
                lcbi = getattr(self, f'lcb{i}') if hasattr(self, f'lcb{i}') else None
                if isinstance(lci, torch.nn.Embedding) or isinstance(lci, torch.nn.EmbeddingBag):
                    lci = lci.weight
                n_in_i, n_out_i = self.lc_dims[i][-2], self.lc_dims[i][-1]
                if lci is not None:
                    val = self.lc_init[i]
                    nn.init.uniform_(lci, -val, val)
                if lcbi is not None:
                    val = self.lcb_init[i]
                    nn.init.uniform_(lcbi, -val, val)

    # 初始化MLP层的权重和偏置
    def init_w_b(self):
        with torch.no_grad():
            for l in range(self.num_layers):
                out_dim, in_dim = self.w_dims[l]

                wi = self.backbone[l].weight
                if self.w_init[l] is None:
                    pass
                elif self.w_init[l] == 'act':
                    if self.act == 2:
                        if l == 0:
                            nn.init.uniform_(wi, -1 / in_dim, 1 / in_dim)
                        else:
                            nn.init.uniform_(wi, -np.sqrt(6 / in_dim) / 30, np.sqrt(6 / in_dim) / 30)
                else:
                    val = self.w_init[l]
                    nn.init.uniform_(wi, -val, val)
                
                bi = self.backbone[l].bias
                if self.b_init[l] is not None:
                    val = self.b_init[l]
                    nn.init.uniform_(bi, -val, val)
    
    # get_kc和get_ks方法用于获取特定的RBF参数
    def get_kc(self, suffix):
        return getattr(self, 'kc' + suffix).weight

    def get_ks(self, suffix):
        return getattr(self, 'ks' + suffix).weight.view(-1, *self.ks_dims)

    # 裁剪RBF参数的值，确保其在合理范围内，防止训练过程中的不稳定性
    def clip_rbf_params(self):
        for k in self.rbf_suffixes:
            if getattr(self, 'ks'+k).weight.requires_grad:
                if k == '0':
                    vmin_factor = 100
                    vmax_factor = 10
                else:
                    vmin_factor = 1
                    vmax_factor = 2
                util_misc.clip_kw_sq(getattr(self, 'ks'+k).weight, self.rbf_type, 
                                     self.cmin.flip(-1), self.cmax.flip(-1), 
                                     self.s_dims.flip(-1), is_ks=True, is_flat=True, 
                                     vmin_factor=vmin_factor, vmax_factor=vmax_factor,
                                     vmin_min=None)

    # 计算给定点集x_g在核中心kc上的邻近点索引
    def update_point_kernel_idx_i(self, x_g, kc, device=0):
        '''
        x_g: [p d], tensor
        '''
        kd_tree = KDTree(kc.detach().cpu().numpy())
        _, kernel_idx = util_clustering.query_chunked(kd_tree, x_g.cpu().numpy(), 
            k=self.point_nn_kernel, sqr_dists=True, chunk_size=int(2e8), return_dist=False)
        point_kernel_idx = torch.tensor(kernel_idx.astype(np.int32), device=device)  # [p topk]
        if point_kernel_idx.ndim == 1:
            point_kernel_idx = point_kernel_idx[:, None]
        return point_kernel_idx
    
    # 更新每个RBF后缀的点核索引
    def update_point_kernel_idx(self, x_g, device=0):
        t = time.time()
        for k in self.rbf_suffixes:
            if not self.kc_init_regular[k] and self.point_nn_kernel > 0:
                point_kernel_idx = self.update_point_kernel_idx_i(x_g, getattr(self, f'kc{k}').weight, device)
                setattr(self, f'point_kernel_idx_{k}', point_kernel_idx)
        print('Update point kernel idx', time.time() - t)

    # 获取点的邻近核索引
    def forward_kernel_idx(self, x_g, point_idx, suffix):
        if self.use_train_knn and hasattr(self, 'point_kernel_idx_' + suffix):
            # Use pre-computed knn
            point_kernel_idx = getattr(self, 'point_kernel_idx_' + suffix)
            kernel_idx = point_kernel_idx[point_idx.to(point_kernel_idx.device)].to(x_g.device)  # [p topk]
        elif self.kc_init_regular[suffix]:
            # Find first ring neighbors in regular grid
            kernel_idx = util_misc.get_multi_index_first_ring(x_g, 
                getattr(self, f'k_dims_{suffix}'), getattr(self, f'kci{suffix}'), 
                self.cmin_gpu.flip(-1), self.cmax_gpu.flip(-1), True)  # (p, n_ngb, n_in)
            kernel_idx = util_misc.ravel_multi_index(kernel_idx, getattr(self, f'k_dims_{suffix}'))  # (p, n_ngb)
        else:
            # Knn on the fly
            kd_tree = KDTree(getattr(self, 'kc' + suffix).weight.detach().cpu().numpy())
            _, kernel_idx = util_clustering.query_chunked(kd_tree, x_g.cpu().numpy(), 
                k=self.point_nn_kernel, sqr_dists=True, chunk_size=int(2e8), return_dist=False)
            kernel_idx = torch.tensor(kernel_idx.astype(np.int32), device=x_g.device)  # [p topk]
            if kernel_idx.ndim == 1:
                kernel_idx = kernel_idx[:, None]
            
        return kernel_idx

    #TODO 计算径向基函数(RBF)的输出
    def forward_rbf(self, x_g, kernel_idx, suffix):
        """
        x_g: 输入的坐标张量
        kernel_idx: 核索引 
        suffix: 用于从对象中获取不同的核函数参数的后缀
        """
        if kernel_idx is None:  # Use all kernels for each point
            kc = getattr(self, 'kc' + suffix).weight[None]  # [1 k d] 核中心 k为核数量; d为坐标维度
            ks = getattr(self, 'ks' + suffix).weight[None]  # 核尺度
            ks = ks.view(*ks.shape[:2], *self.ks_dims)  # [1 k d d] or [1 k 1]
            rbf_out, _ = self.rbf_fn(x_g, kc, ks)  # [p k_topk]
        else: # 采用核索引指定的部分核
            kc = getattr(self, 'kc' + suffix)(kernel_idx)  # [p k d]
            ks = getattr(self, 'ks' + suffix)(kernel_idx).view(*kernel_idx.shape, *self.ks_dims)  # [p k d d] or [p k d] or [p k 1]
            rbf_out, _ = self.rbf_fn(x_g, kc, ks)  # [p k_topk] p为点数量; k_topk表示使用的核数量
        return rbf_out

    def forward(self, x_g, point_idx, **kwargs):
        """
        x_g: (p, d), coords within entire domain,  (p 是点的数量; d 是维度)
        point_idx: (p), idx within block (点的索引)
        return: (p, n_out)
        """
        out_other = {} # 额外的输出信息

        suffix = '0' # 与获取核参数相关

        # 使用所有的核函数计算每个点
        if self.point_nn_kernel <= 0:  # Use all kernels for each point
            rbf_out = self.forward_rbf(x_g, None, suffix)  # [p nk]
            out = rbf_out @ self.lc0.weight  # [p hfl]
        else: 
            # 选取核函数
            #print("#####################################Data shape:", x_g.shape)
            #print("####################################Query shape:", point_idx.shape)


            kernel_idx = self.forward_kernel_idx(x_g, point_idx, suffix)
            rbf_out = self.forward_rbf(x_g, kernel_idx, suffix)  # [p k_topk]
            if self.rbf_lc0_normalize:
                rbf_out = rbf_out / (rbf_out.detach().sum(-1, keepdim=True) + 1e-8)

            out = self.lc0(kernel_idx)  # [p k_topk d_lc0]
            rbf_out = rbf_out[..., None]  # [p k_topk 1]

            # 频域处理
            if len(self.pe_lc0_rbf_freq) >= 2 and self.pe_lc0_rbf_keep < out.shape[-1]:
                if self.pe_lc0_rbf_keep > 0:
                    rbf_out = torch.cat([rbf_out.expand(-1, -1, self.pe_lc0_rbf_keep), 
                        torch.sin(rbf_out * self.pe_lc0_rbf_freqs[None, None])], -1)  # [p k_topk d_lc0]
                else:
                    rbf_out = torch.sin(rbf_out * self.pe_lc0_rbf_freqs[None, None])  # [p k_topk d_lc0]

            #TODO 自适应RBF的核心计算步骤
            out = (out * rbf_out).sum(1)  # [p d_lc0]

        if self.num_levels > 0:
            #TODO self.hg0()是初始化的hashgrid编码器
            out_hg = self.hg0(x_g / self.cmax_gpu.flip(-1)[None])  # [p d_hg0]
        else:
            out_hg = None
            
        #TODO 同时采用自适应RBF和基于网格的RBF
        if out_hg is not None:
            out = torch.cat([out_hg, out], -1)
        out = out + self.lcb0[None]


        h = out
        if self.lc_act == 0:
            pass
        elif self.lc_act == 1:
            h = F.relu(h, inplace=True)
        elif self.lc_act == 2:
            h = util_network.scaledsin_activation(h, self.a0[None])
        else:
            raise NotImplementedError
        

        # 位置编码
        if self.pe_x is not None:
            h = torch.cat((h, self.pe_x(x_g)), dim=-1)

        #TODO 最终的MLP解码器
        for l in range(self.num_layers):
            h = self.backbone[l](h)
            
            if l == 0 and self.pe_lc0_freqs is not None:
                h = h + torch.sin(h * self.pe_lc0_freqs[None])
            elif l != self.num_layers - 1:
                if self.act == 0:
                    pass
                elif self.act == 1:
                    h = F.relu(h, inplace=True)
                elif self.act == 2:
                    h = util_network.scaledsin_activation(h, getattr(self, f'a{l+1}')[None])
                elif self.act == 3:
                    h = util_network.gaussian_activation(h, 30)
                else:
                    raise NotImplementedError

        return h, out_other


if __name__== '__main__':
    #############Test Model Complexity #############
    #import argparse
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    

    x = torch.randn(1, 48, 48, 3).cuda()
    point_idx = torch.randn(11).cuda()


    # 创建一个空的 args 对象
    #args = argparse.Namespace()
    # Model
    cmin = 0
    cmax = 1
    s_dims = 2
    
    rbf_type = 'ivq_a'
    

    
    lc_act = 'none'
  
    lc_init = [1e-4]
    lcb_init = [1e-4]
    w_init = [None, None, None]
    b_init = [None, None, None]
    a_init = [9, 30, 30, 30]

    rbf_suffixes = ['0']

    kc_init_config = {}
    kc_init_config['0'] = {'type': 'v3', 'points_sampling': 1, 'reg_sampling': 0, 'weight_exp': 1, 'weight_thres': 0, 'n_iter': 10, 'weight_src': 'gt_grad'}

    kw_init_config = {}
    kw_init_config['0'] = {'type': 'v3', 'points_sampling': 1, 'alpha': 0.5, 'weight_exp': 1, 'weight_thres': 0, 'weight_src': 'gt_grad'}

    pe_lc0_freq = [1e1, 1e2]
    pe_lc0_rbf_freq = [2**-3, 2**12]

    pe_freqs = []
    pe_hg0_freq = []
    pe_lc0_rbf_keep = 0


    model = IMGNetwork(cmin=cmin, cmax=cmax, w_init=w_init, b_init=b_init, a_init=a_init).cuda()
    #model = rbf(args).cuda()
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output[0].shape)
