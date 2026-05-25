import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register


################################# 管理MLP的权重与偏置(LIIF-A) #####################################

class gen_basis(nn.Module):
    def __init__(self, args):
        super(gen_basis, self).__init__()
        self.basis_num =  args.basis_num # 输入特征维度
        self.hidden = args.hidden # 隐藏层维度
        self.state = args.state # [train, ]
        self.path=args.path # 测试加载模型路径

    #TODO 初始化权重与偏置
    def init_basis_bias(self):
        # 创建权重
        self.w0 = nn.Parameter(torch.Tensor(self.basis_num,self.hidden*580), requires_grad=True)
        nn.init.kaiming_uniform_(self.w0, a=math.sqrt(5))
        self.w1 = nn.Parameter(torch.Tensor(self.basis_num,self.hidden*self.hidden), requires_grad=True)
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        self.w2 = nn.Parameter(torch.Tensor(self.basis_num,self.hidden*self.hidden), requires_grad=True)
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        self.w3 = nn.Parameter(torch.Tensor(self.basis_num,self.hidden*self.hidden), requires_grad=True)
        nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))
        self.w4 = nn.Parameter(torch.Tensor(self.basis_num,3*self.hidden), requires_grad=True)
        # 初始化权重
        nn.init.kaiming_uniform_(self.w4, a=math.sqrt(5))
        basis = [self.w0, self.w1, self.w2, self.w3, self.w4]

        # 创建偏置
        self.bias1 = nn.Parameter(torch.Tensor(self.basis_num,self.hidden), requires_grad=True)
        self.bias2 = nn.Parameter(torch.Tensor(self.basis_num,self.hidden), requires_grad=True)
        self.bias3 = nn.Parameter(torch.Tensor(self.basis_num,self.hidden), requires_grad=True)
        self.bias4 = nn.Parameter(torch.Tensor(self.basis_num,self.hidden), requires_grad=True)
        self.bias5 = nn.Parameter(torch.Tensor(self.basis_num,3), requires_grad=True)
        bias = [self.bias1,self.bias2,self.bias3,self.bias4,self.bias5]

        # 初始化偏置
        for i in range(len(bias)):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(basis[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias[i], -bound, bound)

        return basis,bias


    #TODO 从指定文件中加载预训练的权重和偏置
    def load_basis_for_test_kaiming(self,path):
        model_spec = torch.load(path)['model']
        w0 = model_spec['sd']['basis.w0']
        w1 = model_spec['sd']['basis.w1']
        w2 = model_spec['sd']['basis.w2']
        w3 = model_spec['sd']['basis.w3']
        w4 = model_spec['sd']['basis.w4']
        b0 = model_spec['sd']['basis.bias1']
        b1 = model_spec['sd']['basis.bias2']
        b2 = model_spec['sd']['basis.bias3']
        b3 = model_spec['sd']['basis.bias4']
        b4 = model_spec['sd']['basis.bias5']
        torch.cuda.empty_cache()
        return [w0,w1,w2,w3,w4],[b0,b1,b2,b3,b4]

    def forward(self):
        if self.state=='train':
            print('init_basis_use_kaiming')
            res=self.init_basis_bias()
        else: # test
            print('load_basis_from_model')
            res=self.load_basis_for_test_kaiming(self.path)
        return res

@register('basis')
def make_basis(basis_num=10,hidden=16,state=None,path=None):
    args = Namespace()
    args.basis_num = basis_num
    args.hidden = hidden
    args.state = state
    args.path = path
    return gen_basis(args)
