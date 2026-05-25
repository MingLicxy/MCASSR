#TODO cycle_liif专用
import math
import itertools
import torch
import numpy as np
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F


class CycleLoss(nn.Module):
    def __init__(self, weight_cycle=1.):
        super(CycleLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.weight_cycle = weight_cycle
        
    def forward(self, hr_pred, hr_gt, lr_pred, lr_gt):
        hr_loss_value = self.l1_loss(hr_pred, hr_gt)
        lr_loss_value = self.l1_loss(lr_pred, lr_gt)
        return hr_loss_value + self.weight_cycle * lr_loss_value