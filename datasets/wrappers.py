import functools
import random
import math
import copy
from PIL import Image
import cv2
import numpy as np
import torch
import torch.fft
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
import torch.nn as nn
from datasets import register
from utils import to_pixel_samples
from utils import make_coord

############################################# 功能函数 #############################################
# def resize_fn(img, size):
#    return transforms.ToTensor()(
#        transforms.Resize(size, Image.BICUBIC)(
#            transforms.ToPILImage()(img)))
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img))
    )

# 将一个较大的缩放操作（scale_factor）分解成多个较小的缩放操作（<=scale_base）
def sample_system_scale(scale_factor, scale_base):
    scale_it = []
    s = copy.copy(scale_factor)
    
    if s <= scale_base:
        scale_it.append(s)
    else:
        scale_it.append(scale_base)
        s = s / scale_base

        while s > 1:
            if s >= scale_base:
                scale_it.append(scale_base)
            else:
                scale_it.append(s)
            s = s / scale_base

    return scale_it

# 小波变换
def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0)\
        .expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0)\
        .expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0)\
        .expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0)\
        .expand(in_channels, -1, -1, -1).clone()

    return LL, LH, HL, HH


# Learnable wavelet
def get_learned_wav(in_channels, pool=True):

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0)

    return LL, LH, HL, HH

#BUG K-space裁剪下采样（保留低频去除高频）



########################################################################################################






############################################# 成对数据集处理 #############################################

# in-scale SR测试时，wrapper要对两个文件夹中的数据进行处理
# 固定尺度训练时同上（eg：x2）
@register('sr-implicit-paired') 
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False,
                  sample_q=None, cell_decode=True):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q #TODO 测试未使用
        self.cell_decode = cell_decode

    # 相当于未定义collate_fn，还原默认的batch-stack行为
    def collate_fn(self, datas):
        return default_collate(datas)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #TODO 取得成对图像数据
        img_lr, img_hr = self.dataset[idx]
        
        # 利用HR与LR的高度h之比确定缩放因子s（宽度w之比也许不等于s）
        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale

        # 未指定输入尺寸时将LR尺寸作为输入尺寸
        if self.inp_size is None: 
            h_lr, w_lr = img_lr.shape[-2:]
            # 依据输入尺寸与缩放因子裁剪HR使之与LR对齐
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size # h_lr=w_lr=48
            # 随机选取裁剪起始位置，并且裁剪LR
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            # 在HR中找到相应的裁剪起始位置，并且裁剪HR
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        # 数据增强（水平翻转，竖直翻转，对角线转置）
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # 获取HR的coord-rgb pairs
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        # 抽样特定数量的采样点坐标以及对应的RGB值
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        # 表示各个采样点处的像素尺寸
        cell = torch.ones_like(hr_coord)

        if self.cell_decode:
           cell[:, 0] *= 2 / crop_hr.shape[-2]
           cell[:, 1] *= 2 / crop_hr.shape[-1]
        else:
           cell[:, 0] *= 1 / crop_hr.shape[-2]
           cell[:, 1] *= 1 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

 

#TODO liif_cycle专用（返回LR的coord/cell）
@register('sr-implicit-paired-cycle') 
class SRImplicitPairedCycle(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False,
                  sample_q=None, cell_decode=True):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q #TODO 测试未使用
        self.cell_decode = cell_decode

    # 相当于未定义collate_fn，还原默认的batch-stack行为
    def collate_fn(self, datas):
        return default_collate(datas)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #TODO 取得成对图像数据
        img_lr, img_hr = self.dataset[idx]
        
        # 利用HR与LR的高度h之比确定缩放因子s（宽度w之比也许不等于s）
        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale

        # 未指定输入尺寸时将LR尺寸作为输入尺寸
        if self.inp_size is None: 
            h_lr, w_lr = img_lr.shape[-2:]
            # 依据输入尺寸与缩放因子裁剪HR使之与LR对齐
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size # h_lr=w_lr=48
            # 随机选取裁剪起始位置，并且裁剪LR
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            # 在HR中找到相应的裁剪起始位置，并且裁剪HR
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        # 数据增强（水平翻转，竖直翻转，对角线转置）
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # 获取HR的coord-rgb pairs
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        lr_coord, lr_rgb = to_pixel_samples(crop_lr.contiguous())

        # 抽样特定数量的采样点坐标以及对应的RGB值
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        # 表示各个采样点处的像素尺寸
        hr_cell = torch.ones_like(hr_coord)
        if self.cell_decode:
           hr_cell[:, 0] *= 2 / crop_hr.shape[-2]
           hr_cell[:, 1] *= 2 / crop_hr.shape[-1]
        else:
           hr_cell[:, 0] *= 1 / crop_hr.shape[-2]
           hr_cell[:, 1] *= 1 / crop_hr.shape[-1]

        lr_cell = torch.ones_like(lr_coord)
        if self.cell_decode:
           lr_cell[:, 0] *= 2 / crop_lr.shape[-2]
           lr_cell[:, 1] *= 2 / crop_lr.shape[-1]
        else:
           lr_cell[:, 0] *= 1 / crop_lr.shape[-2]
           lr_cell[:, 1] *= 1 / crop_lr.shape[-1]

        return {
            'inp': crop_lr,
            'hr_coord': hr_coord,
            'lr_coord': lr_coord,
            'hr_cell': hr_cell,
            'lr_cell': lr_cell,
            'hr_gt': hr_rgb,
            'lr_gt': lr_rgb
        }

#TODO LMF定义
@register('sr-implicit-paired-fast')
class SRImplicitPairedFast(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            h_hr = s * h_lr
            w_hr = s * w_lr
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        
        if self.inp_size is not None:
            x0 = random.randint(0, h_hr - h_lr)
            y0 = random.randint(0, w_hr - w_lr)
            
            hr_coord = hr_coord[x0:x0+self.inp_size, y0:y0+self.inp_size, :]
            hr_rgb = crop_hr[:, x0:x0+self.inp_size, y0:y0+self.inp_size]
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
    

#TODO 基于小波变换的的数据加载
@register('sr-implicit-paired-wave')
class SRImplicitPairedWave(Dataset):

    def __init__(self, dataset, inp_size=None, batch_size=1, window_size=0, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
        self.window_size = window_size
        self.batch_size = batch_size
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels=1)

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, datas):
        coords = []
        hr_list = []
        lr_list = []
        lr_h_list = []     
        lr_l_list = [] 

        scale = datas[0]['img_hr'].shape[-2] // datas[0]['img_lr'].shape[-2]

        if self.inp_size is None:
            # batch_size: 1          

            for idx, data in enumerate(datas):
                lr_h = data['img_lr'].shape[-2]
                lr_w = data['img_lr'].shape[-1]     
                img_h = data['img_hr'].shape[-2]
                img_w = data['img_hr'].shape[-1]                   
                img_wave_h = img_h//2 
                img_wave_w = img_w//2                     

                lr_gray = (255 * data['img_lr']).permute(1,2,0).numpy().astype(np.uint8)
  
                lr_gray = cv2.cvtColor(lr_gray, cv2.COLOR_RGB2GRAY)

                lr_gray = torch.from_numpy(lr_gray)
  
                lr_gray = lr_gray.unsqueeze(0).unsqueeze(0).to(torch.float32) / 255

                lr_ll = self.LL(lr_gray)
                lr_hl = self.HL(lr_gray)
                lr_lh = self.LH(lr_gray)
                lr_hh = self.HH(lr_gray)


                lr_l = lr_ll
                lr_h = torch.cat([lr_hl, lr_lh, lr_hh], dim=1)

                hr_list.append(data['img_hr'])
                lr_list.append(data['img_lr'])
                #hr_h_list.append(crop_hr_h.squeeze(0))
                lr_h_list.append(lr_h.squeeze(0))  
                lr_l_list.append(lr_l.squeeze(0))

        else:
            lr_h = self.inp_size
            lr_w = self.inp_size

            img_h = round(lr_h * scale)
            img_w = round(lr_w * scale)

            img_wave_h = img_h//2 
            img_wave_w = img_w//2 

            coords = make_coord((img_h, img_w), flatten=False).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

            for idx, data in enumerate(datas):
                h0 = random.randint(0, data['img_lr'].shape[-2] - self.inp_size)
                w0 = random.randint(0, data['img_lr'].shape[-1] - self.inp_size)
                crop_lr = data['img_lr'][:, h0:h0 + self.inp_size, w0:w0 + self.inp_size]
                hr_size = self.inp_size * scale
                h1 = h0 * scale
                w1 = w0 * scale
                crop_hr = data['img_hr'][:, h1:h1 + hr_size, w1:w1 + hr_size]

                crop_lr_gray = (255 * crop_lr).permute(1,2,0).numpy().astype(np.uint8)
    
                crop_lr_gray = cv2.cvtColor(crop_lr_gray, cv2.COLOR_RGB2GRAY)

                crop_lr_gray = torch.from_numpy(crop_lr_gray)

                crop_lr_gray = crop_lr_gray.unsqueeze(0).unsqueeze(0).to(torch.float32) / 255

                crop_lr_ll = self.LL(crop_lr_gray)
                crop_lr_hl = self.HL(crop_lr_gray)
                crop_lr_lh = self.LH(crop_lr_gray)
                crop_lr_hh = self.HH(crop_lr_gray)

                crop_lr_l = crop_lr_ll
                crop_lr_h = torch.cat([crop_lr_hl, crop_lr_lh, crop_lr_hh], dim=1)
                #crop_hr_h = torch.cat([crop_hr_hl, crop_hr_lh, crop_hr_hh], dim=1)
                #hr_coord_samples, hr_rgb_samples = to_pixel_samples(crop_hr.contiguous())
                #wave_hr_h_coord_samples, wave_hr_rgb_samples = to_pixel_samples(crop_hr_h.contiguous())  

                hr_list.append(crop_hr)
                lr_list.append(crop_lr)
                #hr_h_list.append(crop_hr_h.squeeze(0))
                lr_h_list.append(crop_lr_h.squeeze(0))                
                lr_l_list.append(crop_lr_l.squeeze(0))

        coords = make_coord((img_h, img_w), flatten=False) \
                                .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        #lr_list = [resize_fn(hr_list[i], (lr_h, lr_w)) for i in range(len(hr_list))]
        inp = torch.stack(lr_list, dim=0)
        hr_rgb = torch.stack(hr_list, dim=0)
        wave_inp = torch.stack(lr_h_list, dim=0)
        wave_inp_l = torch.stack(lr_l_list, dim=0)
        #wave_hr_rgb = torch.stack(hr_h_list, dim=0)       

        cell = torch.ones(2)
        cell[0] *= 2. / img_h
        cell[1] *= 2. / img_w
        cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

        wave_cell = torch.ones(2)
        wave_cell[0] *= 2. / img_wave_h
        wave_cell[1] *= 2. / img_wave_w
        wave_cell = wave_cell.unsqueeze(0).repeat(self.batch_size, 1)       

        if self.inp_size is None and self.window_size != 0:
            # SwinIR Evaluation - reflection padding
            # batch size : 1 for testing
            h_old, w_old = inp.shape[-2:]
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old

            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[..., :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[..., :w_old + w_pad]

            lr_h += h_pad
            lr_w += w_pad

            img_h = round(lr_h * scale)
            img_w = round(lr_w * scale)

            img_wave_h = img_h//2 
            img_wave_w = img_w//2

            coords = make_coord((img_h, img_w), flatten=False) \
                            .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)


            cell = torch.ones(2)
            cell[0] *= 2. / img_h
            cell[1] *= 2. / img_w
            cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

            wave_cell = torch.ones(2)
            wave_cell[0] *= 2. / img_wave_h
            wave_cell[1] *= 2. / img_wave_w
            wave_cell = wave_cell.unsqueeze(0).repeat(self.batch_size, 1)            

        if self.sample_q is None:
            sample_coord = None
        else:
            sample_coord = []       
            for i in range(len(hr_list)):
                flatten_coord = coords[i].reshape(-1, 2)
                sample_list = np.random.choice(flatten_coord.shape[0], self.sample_q, replace=False)
                sample_flatten_coord = flatten_coord[sample_list, :]

                sample_coord.append(sample_flatten_coord)

            sample_coord = torch.stack(sample_coord, dim=0)
        #TODO 返回数据类型是字典（返回小波子带是用来干嘛的）
        return {'inp': inp, 'gt': hr_rgb, 'coords': coords, 'cell': cell, 'sample_coord': sample_coord,
                'wave_lr': wave_inp, 'wave_lr_l': wave_inp_l, 'wave_cell': wave_cell}   

########################################################################################################





############################################# 单数据集处理 #############################################

# 训练时对单个文件夹中的图像数据进行下采样获得HR-LR pairs
# out-of-scale SR测试时同上
@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, cell_decode=True):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q # 在HR中的随机采样点数量
        self.cell_decode = cell_decode

    def collate_fn(self, datas):
        return default_collate(datas)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #TODO 取得单个图像数据
        img = self.dataset[idx]

        # 缩放因子在[scale-min, scale-max]随机采样
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            # 将输入尺寸作为LR的宽高（w_lr=h_lr=int_size）
            h_lr = self.inp_size  # 48
            w_lr = self.inp_size
            h_hr = round(h_lr * s) # [48x1, 48x4=192]
            w_hr = round(w_lr * s)

            #TODO 这里有BUG:要求数据集中的图像的尺寸大于等于192
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        
        # to_pixel_samples()内部调用了make_coord()
        # crop_hr->r_rgb：[c=3,h_hr,w_hr]->[h_hr*w_hr,c=3]; hr_coord：[h_hr*w_hr,2]
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        # 在HR中随机选取sample_q个采样点（减轻计算压力）
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            #TODO
            hr_coord = hr_coord[sample_lst] # [sample,2]
            #TODO
            hr_rgb = hr_rgb[sample_lst] # [sample,3]

        #TODO cell维度为[sample,2]沿bs维度堆叠[bs,sample,2]
        cell = torch.ones_like(hr_coord)
        if self.cell_decode:
           cell[:, 0] *= 2 / crop_hr.shape[-2]
           cell[:, 1] *= 2 / crop_hr.shape[-1]
        else:
           cell[:, 0] *= 1 / crop_hr.shape[-2]
           cell[:, 1] *= 1 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

#TODO liif_cycle专用（返回LR的coord/cell）
@register('sr-implicit-downsampled-cycle')
class SRImplicitDownsampledCycle(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, cell_decode=True):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q # 在HR中的随机采样点数量
        self.cell_decode = cell_decode

    def collate_fn(self, datas):
        return default_collate(datas)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #TODO 取得单个图像数据
        img = self.dataset[idx]

        # 缩放因子在[scale-min, scale-max]随机采样
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            # 将输入尺寸作为LR的宽高（w_lr=h_lr=int_size）
            h_lr = self.inp_size  # 48
            w_lr = self.inp_size
            h_hr = round(h_lr * s) # [48x1, 48x4=192]
            w_hr = round(w_lr * s)

            #TODO 这里有BUG:要求数据集中的图像的尺寸大于等于192
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        
        # to_pixel_samples()内部调用了make_coord()
        # crop_hr->hr_rgb：[c=3,h_hr,w_hr]->[h_hr*w_hr,c=3]; hr_coord：[h_hr*w_hr,2]
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        lr_coord, lr_rgb = to_pixel_samples(crop_lr.contiguous())

        # 在HR中随机选取sample_q个采样点（减轻计算压力）
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            #TODO
            hr_coord = hr_coord[sample_lst] # [sample,2]
            #TODO
            hr_rgb = hr_rgb[sample_lst] # [sample,3]

        #TODO cell维度为[sample,2]沿bs维度堆叠[bs,sample,2]
        hr_cell = torch.ones_like(hr_coord)
        if self.cell_decode:
           hr_cell[:, 0] *= 2 / crop_hr.shape[-2]
           hr_cell[:, 1] *= 2 / crop_hr.shape[-1]
        else:
           hr_cell[:, 0] *= 1 / crop_hr.shape[-2]
           hr_cell[:, 1] *= 1 / crop_hr.shape[-1]

        lr_cell = torch.ones_like(lr_coord)
        if self.cell_decode:
           lr_cell[:, 0] *= 2 / crop_lr.shape[-2]
           lr_cell[:, 1] *= 2 / crop_lr.shape[-1]
        else:
           lr_cell[:, 0] *= 1 / crop_lr.shape[-2]
           lr_cell[:, 1] *= 1 / crop_lr.shape[-1]

        return {
            'inp': crop_lr,
            'hr_coord': hr_coord,
            'lr_coord': lr_coord,
            'hr_cell': hr_cell,
            'lr_cell': lr_cell,
            'hr_gt': hr_rgb,
            'lr_gt': lr_rgb
        }




#TODO SRNO使用
# 'sr-implicit-downsampled-fast'和'sr-implicit-downsampled'不同在于hr_rgb.shape和hr_coord.shape
@register('sr-implicit-downsampled-fast')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q # 这里不使用该参数

    def collate_fn(self, datas):
        return default_collate(datas)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr]
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr # [c=3,h_hr,w_hr]

        # 在HR中随机选择(h_lr*w_lr)个像素索引（作用同sample_q）
        if self.inp_size is not None:
            
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            #idx,_ = torch.sort(idx)
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            #TODO
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1]) # [h_lr,w_lr,2]

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            #TODO 
            hr_rgb = hr_rgb.view(crop_hr.shape[0], h_lr, w_lr) # [c=3,h_lr,w_lr]
        
        #TODO cell张量维度是[2,]默认在bs维度上堆叠[bs,2]
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)
        
        return {
            'inp': crop_lr, # [3, inp_size, inp_size]
            'coord': hr_coord, # [inp_size, inp_size, 2]
            'cell': cell, # [2]
            'gt': hr_rgb # [3, inp_size, inp_size]
        }    



#TODO 基于小波变换的LIWT专用
@register('sr-implicit-downsampled-wave')
class SRImplicitDownsampledWave(Dataset):
    
    def __init__(self, dataset, inp_size=None, batch_size=32, scale_min=1, scale_max=None, window_size=0,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.batch_size = batch_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.scale_max_s6 = 6
        self.scale_max_s8 = 8
        self.scale_max_s12 = 12
        #self.scale_max_s18 = 18
        self.augment = augment
        self.window_size = window_size
        self.sample_q = sample_q

        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels=1)

    def collate_fn(self, datas):
        coords = [] # U(1,4)
        hr_list = []
        lr_list = []

        scale = random.uniform(self.scale_min, self.scale_max)

        coords_s6 = [] # U(1,6)
        hr_list_s6 = []
        lr_list_s6 = []

        scale_s6 = random.uniform(self.scale_min, self.scale_max_s6)

        coords_s8 = [] # U(1,8)
        hr_list_s8 = []
        lr_list_s8 = []

        scale_s8 = random.uniform(self.scale_min, self.scale_max_s8)

        if self.inp_size is None:
            # batch_size: 1
            lr_h = math.floor(datas[0]['inp'].shape[-2] / scale + 1e-9)
            lr_w = math.floor(datas[0]['inp'].shape[-1] / scale + 1e-9)

            img_h = round(lr_h * scale)
            img_w = round(lr_w * scale)

            coords = make_coord((img_h, img_w), flatten=False) \
                                .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

            for idx, data in enumerate(datas):
                crop_hr = data['inp'][:, :img_h, :img_w]
                crop_lr = resize_fn(crop_hr, (lr_h, lr_w))

                hr_list.append(crop_hr)
                lr_list.append(crop_lr)
        else:
            lr_h = self.inp_size
            lr_w = self.inp_size

            img_size_min = 9999
            
            for idx, data in enumerate(datas):
                img_size_min = min(img_size_min, data['inp'].shape[-2], data['inp'].shape[-1])

            img_h = round(lr_h * scale)
            img_w = round(lr_w * scale)

            coords = make_coord((img_h, img_w), flatten=False).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

            for idx, data in enumerate(datas):
                h0 = random.randint(0, data['inp'].shape[-2] - img_h)
                w0 = random.randint(0, data['inp'].shape[-1] - img_w)
                crop_hr = data['inp'][:, h0:h0 + img_h, w0:w0 + img_w]
                crop_lr = resize_fn(crop_hr, (lr_h, lr_w))

                hr_list.append(crop_hr)
                lr_list.append(crop_lr)
 

            img_h_s6 = round(lr_h * scale_s6)
            img_w_s6 = round(lr_w * scale_s6)

            coords_s6 = make_coord((img_h_s6, img_w_s6), flatten=False).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

            for idx, data in enumerate(datas):
                h0_s6 = random.randint(0, data['inp'].shape[-2] - img_h_s6)
                w0_s6 = random.randint(0, data['inp'].shape[-1] - img_w_s6)
                crop_hr_s6 = data['inp'][:, h0_s6:h0_s6 + img_h_s6, w0_s6:w0_s6 + img_w_s6]
                crop_lr_s6 = resize_fn(crop_hr_s6, (lr_h, lr_w))

                hr_list_s6.append(crop_hr_s6)
                lr_list_s6.append(crop_lr_s6)

            img_h_s8 = round(lr_h * scale_s8)
            img_w_s8 = round(lr_w * scale_s8)

            coords_s8 = make_coord((img_h_s8, img_w_s8), flatten=False).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

            for idx, data in enumerate(datas):
                h0_s8 = random.randint(0, data['inp'].shape[-2] - img_h_s8)
                w0_s8 = random.randint(0, data['inp'].shape[-1] - img_w_s8)
                crop_hr_s8 = data['inp'][:, h0_s8:h0_s8 + img_h_s8, w0_s8:w0_s8 + img_w_s8]
                crop_lr_s8 = resize_fn(crop_hr_s8, (lr_h, lr_w))

                hr_list_s8.append(crop_hr_s8)
                lr_list_s8.append(crop_lr_s8)

        inp = torch.stack(lr_list, dim=0)
        hr_rgb = torch.stack(hr_list, dim=0)
   
        inp_s6 = torch.stack(lr_list_s6, dim=0)
        hr_rgb_s6 = torch.stack(hr_list_s6, dim=0)

        inp_s8 = torch.stack(lr_list_s8, dim=0)
        hr_rgb_s8 = torch.stack(hr_list_s8, dim=0)

        cell = torch.ones(2)
        cell[0] *= 2. / img_h
        cell[1] *= 2. / img_w
        cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

        cell_s6 = torch.ones(2)
        cell_s6[0] *= 2. / img_h_s6
        cell_s6[1] *= 2. / img_w_s6
        cell_s6 = cell_s6.unsqueeze(0).repeat(self.batch_size, 1)

        cell_s8 = torch.ones(2)
        cell_s8[0] *= 2. / img_h_s8
        cell_s8[1] *= 2. / img_w_s8
        cell_s8 = cell_s8.unsqueeze(0).repeat(self.batch_size, 1)

        if self.inp_size is None and self.window_size != 0:
            # SwinIR Evaluation - reflection padding
            # batch size : 1 for testing
            h_old, w_old = inp.shape[-2:]
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old

            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[..., :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[..., :w_old + w_pad]

            lr_h += h_pad
            lr_w += w_pad

            img_h = round(lr_h * scale)
            img_w = round(lr_w * scale)

            coords = make_coord((img_h, img_w), flatten=False) \
                            .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)


            cell = torch.ones(2)
            cell[0] *= 2. / img_h
            cell[1] *= 2. / img_w
            cell = cell.unsqueeze(0).repeat(self.batch_size, 1)   

        if self.sample_q is None:
            sample_coord = None
        else:
            sample_coord = []       
            for i in range(len(hr_list)):
                flatten_coord = coords[i].reshape(-1, 2)
                sample_list = np.random.choice(flatten_coord.shape[0], self.sample_q, replace=False)
                sample_flatten_coord = flatten_coord[sample_list, :]

                sample_coord.append(sample_flatten_coord)

            sample_coord = torch.stack(sample_coord, dim=0)

            sample_coord_s6 = []       
            for i in range(len(hr_list_s6)):
                flatten_coord_s6 = coords_s6[i].reshape(-1, 2)
                sample_list_s6 = np.random.choice(flatten_coord_s6.shape[0], self.sample_q, replace=False)
                sample_flatten_coord_s6 = flatten_coord_s6[sample_list_s6, :]

                sample_coord_s6.append(sample_flatten_coord_s6)

            sample_coord_s6 = torch.stack(sample_coord_s6, dim=0)

            sample_coord_s8 = []       
            for i in range(len(hr_list_s8)):
                flatten_coord_s8 = coords_s8[i].reshape(-1, 2)
                sample_list_s8 = np.random.choice(flatten_coord_s8.shape[0], self.sample_q, replace=False)
                sample_flatten_coord_s8 = flatten_coord_s8[sample_list_s8, :]

                sample_coord_s8.append(sample_flatten_coord_s8)

            sample_coord_s8 = torch.stack(sample_coord_s8, dim=0)
        
        #TODO 对应三个不同的训练阶段
        return {'inp': inp, 'gt': hr_rgb, 'coords': coords, 'cell': cell, 'sample_coord': sample_coord, 
                'inp_s6': inp_s6, 'gt_s6': hr_rgb_s6, 'coords_s6': coords_s6, 'cell_s6': cell_s6, 'sample_coord_s6': sample_coord_s6, 
                'inp_s8': inp_s8, 'gt_s8': hr_rgb_s8, 'coords_s8': coords_s8, 'cell_s8': cell_s8, 'sample_coord_s8': sample_coord_s8
                }

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img = self.dataset[idx]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            img = augment(img)

        return {'inp': img}




@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None, augment=False,
                  gt_resize=None, sample_q=None, cell_decode=True):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q
        self.cell_decode = cell_decode

    def collate_fn(self, datas):
        return default_collate(datas)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        if self.cell_decode:
           cell[:, 0] *= 2 / img_hr.shape[-2]
           cell[:, 1] *= 2 / img_hr.shape[-1]
        else:
           cell[:, 0] *= 1 / img_hr.shape[-2]
           cell[:, 1] *= 1 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

#TODO LMF定义
@register('sr-implicit-fixed-resolution')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_res, out_res):
        self.dataset = dataset
        self.inp_res = inp_res if inp_res else [720, 1280]
        self.out_res = out_res if out_res else [1440, 2560]
        self.aspect_ratio = inp_res[-1] / inp_res[-2]  # w / h

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]

        if self.aspect_ratio > img.shape[-1] / img.shape[-2]:
            # Crop to the aspect_ratio
            h_crop = round(img.shape[-1] / self.aspect_ratio)
            img = img[:, (img.shape[-2] - h_crop) // 2:(img.shape[-2] - h_crop) // 2 + h_crop, :]
        elif self.aspect_ratio < img.shape[-1] / img.shape[-2]:
            # Crop to the aspect_ratio
            w_crop = round(img.shape[-2] * self.aspect_ratio)
            img = img[:, :, (img.shape[-1] - w_crop) // 2:(img.shape[-1] - w_crop) // 2 + w_crop]

        [h_lr, w_lr] = self.inp_res
        [h_hr, w_hr] = self.out_res
        img_down = resize_fn(img, (h_lr, w_lr))

        coord = make_coord(self.out_res)
        inp_coord = make_coord(self.inp_res, flatten=False)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h_hr
        cell[:, 1] *= 2 / w_hr

        return {
            'inp': img_down,
            'inp_coord': inp_coord,
            'coord': coord,
            'cell': cell,
            # 'gt': hr_rgb
        }

#TODO CLIP使用
# 成对数据的级联处理（in-of-scale测试，固定尺度训练）
@register('sr-implicit-paired-cascaded')
class SRImplicitPairedCascaded(Dataset):
    def __init__(
        self,
        dataset,
        inp_size=None,
        batch_size=16,
        scale_base=2,
        sample_q=None,
        window_size=0,
        augment=False
    ):
        self.dataset = dataset
        self.inp_size = inp_size
        self.batch_size = batch_size
        self.scale_base = scale_base
        self.sample_q = sample_q
        self.window_size = window_size
        self.augment = augment

    # collate_fn()负责将单个样本列表（通常是由多个 __getitem__ 返回的结果组成的列表）合并成一个批次
    def collate_fn(self, datas):
        # 获取尺度因子
        s = datas[0]['img_hr'].shape[-2] // datas[0]['img_lr'].shape[-2]
        # 分解尺度因子得到级联尺度因子列表（s<=4;scale_base=4）
        self.scale_it = sample_system_scale(s, self.scale_base)

        # 存储坐标网格，HR和LR
        coords = []
        hr_list = []
        lr_list = []
        
        # 未指定输入尺寸
        if self.inp_size is None:
            # batch_size: 1
            for data in (datas):
                lr_h = data['img_lr'].shape[-2]
                lr_w = data['img_lr'].shape[-1]

                lr_list.append(data['img_lr'])
                hr_list.append(data['img_hr'])
        else:
            lr_h = self.inp_size
            lr_w = self.inp_size

            for idx, data in enumerate(datas):
                h0 = random.randint(0, data['img_lr'].shape[-2] - self.inp_size)
                w0 = random.randint(0, data['img_lr'].shape[-1] - self.inp_size)
                crop_lr = data['img_lr'][:, h0:h0 + self.inp_size, w0:w0 + self.inp_size]
                hr_size = self.inp_size * s
                h1 = h0 * s
                w1 = w0 * s
                crop_hr = data['img_hr'][:, h1:h1 + hr_size, w1:w1 + hr_size]

                lr_list.append(crop_lr)
                hr_list.append(crop_hr)

        # 获取对应scale_it中每个级联尺度因子对应的坐标网格
        for idx in range(len(self.scale_it)):
            img_h = round(lr_h * np.prod(self.scale_it[:idx + 1]))
            img_w = round(lr_w * np.prod(self.scale_it[:idx + 1]))

            coord = make_coord((img_h, img_w),
                               flatten=False).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
            coords.append(coord)

        # 获取HR的h,w（prod算总乘积，round四舍五入）
        hr_h = round(lr_h * np.prod(self.scale_it))
        hr_w = round(lr_w * np.prod(self.scale_it))

        # 根据获取的h,w裁剪HR
        for idx in range(len(hr_list)):
            hr_list[idx] = hr_list[idx][..., :hr_h, :hr_w]

        # 将HR-LR转化成张量，并沿bs维度堆叠
        inp = torch.stack(lr_list, dim=0)
        hr_rgb = torch.stack(hr_list, dim=0)

        # 单元格尺寸
        cell = torch.ones(2)
        cell[0] *= 2. / img_h
        cell[1] *= 2. / img_w
        cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

        if self.inp_size is None and self.window_size != 0:
            # SwinIR Evaluation - reflection padding
            # batch size : 1 for testing
            h_old, w_old = inp.shape[-2:]
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old

            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[..., :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[..., :w_old + w_pad]

            lr_h += h_pad
            lr_w += w_pad

            for idx in range(len(self.scale_it)):
                img_h = round(lr_h * np.prod(self.scale_it[:idx + 1]))
                img_w = round(lr_w * np.prod(self.scale_it[:idx + 1]))

                coord = make_coord((img_h, img_w), flatten=False) \
                                .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
                coords[idx] = coord

            cell = torch.ones(2)
            cell[0] *= 2. / img_h
            cell[1] *= 2. / img_w
            cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

        # 获取采样坐标
        if self.sample_q is None:
            sample_coord = None
        else:
            sample_coord = []
            for i in range(len(hr_list)):
                flatten_coord = coords[-1][i].reshape(-1, 2)
                sample_list = np.random.choice(flatten_coord.shape[0], self.sample_q, replace=False)
                sample_flatten_coord = flatten_coord[sample_list, :]
                sample_coord.append(sample_flatten_coord)
            sample_coord = torch.stack(sample_coord, dim=0)

        return {'inp': inp,
                'gt': hr_rgb,
                'coords': coords,
                'cell': cell,
                'sample_coord': sample_coord}

    def __len__(self):
        return len(self.dataset)

    # DataLoader直接调用的是Dataset子类的__getitem__和__len__方法
    # __getitem__方法负责从数据集中获取单个样本
    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        
        # 数据增强
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            img_lr = augment(img_lr)
            img_hr = augment(img_hr)

        return {'img_lr': img_lr, 'img_hr': img_hr}

#TODO CLIP使用
# 单文件夹的图像数据级联处理（out-of-scale测试，训练过程）
@register('sr-implicit-downsampled-cascaded')
class SRImplicitDownsampledCascaded(Dataset):
    def __init__(
        self,
        dataset,
        inp_size=None,
        batch_size=16,
        scale_base=4, # 当scale<=4时，不使用级联策略
        scale_min=1, # [scale_min, scale_max]
        scale_max=None,
        sample_q=None,
        k=1, # 级联层级数
        window_size=0, #TODO 要窗口干嘛？
        augment=False,
        phase='train' # 所处状态
    ):
        self.dataset = dataset
        self.inp_size = inp_size
        self.batch_size = batch_size
        self.scale_base = scale_base
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.sample_q = sample_q
        self.k = k
        self.window_size = window_size
        self.augment = augment
        self.phase = phase

        self.counter = 0

    # 组织__getitem__的输出（其功能也可以直接在__getitem__中实现）
    def collate_fn(self, datas):
        coords = []
        hr_list = []

        #TODO 由inp_size决定以不同的方式获取级联尺度因子列表
        if self.inp_size is None: # test: inp_size=none
            self.scale_it = sample_system_scale(self.scale_max, self.scale_base)
        else: # train: inp_size=48
            self.scale_it = []
            for idx in range(len(self.scale_max)):
                self.scale_it.append(random.uniform(self.scale_min[idx], self.scale_max[idx]))

        if self.phase == 'train':
            if self.counter % self.k == 0:
                self.counter = 1
            else:
                del self.scale_it[self.counter:]
                self.counter += 1

        if self.inp_size is None: 
            # batch_size: 1
            lr_h = math.floor(datas[0]['inp'].shape[-2] / np.prod(self.scale_it) + 1e-9)
            lr_w = math.floor(datas[0]['inp'].shape[-1] / np.prod(self.scale_it) + 1e-9)

            for idx in range(len(self.scale_it)):
                img_h = round(lr_h * np.prod(self.scale_it[:idx + 1]))
                img_w = round(lr_w * np.prod(self.scale_it[:idx + 1]))

                coord = make_coord((img_h, img_w), flatten=False) \
                                    .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

                coords.append(coord)

            for idx, data in enumerate(datas):
                crop_hr = data['inp'][:, :img_h, :img_w]
                hr_list.append(crop_hr)
        else: 
            lr_h = self.inp_size
            lr_w = self.inp_size

            img_size_min = 9999

            # 计算输入图像的最小尺寸
            for idx, data in enumerate(datas):
                img_size_min = min(img_size_min, data['inp'].shape[-2], data['inp'].shape[-1])

            # 防止在最小尺寸图像上裁剪出错
            if np.prod(self.scale_it) * self.inp_size > img_size_min:
                img_ratio = img_size_min / (np.prod(self.scale_it) * self.inp_size)
                self.scale_it[0] *= img_ratio

            # 获取每个级联尺度因子对应的坐标网格
            for idx in range(len(self.scale_it)):
                img_h = round(lr_h * np.prod(self.scale_it[:idx + 1]))
                img_w = round(lr_w * np.prod(self.scale_it[:idx + 1]))

                coord = make_coord((img_h, img_w),
                                   flatten=False).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
                coords.append(coord) 
            
            # 随机裁剪获取HR
            for idx, data in enumerate(datas):
                h0 = random.randint(0, data['inp'].shape[-2] - img_h)
                w0 = random.randint(0, data['inp'].shape[-1] - img_w)
                crop_hr = data['inp'][:, h0:h0 + img_h, w0:w0 + img_w]
                hr_list.append(crop_hr)
        
        # 对HR进行双三次下采样获取对应的LR
        lr_list = [resize_fn(hr_list[i], (lr_h, lr_w)) for i in range(len(hr_list))]
        inp = torch.stack(lr_list, dim=0)
        hr_rgb = torch.stack(hr_list, dim=0)

        cell = torch.ones(2)
        cell[0] *= 2. / img_h
        cell[1] *= 2. / img_w
        cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

        if self.inp_size is None and self.window_size != 0:
            # SwinIR Evaluation - reflection padding
            # batch size : 1 for testing
            h_old, w_old = inp.shape[-2:]
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old

            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[..., :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[..., :w_old + w_pad]

            lr_h += h_pad
            lr_w += w_pad

            for idx in range(len(self.scale_it)):
                img_h = round(lr_h * np.prod(self.scale_it[:idx + 1]))
                img_w = round(lr_w * np.prod(self.scale_it[:idx + 1]))

                coord = make_coord((img_h, img_w), flatten=False) \
                                .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
                coords[idx] = coord

            cell = torch.ones(2)
            cell[0] *= 2. / img_h
            cell[1] *= 2. / img_w
            cell = cell.unsqueeze(0).repeat(self.batch_size, 1)
        
        # 根据给定的sample_q值从HR的坐标网格中随机采样一部分坐标，并将这些采样的坐标存储在sample_coord
        if self.sample_q is None:
            sample_coord = None
        else:
            sample_coord = []
            for i in range(len(hr_list)):
                # 将坐标网格展开为一个二维张量，每行包含一个(x, y)坐标
                flatten_coord = coords[-1][i].reshape(-1, 2)
                sample_list = np.random.choice(flatten_coord.shape[0], self.sample_q, replace=False)
                sample_flatten_coord = flatten_coord[sample_list, :]
                sample_coord.append(sample_flatten_coord)
            # [len(hr_list), sample_q, 2]
            sample_coord = torch.stack(sample_coord, dim=0)

        return {'inp': inp,
                'gt': hr_rgb,
                'coords': coords,
                'cell': cell,
                'sample_coord': sample_coord}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            img = augment(img)

        return {'inp': img}
########################################################################################################






############################################# 特殊方法数据处理 #############################################
#TODO OPE train
@register('ope-sample-train')  # abandoned
class OPE_sample(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=True, norm=True, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.norm = norm  #TODO 归一化

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        ### prepare sample
        s = random.uniform(self.scale_min, self.scale_max)

        w_lr = self.inp_size
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img.shape[-2] - w_hr)
        y0 = random.randint(0, img.shape[-1] - w_hr)
        crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        #TODO
        crop_lr = resize_fn(crop_hr, w_lr)  # img_lr_s

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        #TODO
        if self.norm:
            crop_lr = (crop_lr - 0.5) / 0.5
            hr_rgb = (hr_rgb - 0.5) / 0.5

        sample_batch = {
            'lr_img': crop_lr,
            'coords_sample': hr_coord,
            'gt_sample': hr_rgb
        }

        return sample_batch


@register('ope-sample-train-2')
class OPE_sample_2(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=True, norm=True, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.norm = norm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        ### prepare sample
        s = random.uniform(self.scale_min, self.scale_max)

        w_lr = self.inp_size
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img.shape[-2] - w_hr)
        y0 = random.randint(0, img.shape[-1] - w_hr)
        crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr = resize_fn(crop_hr, w_lr)  # img_lr_s

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        #TODO
        crop_hr = resize_fn(crop_hr, w_hr * 3)
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        if self.norm:
            crop_lr = (crop_lr - 0.5) / 0.5
            hr_rgb = (hr_rgb - 0.5) / 0.5

        sample_batch = {
            'lr_img': crop_lr,
            'coords_sample': hr_coord,
            'gt_sample': hr_rgb
        }

        return sample_batch


#TODO OPE test
@register('ope-patch-eval')  # use for eval patch
class OPE_patch(Dataset):
    def __init__(self, dataset, inp_size=None, scale_factor=None, augment=False, norm=True):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.norm = norm
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        # crop_size = int(self.inp_size * (1 + random.random()))
        crop_size = int(self.inp_size * self.scale_factor)
        img_h, img_w = img.shape[-2:]
        if crop_size >= min(img_h, img_w):
            crop_size_t = min(img_h, img_w)
            x0 = random.randint(0, img_h - crop_size_t)
            y0 = random.randint(0, img_w - crop_size_t)
            img_hr = img[:, x0:x0 + crop_size_t, y0:y0 + crop_size_t]
            img_hr = resize_fn(img_hr, crop_size)
            img_lr = resize_fn(img_hr, self.inp_size)
        else:
            x0 = random.randint(0, img_h - crop_size)
            y0 = random.randint(0, img_w - crop_size)
            img_hr = img[:, x0:x0 + crop_size, y0:y0 + crop_size]
            img_lr = resize_fn(img_hr, self.inp_size)
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            img_lr = augment(img_lr)
            img_hr = augment(img_hr)

        if self.norm:
            img_lr = (img_lr - 0.5) / 0.5
            img_hr = (img_hr - 0.5) / 0.5

        return {
            'lr': img_lr,  # C,h,w
            'hr': img_hr,  # C,H,W
        }

#TODO OPE test paired
@register('sr-cut-paired')  # use for test div2k/benchmark x2/x3/x4
class SRcutPaired(Dataset):

    def __init__(self, dataset, inp_size=None, norm=True, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.norm = norm #TODO

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2]  # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        
        #TODO
        if self.norm:
            crop_lr = (crop_lr - 0.5) / 0.5
            crop_hr = (crop_hr - 0.5) / 0.5

        return {
            'lr': crop_lr,
            'gt': crop_hr,
        }

#TODO OPE test non-paired
@register('sr-cut-downsampled-test')  # use for div2k/benchmark x6 x8 ... x20
class CutDownsampledTest(Dataset):
    def __init__(self, dataset, test_scale=None, augment=False, norm=True):
        self.dataset = dataset
        self.augment = augment
        self.norm = norm #TODO
        self.test_scale = test_scale #TODO 测试上采样尺度

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = self.test_scale
        h_lr = math.floor(img.shape[-2] / s + 1e-9)
        w_lr = math.floor(img.shape[-1] / s + 1e-9)
        img = img[:, :round(h_lr * s), :round(w_lr * s)]
        img_down = resize_fn(img, (h_lr, w_lr))
        crop_lr, crop_gt = img_down, img

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_gt = augment(crop_gt)

        if self.norm:
            crop_lr = (crop_lr - 0.5) / 0.5
            crop_gt = (crop_gt - 0.5) / 0.5

        return {
            'lr': crop_lr,
            'gt': crop_gt,
        }





#TODO COZ（真实连续超分）使用
@register('sr-implicit-float-paired-test')
class SRImplicitFloatPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    #TODO
    def set_test_scale(self, evalscale):
        self.dataset.set_test_scale(evalscale)

    def collate_fn(self, datas):
        return default_collate(datas)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        crop_lr, crop_hr = self.dataset[idx]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

#TODO COZ（真实连续超分）使用
@register("sr-implicit-float-paired")
class SRImplicitFloatPaired(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def set_test_scale(self, evalscale):
        self.dataset.set_test_scale(evalscale)

    def collate_fn(self, datas):
        return default_collate(datas)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        w, h = img_hr.shape[-2:]
        # print(img_lr.shape)

        s = img_hr.shape[-2] / img_lr.shape[-2]  # assume int scale
        # print(s)
        if self.inp_size is None:
            # h_lr, w_lr = img_lr.shape[-2:]
            # img_hr = img_hr[:, :int(h_lr * s), :int(w_lr * s)]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0 : x0 + w_lr, y0 : y0 + w_lr]
            w_hr = round(w_lr * s)
            x1 = round(x0 * s)
            y1 = round(y0 * s)
            crop_hr = img_hr[:, x1 : x1 + w_hr, y1 : y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            try:
                sample_lst = np.random.choice(
                    len(hr_coord), self.sample_q, replace=False
                )
                hr_coord = hr_coord[sample_lst]
                hr_rgb = hr_rgb[sample_lst]

            except:
                print(len(hr_coord), s)
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            "inp": crop_lr,
            "coord": hr_coord,
            "cell": cell,
            "gt": hr_rgb,
            "w": w,
            "h": h,
        }
########################################################################################################







######################################### 多对比MRI连续超分 ###############################################

#TODO 多对比MRI专用数据处理（in-scale测试时采用）
@register('mc-sr-implicit-paired')
class MCSRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False,
                  sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def collate_fn(self, datas):
        return default_collate(datas)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #TODO 同时取得三张图形数据
        img_lr, img_hr, img_ref = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale

        # 默认input_size is None
        h_lr, w_lr = img_lr.shape[-2:]
        img_hr = img_hr[:, :h_lr * s, :w_lr * s]
        img_ref = img_ref[:, :h_lr * s, :w_lr * s]
        crop_lr, crop_hr, crop_ref = img_lr, img_hr, img_ref
        crop_ref_lr = resize_fn(img_ref, w_lr)


        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x
            ###### Tar #######
            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            ###### Ref #######
            crop_ref_lr = augment(crop_ref_lr)
            crop_ref = augment(crop_ref)

        ###### Tar #######
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        ###### Ref #######
        _, ref_rgb = to_pixel_samples(crop_ref.contiguous())
        # print(ref_rgb.shape)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            ###### Tar #######
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            ###### Ref #######
            ref_rgb = ref_rgb[sample_lst]
        ref_w = int(np.sqrt(ref_rgb.shape[0]))
        ref_c = ref_rgb.shape[1]
        # print(ref_w,ref_c)
        ref_hr = ref_rgb.contiguous().view(ref_c, ref_w, ref_w)
        ###### Tar #######
        cell = torch.ones_like(hr_coord)
        # 默认cell_decode is True
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]
        

        return {
            'inp': crop_lr,
            'inp_hr_coord': hr_coord,
            'inp_cell': cell,
            'ref': crop_ref_lr, # LR参考
            'ref_hr': ref_hr, # HR参考
            'gt': hr_rgb
        }
    
#TODO 训练时采用，out-of-scale测试时采用
@register('mc-sr-implicit-downsampled')
class MCSRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def collate_fn(self, datas):
        return default_collate(datas)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #TODO 一次取得两个对比度的MRI图像，实际上PD作为Ref
        T2_img, T1_img = self.dataset[idx]
        
        #TODO train:[1,4]  test:[2,3,4,6,8,12] 公倍数: 24
        s = random.uniform(self.scale_min, self.scale_max) 
        
        if self.inp_size is None: # 为测试做准备，模型对输入LR的尺寸有要求
            h_lr = math.floor(T2_img.shape[-2] / s + 1e-9) # h_lr是整数
            w_lr = math.floor(T2_img.shape[-1] / s + 1e-9)

            ##### TODO 这是RCT的特异性设置：调整 h_lr 和 w_lr 使其满足整除要求 #####
            #h_lr = (h_lr // 16) * 16  # H和W都要即能被4整除又能被16整除 
            #w_lr = (w_lr // 16) * 16  
            #print("##############################################", h_lr) 

            h_hr = round(h_lr * s) # 如果s是整数，那么h_hr也是整数
            w_hr = round(w_lr * s)
            #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", h_hr)
            
            # 确保高分辨率裁剪尺寸不超过原图大小
            h_hr = min(h_hr, T2_img.shape[-2])
            w_hr = min(w_hr, T2_img.shape[-1])

            T2_crop_hr = T2_img[:, :h_hr, :w_hr] # assume round int
            T1_crop_hr = T1_img[:, :h_hr, :w_hr]
            T2_crop_lr = resize_fn(T2_crop_hr, (h_lr, w_lr))
            T1_crop_lr = resize_fn(T1_crop_hr, (h_lr, w_lr))
            #print("##########################################", T2_crop_lr.shape) # [3, 120, 112]
        elif self.inp_size == "K-space": #BUG 仅在测试阶段采用，未对RCT进行适配
            C,H,W = T2_img.shape
            h_lr = math.floor(T2_img.shape[-2] / s + 1e-9)
            w_lr = math.floor(T2_img.shape[-1] / s + 1e-9)

            ##### TODO 这是RCT的特异性设置：调整 h_lr 和 w_lr 使其满足整除要求 #####
            #h_lr = (h_lr // 16) * 16  
            #w_lr = (w_lr // 16) * 16 
            
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)

            h_hr = min(h_hr, T2_img.shape[-2])
            w_hr = min(w_hr, T2_img.shape[-1])

            T2_crop_hr = T2_img[:, :h_hr, :w_hr] # 因为h_hr尺寸接近T1_img原始尺寸，所以不必中心裁剪
            T1_crop_hr = T1_img[:, :h_hr, :w_hr]
            #TODO 中心裁剪（可能影响测试效果）
            #T2_crop_hr = T2_img[:,H//2-math.floor(h_hr/2):H//2+math.ceil(h_hr/2),W//2-math.floor(w_hr/2):W//2+math.ceil(w_hr/2)]
            #T1_crop_hr = T1_img[:,H//2-math.floor(h_hr/2):H//2+math.ceil(h_hr/2),W//2-math.floor(w_hr/2):W//2+math.ceil(w_hr/2)]



        else:
            # 训练/测试阶段裁剪inp_size尺度的patch
            w_lr = self.inp_size
            w_hr = round(w_lr * s) # 训练时 64 x 4 = 256 
            #print("##########################################", T2_img.shape) # [3, 369, 369]
            #print("##########################################", w_hr) # 384
            #TODO 训练时的HR补丁尺寸: [48x2, 48x3, 48x4, 48x6, 48x8, 48x12]
            x0 = random.randint(0, T2_img.shape[-2] - w_hr)
            y0 = random.randint(0, T2_img.shape[-1] - w_hr)

            ####### prepare inp #########
            T2_crop_hr = T2_img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            T2_crop_lr = resize_fn(T2_crop_hr, w_lr)
            ####### prepare ref #########
            T1_crop_hr = T1_img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            T1_crop_lr = resize_fn(T1_crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            ####### prepare inp #########
            T2_crop_lr = augment(T2_crop_lr)
            T2_crop_hr = augment(T2_crop_hr)
            ####### prepare ref #########
            T1_crop_lr = augment(T1_crop_lr)
            T1_crop_hr = augment(T1_crop_hr)

        ####### prepare inp #########
        T2_hr_coord, T2_hr_rgb = to_pixel_samples(T2_crop_hr.contiguous())

        ####### prepare ref #########
        _, T1_hr_rgb = to_pixel_samples(T1_crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(T2_hr_coord), self.sample_q, replace=False)
            ####### prepare inp #########
            T2_hr_coord = T2_hr_coord[sample_lst]
            T2_hr_rgb = T2_hr_rgb[sample_lst]
            ####### prepare ref #########
            T1_hr_rgb = T1_hr_rgb[sample_lst]

        ref_w = int(np.sqrt(T1_hr_rgb.shape[0]))
        ref_c = T1_hr_rgb.shape[1]
        #T1_ref_hr = T1_hr_rgb.view(ref_c, ref_w, ref_w)
        T1_ref_hr = T1_hr_rgb.contiguous().view(ref_c, ref_w, ref_w)
        ####### prepare inp #########
        T2_cell = torch.ones_like(T2_hr_coord)
        T2_cell[:, 0] *= 2 / T2_crop_hr.shape[-2]
        T2_cell[:, 1] *= 2 / T2_crop_hr.shape[-1]


        return { #TODO T2-Target；T1-Reference
            'inp': T2_crop_lr,
            'inp_hr_coord': T2_hr_coord,
            'inp_cell': T2_cell, 
            'ref': T1_crop_lr,
            'ref_hr': T1_ref_hr,
            'gt': T2_hr_rgb,
        }
########################################################################################################