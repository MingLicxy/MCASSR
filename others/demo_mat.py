import argparse
import os
import math
from PIL import Image
import scipy.io as sio
import numpy as np

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict

### 方案一：这里的实现逻辑是HR尺度不变，下采样HR获取不同尺度对应的LR作为输入，目的是恢复到HR分辨率
### 方案二：另一种逻辑是输入相同LR（可以保证其尺度被16整除）将其转换为[2,H,W]张量（虚部为0）进行输入，目标HR尺度由--resolution控制
### 方案三：利用K-space裁剪下采样获取指定尺度的LR（可以保证其尺度被16整除）得到[2,H,W]张量（虚部不一定为零），目标HR尺度由--resolution控制

#BUG K-space裁剪下采样（保留低频去除高频）
def fft2c(x):
    """
    接收复数类型数组输入
    x is a complex shapes [H,W,C]
    """
    S = x.shape
    x.reshape(S[0],S[1],-1)
    res = 1 / math.sqrt(S[0]*S[1]) * np.fft.fftshift(np.fft.fft2(x,axes=[0,1]),axes=[0,1])
    return res

def ifft2c(x):
    """
    x is a complex shapes [H,W,C]
    """
    S = x.shape
    x.reshape(S[0],S[1],-1)
    x = np.fft.ifftshift(x,axes=[0,1])
    res = math.sqrt(S[0]*S[1]) * np.fft.ifft2(x,axes=[0,1])
    return res

def kspace_crop(x, s):
    """
    input:
        接受复数类型数组输入
        x: input HR complex matrix [H,W]
        s: 下采样因子(可能非整数)
        
    output:
        x_hr: [H,W,2] 浮点数类型
        x_lr: [H/s,W/s,2]
    """
    #TODO 对 x 进行被24整除的中心裁剪，为s=[2,3,4,6,8,12]做准备
    h_hr,w_hr = x.shape
    H_hr = round(math.floor(h_hr / 24) * 24)
    W_hr = round(math.floor(w_hr / 24) * 24)
    x_crop_hr = x[h_hr//2-math.floor(H_hr/2):h_hr//2+math.ceil(H_hr/2),w_hr//2-math.floor(W_hr/2):w_hr//2+math.ceil(W_hr/2)]
    


    H_lr = math.floor(H_hr / s + 1e-9) # h_lr是整数
    W_lr = math.floor(W_hr / s + 1e-9)

    ##### TODO 这是RCT的特异性设置：调整 h_lr 和 w_lr 使其满足整除要求 #####
    # 这里没有考虑RCT的整除限制
    # H_lr = (h_lr // 16) * 16  
    # W_lr = (w_lr // 16) * 16 
 

    fs_crop_hr = fft2c(x_crop_hr) # 转换到频域（形状不变） 
    H_c,W_c = H_hr//2,W_hr//2 # x_crop_hr的中心坐标

    #TODO K-space裁剪下采样（中心裁剪）
    fs_crop_lr = fs_crop_hr[H_c-math.floor(H_lr/2):H_c+math.ceil(H_lr/2),W_c-math.floor(W_lr/2):W_c+math.ceil(W_lr/2)]
    x_hr = x_crop_hr # x是复数类型，x_hr也是复数类型
    x_lr = ifft2c(fs_crop_lr) # 逆傅里叶变换输出复数类型


    x_hr_real = x_hr.real
    x_hr_real = x_hr_real[ :, :,np.newaxis] # 增加一个维度 [H,W,1]
    x_hr_imag = x_hr.imag
    x_hr_imag = x_hr_imag[ :, :,np.newaxis]
    x_hr = np.concatenate((x_hr_real,x_hr_imag),2) # [H,W,2] 浮点数类型

    x_lr_real = x_lr.real
    x_lr_real = x_lr_real[ :, :,np.newaxis]
    x_lr_imag = x_lr.imag
    x_lr_imag = x_lr_imag[ :, :,np.newaxis]
    x_lr = np.concatenate((x_lr_real,x_lr_imag),2)


    # 标准化处理
    if np.max(x_hr)!=0:
        # math.sqrt(scale[0]*scale[1])用于补偿频域操作可能导致的像素值变化
        x_lr = x_lr/(s*np.max(x_hr)) 
        x_hr = x_hr/np.max(x_hr)

    # 将其转化为[2,H,w]的Tensor
    x_hr = np.ascontiguousarray(x_hr.transpose(2, 0, 1))
    x_hr = torch.from_numpy(x_hr).float()
    x_lr = np.ascontiguousarray(x_lr.transpose(2, 0, 1))
    x_lr = torch.from_numpy(x_lr).float()

    return x_hr, x_lr # [2,H,W]的Tensor

################## 负责.mat数据的去归一化与可视化 ##################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    # parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #TODO 定义上采样尺度
    scale = 2 # [2,3,4,6,8,12]

    #TODO 直接输入原始尺寸的.mat文件
    img = sio.loadmat(args.input)['dcm'] 
    # 对其进行指定尺度的K-space裁剪下采样，再输入模型
    img_hr, img_lr = kspace_crop(img, scale)

    # 加载预训练好的模型
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    
    # h,w是output分辨率（在做可视化时h,w一般不变，变的是输入图像的尺寸）
    # h, w = list(map(int, args.resolution.split(',')))
    h = img_hr.shape(-2)
    w = img_hr.shape(-1)

    # coord网格中心点坐标维度[h,w,2],坐标范围[-1,1]=>cell像素尺寸[2/h,2/w]
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    pred = batched_predict(model, ((img_lr - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 2).permute(2, 0, 1).cpu() # [2,h,w] Tensor 通道维上分别是复数的实部与虚部
    
    # 获取实部虚部
    pred_real = pred[0]
    pred_imag = pred[1]
    # 计算幅值
    magnitude = torch.sqrt(pred_real**2 + pred_imag**2)
    # 归一化幅值
    magnitude_normalized = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    # 保存可视化预测结果
    transforms.ToPILImage()(magnitude_normalized).save(args.output)
