import random
import math
import numpy as np
import skimage.color as sc
import torch
from torchvision import transforms


# 获取切片
def get_patch(*args, patch_size=96, scale=1, scale2=1):
    ih, iw = args[0].shape[:2]  ## LR image

    tp = int(round(scale * patch_size))
    tp2 = int(round(scale2 * patch_size))
    ip = patch_size
    
    if scale==int(scale):
        step = 1
    elif (scale*2)== int(scale*2):
        step = 2
    elif (scale*5) == int(scale*5):
        step = 5
    else:
        step = 10


    if scale2==int(scale2):
        step2 = 1
    elif (scale2*2)== int(scale2*2):
        step2 = 2
    elif (scale2*5) == int(scale2*5):
        step2 = 5
    else:
        step2 = 10

    if (ih-ip)//step==0:
        iy = 0
    else:
        iy = random.randrange(0, (ih-ip)//step) * step
        
    if (iw-ip)//step==0:
        ix = 0
    else:
        ix = random.randrange(0, (iw-ip)//step2) * step2

    tx, ty = int(round(scale2 * ix)), int(round(scale * iy))

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp2, :] for a in args[1:]]
    ]

    return ret

# 做通道适配
def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2: # [H,W]->[H,W,1]
            img = np.expand_dims(img, axis=2)

        c = img.shape[2] # 图像通道数
        if n_channels == 1 and c == 3: # 3->1
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1: # 1->3
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

# Numpy->Tensor
def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1))) # [H,W,C]->[C,H,W]
        tensor = torch.from_numpy(np_transpose).float()
        #tensor.mul_(rgb_range / 255) # 归一化

        return tensor

    return [_np2Tensor(a) for a in args]

# 数据增强
def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img, rot=True):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot:
            if rot90: img = img.transpose(1, 0, 2)

        return img

    out = []

    if args[1].shape[0] == args[1].shape[1]:
        for arg in args:
            out.append(_augment(arg))
    else:
        for arg in args:
            out.append(_augment(arg, rot=False))

    return out

# 测试阶段数据预处理方法
def crop_border(img_hr, img_lr, img_ref_hr, img_ref_lr, scale, scale2):
        C, H_lr, W_lr = img_lr.size()
        C, H_hr, W_hr = img_hr.size()
        H = H_lr if round(H_lr * scale) <= H_hr else math.floor(H_hr / scale)
        W = W_lr if round(W_lr * scale2) <= W_hr else math.floor(W_hr / scale2)

        step = []
        for s in [scale, scale2]:
            if s == int(s):
                step.append(1)
            elif s * 2 == int(s * 2):
                step.append(2)
            elif s * 5 == int(s * 5):
                step.append(5)
            elif s * 10 == int(s * 10):
                step.append(10)
            elif s * 20 == int(s * 20):
                step.append(20)
            elif s * 50 == int(s * 50):
                step.append(50)

        H_new = H // step[0] * step[0]
        if H_new % 2 == 1:
            H_new = H // (step[0] * 2) * step[0] * 2

        W_new = W // step[1] * step[1]
        if W_new % 2 == 1:
            W_new = W // (step[1] * 2) * step[1] * 2

        img_lr = img_lr[:, :H_new, :W_new]
        img_hr = img_hr[:, :round(scale * H_new), :round(scale2 * W_new)]

        if img_ref_hr is not None:
            img_ref_hr = img_ref_hr[:, :round(scale * H_new), :round(scale2 * W_new)]
            img_ref_lr = img_ref_lr[:, :H_new, :W_new]
            
        return img_hr, img_lr, img_ref_hr, img_ref_lr
