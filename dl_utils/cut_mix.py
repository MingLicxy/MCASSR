import torch
import random

#TODO cutMix数据增强：函数的输出是一个大小为低分辨率图像的尺寸,但其中包含了位置和尺寸随机的高分辨率图像的区域
def cutMix(imageLR, imageHR):
    """
    cut mix operation for img super-resolution
    :param imageLR:
    :param imageHR:
    :return: image with size 256(LR) which have a patch of img512(HR)
    """
    N, C, H, W = imageLR.shape
    cN, cC, cH, cW = imageHR.shape
    ratio = random.random() * 0.5 + 0.5  # patch range in 0.3-0.6
    rH, rW = random.randint(0, H - int(ratio * H)), random.randint(0, W - int(ratio * W))
    ds_imageHR = torch.nn.functional.interpolate(imageHR, (H, W))
    ret = imageLR.clone()
    mix0 = random.random()
    mix1 = random.random()
    mix2 = random.random()
    if mix0 < 0.5:
        ret[:, 0, rH:rH + int(ratio * H), rW: rW + int(ratio * W)] = ds_imageHR[:, 0, rH: rH + int(ratio * H),
                                                                     rW: rW + int(ratio * W)]
    if mix1 < 0.5:
        ret[:, 1, rH:rH + int(ratio * H), rW: rW + int(ratio * W)] = ds_imageHR[:, 1, rH: rH + int(ratio * H),
                                                                     rW: rW + int(ratio * W)]
    if mix2 < 0.5:
        ret[:, 2, rH:rH + int(ratio * H), rW: rW + int(ratio * W)] = ds_imageHR[:, 2, rH: rH + int(ratio * H),
                                                                     rW: rW + int(ratio * W)]
    return ret