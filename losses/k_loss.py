import torch
import torch.nn as nn
import math

#TODO 适用于MRI图像（两通道分别表示复数实部和虚部）
class MRI_KLoss(nn.Module):
    def __init__(self, args):
        super(MRI_KLoss, self).__init__()

    def forward(self, sr, hr, shape1, shape2, FSsr=None):
        b, c, h, w = hr.shape
        # 将两个通道维转化为复数
        hr_comp = hr[:, 0:1, :, :] + 1j * hr[:, 1:2, :, :]

        if FSsr is None:
            sr_comp = sr[:, 0:1, :, :] + 1j * sr[:, 1:2, :, :]
            FSsr = 1 / math.sqrt(h * w) * torch.fft.fftn(sr_comp, dim=[2, 3])
            FSsr = torch.fft.fftshift(FSsr, dim=[2, 3])
        
        FShr = 1 / math.sqrt(h * w) * torch.fft.fftn(hr_comp, dim=[2, 3])
        FShr = torch.fft.fftshift(FShr, dim=[2, 3])
        
        mask = torch.ones_like(FSsr)
        mask[:, :, h // 2 - math.floor(shape1 / 2):h // 2 + math.ceil(shape1 / 2), w // 2 - math.floor(shape2 / 2):w // 2 + math.ceil(shape2 / 2)] = 0        

        loss = torch.mean(torch.abs((FSsr - FShr) * mask))      
        return loss

# if __name__ == "__main__":
#     model_test = KLoss(0)
#     # 存在MRI图像像素值为复数，其实部与虚部对应两个通道维
#     a = torch.randn([5, 2, 20, 20])
#     b = torch.randn([5, 2, 20, 20])
#     shape1 = 5
#     shape2 = 5
#     loss = model_test(a, b, shape1, shape2)
#     print(loss)



class RGB_KLoss(nn.Module):
    def __init__(self, args):
        super(RGB_KLoss, self).__init__()

    def forward(self, sr, hr, shape1, shape2, FSsr=None):
        b, c, h, w = hr.shape
        
        if FSsr is None:
            FSsr = []
            for i in range(c):
                sr_channel = sr[:, i:i+1, :, :]
                FSsr_channel = 1 / math.sqrt(h * w) * torch.fft.fftn(sr_channel, dim=[2, 3])
                FSsr_channel = torch.fft.fftshift(FSsr_channel, dim=[2, 3])
                FSsr.append(FSsr_channel)
            FSsr = torch.stack(FSsr, dim=1)
        
        FShr = []
        for i in range(c):
            hr_channel = hr[:, i:i+1, :, :]
            FShr_channel = 1 / math.sqrt(h * w) * torch.fft.fftn(hr_channel, dim=[2, 3])
            FShr_channel = torch.fft.fftshift(FShr_channel, dim=[2, 3])
            FShr.append(FShr_channel)
        FShr = torch.stack(FShr, dim=1)
        
        mask = torch.ones_like(FSsr)
        mask[:, :, h // 2 - math.floor(shape1 / 2):h // 2 + math.ceil(shape1 / 2), w // 2 - math.floor(shape2 / 2):w // 2 + math.ceil(shape2 / 2)] = 0        

        loss = torch.mean(torch.abs((FSsr - FShr) * mask))      
        return loss

if __name__ == "__main__":
    model_test = RGB_KLoss(0)
    a = torch.randn([5, 3, 20, 20])  # RGB三通道图像
    b = torch.randn([5, 3, 20, 20])  # RGB三通道图像
    shape1 = 5
    shape2 = 5
    loss = model_test(a, b, shape1, shape2)
    print(loss)
