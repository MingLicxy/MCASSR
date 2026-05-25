"""
The code come from https://github.com/XLearning-SCU/2023-CVPR-FPL/blob/main/codes/utils/util.py
从图像像素长尾分布的角度提出更关注高频细节的损失函数
"""
import torch
import functools
import torch.nn.functional as F

class focal_pixel_learning(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_sp, self.gamma_sp = 1, 0.5
        self.alpha_lp, self.gamma_lp = 1, 1
        self.upscale_func = functools.partial(
            F.interpolate, mode='bicubic', align_corners=False)
        self.weig_func = lambda x, y, z: torch.exp((x-x.min()) / (x.max()-x.min()) * y) * z

    def forward(self, x, hr, lr):
        f_BI_x = self.upscale_func(lr, size=hr.size()[2:])

        y_sp = torch.abs(hr - f_BI_x)
        w_y_sp = self.weig_func(y_sp, self.alpha_sp, self.gamma_sp).detach()

        y_lp = torch.abs(hr - f_BI_x - x)
        w_y_lp = self.weig_func(y_lp, self.alpha_lp, self.gamma_lp).detach()

        y_hat = hr - f_BI_x
        loss = torch.mean(w_y_sp * w_y_lp * torch.abs(x - y_hat))

        return loss