import torch

def ncc_loss(sr, hr):
    """
    计算归一化互相关（NCC）损失。

    参数:
    sr (torch.Tensor): 超分辨率图像（预测的图像）。
    hr (torch.Tensor): 高分辨率图像（真实的图像）。

    返回:
    torch.Tensor: NCC 损失，值在 0 到 2 之间。
    """
    # 将图像展平成一维向量
    sr_mean = torch.mean(sr)
    hr_mean = torch.mean(hr)
    
    # 去除均值
    sr_centered = sr - sr_mean
    hr_centered = hr - hr_mean
    
    # 计算内积（点积）
    numerator = torch.sum(sr_centered * hr_centered)
    
    # 计算标准差
    sr_std = torch.sqrt(torch.sum(sr_centered ** 2))
    hr_std = torch.sqrt(torch.sum(hr_centered ** 2))
    
    # 计算NCC
    ncc = numerator / (sr_std * hr_std + 1e-8)  # 避免除以零
    
    # 损失为 1 - NCC, 范围为 0 到 2
    loss = 1 - ncc
    
    return loss
