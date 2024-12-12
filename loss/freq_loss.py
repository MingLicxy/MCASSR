import torch
import torch.nn.functional as F
#import pywt
import pytorch_wavelets as pw
from torchvision.transforms import functional as trans_fn


# 傅里叶损失（对复数表示的实部和虚部分别做损失）
def fft_mse_loss(img1, img2):
    img1_fft = torch.fft.fftn(img1, dim=(2,3),norm="ortho")
    img2_fft = torch.fft.fftn(img2, dim=(2,3),norm="ortho")
    # Splitting x and y into real and imaginary parts
    x_real, x_imag = torch.real(img1_fft), torch.imag(img1_fft)
    y_real, y_imag = torch.real(img2_fft), torch.imag(img2_fft)
    # Calculate the MSE between the real and imaginary parts separately
    #mse_real = torch.nn.MSELoss()(x_real, y_real)
    #mse_imag = torch.nn.MSELoss()(x_imag, y_imag)
    mse_real = F.mse_loss(x_real, y_real)
    mse_imag = F.mse_loss(x_imag, y_imag)
    return mse_imag+mse_real








    
# 定义相位（phase）振幅（amplitude）损失   
def pha_amp_loss(img1, img2, alpha=0.01 ,beta=0.01):
    img1_fft = torch.fft.rfft2(img1, norm='backward')
    img2_fft = torch.fft.rfft2(img2, norm='backward')
    # 分别计算x和y的幅度和相位
    x_pha, x_amp = torch.angle(img1_fft), torch.abs(img1_fft)
    y_pha, y_amp = torch.angle(img2_fft), torch.abs(img2_fft)
    #l1_pha = torch.nn.L1Loss()(x_pha, y_pha)
    #l1_amp = torch.nn.L1Loss()(x_amp, y_amp)
    l1_pha = F.l1_loss(x_pha, y_pha)
    l1_amp = F.l1_loss(x_amp, y_amp)
    return alpha*l1_pha + beta*l1_amp


# 小波域损失
def dwt_mse_loss(x, y ,J=4):
    # Perform 4-level 2D discrete wavelet transform on both images
    x_dwt_f = pw.DWTForward(J=J, wave='haar', mode='symmetric')
    y_dwt_f = pw.DWTForward(J=J, wave='haar', mode='symmetric')
    x_dwt_f.cuda()
    y_dwt_f.cuda()
    x_dwt=x_dwt_f(x)[1]
    y_dwt=y_dwt_f(y)[1]
    h_mse,v_mse,d_mse=0,0,0
    for i in range(J):
        # Calculate MSE between the coefficients of each subband
        h_mse += F.mse_loss(x_dwt[i][:,:,0,:,:], y_dwt[i][:,:,0,:,:])
        v_mse += F.mse_loss(x_dwt[i][:,:,1,:,:], y_dwt[i][:,:,1,:,:])
        d_mse += F.mse_loss(x_dwt[i][:,:,2,:,:], y_dwt[i][:,:,2,:,:])

    # Sum the MSE losses across subbands and return
    return h_mse + v_mse + d_mse

#TODO 定义循环一致损失（这个损失是否有用？有无必要在频域上使用？）
# 循环一致损失有没有用的关键在于：其下采样过程也应该是可学习的
def consistency_loss(lr_img, sr_img, scale_factor=4):
   
    # 对SR进行下采样（size设置为LR的HxW）
    downsampled_sr = trans_fn.resize(sr_img, size=lr_img.shape[-2:], 
                                     interpolation=trans_fn.InterpolationMode.BICUBIC)
    # 计算下采样损失
    loss = F.mse_loss(downsampled_sr, lr_img)
    return loss

# Charbonnier Loss(平滑的L1损失)
def charbonnier_loss(x, y, epsilon=1e-6):
    diff = x - y
    loss = torch.sqrt(diff * diff + epsilon * epsilon)
    return torch.mean(loss)


####################################### 定义组合损失 #####################################

def loss_comb1(lr_img, sr_img, hr_img , alpha=0.2 ,beta=0.1):
    # Calculation of MSE loss in the frequency domain
    loss_fft = fft_mse_loss(sr_img, hr_img)
    # Calculating multilevel discrete wavelet transform MSE losses
    loss_dwt = dwt_mse_loss(sr_img, hr_img)
    return alpha*loss_fft + beta*loss_dwt


def loss_comb2(lr_img, sr_img, hr_img):
    # 对空域分支输出做空域L1Loss
    loss_spa = F.l1_loss(sr_img['img_out'], hr_img)
    # 对频域分支输出做空域L1Loss
    loss_fre = F.l1_loss(sr_img['img_fre'], hr_img)
    # 对频域分支输出做相位幅值损失
    loss_pha_amp = pha_amp_loss(sr_img['img_fre'], hr_img)
    # 对空域分支输出做循环一致损失（consistency_lost可以与空域损失L1Loss做一个参数加权）
    loss_con = consistency_loss(lr_img, sr_img['img_out'])
    return loss_spa + loss_fre + loss_pha_amp + loss_con
    #return loss_spa + loss_fre + loss_pha_amp 

