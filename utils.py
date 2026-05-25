import os
import time
import shutil
import math
from math import log10
import random
import cv2
import torch

import numpy as np
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from skimage.metrics import structural_similarity as ssimcalcu

# 设置随机种子方便实验复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # sets the seed for cpu
    torch.cuda.manual_seed(seed) # Sets the seed for the current GPU.
    torch.cuda.manual_seed_all(seed) #  Sets the seed for the all GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False
    torch.set_deterministic(True)


# 通过定义和选择不同的随机数生成器初始化函数，确保在训练时的随机性和验证/测试时的可重复性
def numpy_random_init(worker_id):
    process_seed = torch.initial_seed()
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([worker_id, base_seed])
    np.random.seed(ss.generate_state(4))

def numpy_fix_init(worker_id):
    np.random.seed(2 << 16 + worker_id)

numpy_init_dict = {"train": numpy_random_init, "val": numpy_fix_init, "test": numpy_fix_init}


#TODO SRNO使用 显示特征图
def show_feature_map(feature_map,layer,name='rgb',rgb=False):
    feature_map = feature_map.squeeze(0)
    #if rgb: feature_map = feature_map.permute(1,2,0)*0.5+0.5
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = math.ceil(np.sqrt(feature_map_num))
    if rgb:
        #plt.figure()
        #plt.imshow(feature_map)
        #plt.axis('off')
        feature_map = cv2.cvtColor(feature_map,cv2.COLOR_BGR2RGB)
        cv2.imwrite('data/'+layer+'/'+name+".png",feature_map*255)
        #plt.show()
    else:
        plt.figure()
        for index in range(1, feature_map_num+1):
            t = (feature_map[index-1]*255).astype(np.uint8)
            t = cv2.applyColorMap(t, cv2.COLORMAP_TWILIGHT)
            plt.subplot(row_num, row_num, index)
            plt.imshow(t, cmap='gray')
            plt.axis('off')
            #ensure_path('data/'+layer)
            cv2.imwrite('data/'+layer+'/'+str(name)+'_'+str(index)+".png",t)
        #plt.show()
        plt.savefig('data/'+layer+'/'+str(name)+".png")


# downsample()子函数
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

#TODO SRNO使用 下采样
def downsample(img, scale_min=1, scale_max=4, inp_size=None, augment=False, epoch=None):
    if epoch<20: s = random.randint(scale_min, scale_max)
    s = random.uniform(scale_min, scale_max)
    #print(s)
    
    if inp_size is None:
        h_lr = math.floor(img.shape[-2] / s + 1e-9)
        w_lr = math.floor(img.shape[-1] / s + 1e-9)
        h_hr = round(h_lr * s)
        w_hr = round(w_lr * s)
        img = img[:, :, :h_hr, :w_hr]
        img_down = torch.stack([resize_fn(x, (h_lr, w_lr)) for x in img], dim=0) 
        crop_lr, crop_hr = img_down, img
    else:
        h_lr = inp_size
        w_lr = inp_size
        h_hr = round(h_lr * s)
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img.shape[-2] - w_hr)
        y0 = random.randint(0, img.shape[-1] - w_hr)
        crop_hr = img[:, :, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr = torch.stack([resize_fn(x, w_lr) for x in crop_hr], dim=0)
    
    if augment == True:
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        dflip = random.random() < 0.5

        def augment(x):
            if hflip: x = x.flip(-2)
            if vflip: x = x.flip(-1)
            if dflip: x = x.transpose(-2, -1)
            return x
        crop_lr = augment(crop_lr)
        crop_hr = augment(crop_hr)

    coord = make_coord([h_hr, w_hr], flatten=False)
    coord = coord.unsqueeze(0).expand(img.shape[0], *coord.shape[:2], 2)

    cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32).unsqueeze(0).expand(img.shape[0], 2)
    return {
        'inp': crop_lr.contiguous(),
        'coord': coord.contiguous(),
        'cell': cell.contiguous(),
        'gt': crop_hr.contiguous()
    }    

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path

# 将日志在控制台上打印并写入文件
def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

# 创建优化器
def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer

# 生成网格中心的坐标（坐标原点在中间像素的中心点，坐标范围默认[-1,1]）
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten: # 默认为true
        ret = ret.view(-1, ret.shape[-1])
    return ret

# 将图像转换为网格坐标-RGB值的成对数据
def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb

def to_pixel_samples_mat(img):
    """ Convert the image to coord-mat pairs.
        img: Tensor, (2, H, W)
    """
    coord = make_coord(img.shape[-2:])
    mat = img.view(2, -1).permute(1, 0)
    return coord, mat

#TODO COZ使用 计算坐标网格偏置
def make_coord_bias(shape,whr ,biasx,biasy,ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * whr)
        if i == 0:
            seq = v0 + r + (2 * r) * (torch.arange(n).float() + biasy)
        else:
            seq = v0 + r + (2 * r) * (torch.arange(n).float() + biasx)            
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1) 
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples_bias(img,whr,biasx,biasy):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord_bias(img.shape[-2:],whr,biasx,biasy)
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


# 计算PSNR
def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        # benchmark在YCbCr空间中的Y通道上计算PSNR
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        # div2k在RGB空间中计算PSNR
        elif dataset == 'div2k':
            # 裁剪边缘像素防止插值造成的边缘伪影影响PSNR的计算
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)

# 计算SSIM
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])

    return gauss / gauss.sum()


# 计算LPIPS
import pyiqa
loss_fn = pyiqa.create_metric('lpips', use_gpu=True)
def lpips(sr, hr):
    lpips_value = loss_fn(sr, hr)
    return lpips_value

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    C1 = 0.01**2
    C2 = 0.03**2

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    ssim_map = ( (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) ) / ( (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    bs, ch, h, w = img1.shape

    window = create_window(window_size, ch)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, ch, size_average)


#TODO 计算两张图像的RMSE
def rmse(img1, img2):
    # 如果img1和img2是torch.Tensor，则将它们转换为 numpy 数组
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    mse = np.mean((img1 - img2) ** 2)
    return np.sqrt(mse)

def mse(img1, img2):
    # 如果img1和img2是torch.Tensor，则将它们转换为 numpy 数组
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    res = np.mean((img1 - img2) ** 2)
    return res



# calc_psnr()中用于保存图像
def save_fig(x, y, pred, fig_name, srresult):
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray)
        ax[0].set_title('LR', fontsize=30)
       
        ax[1].imshow(pred, cmap=plt.cm.gray)
        ax[1].set_title('SR', fontsize=30)
        ax[1].set_xlabel("PSNR:{:.4f}\nSSIM:{:.4f}\nMSE:{:.4f}".format(srresult[0],srresult[1],srresult[2]),fontsize=20)

        ax[2].imshow(y, cmap=plt.cm.gray)
        ax[2].set_title('HR', fontsize=30)
        f.savefig(fig_name)
        plt.close()

#TODO 配合K-space裁剪下采样的指标计算方法以及可视化结果保存（指标计算以及结果可视化都是利用幅值）
def calc_save(lr, sr, hr, FSsr=None, img_name=None, save=False, scale=1, savefile=None, ref=None):
    # 明确一点：数据处于空间域还是频域与其数据类型无关，复数类型数据也可以处于空间域
    if FSsr is not None: #TODO FSsr表示频域sr，需要通过逆傅里叶变换转换到空间域计算幅值
        FSsr = torch.fft.ifftshift((FSsr), dim=[2,3])
        sr = math.sqrt(FSsr.shape[2]*FSsr.shape[3]) * (torch.fft.ifftn(FSsr, dim=[2,3]))
        srmagnitude = torch.abs(sr) 
    else:
        # 直接在空间域计算幅值
        srmagnitude = (sr[:, 0:1, :, :] ** 2 + sr[:, 1:2, :, :] ** 2).sqrt()

    lrmagnitude = (lr[:, 0:1, :, :] ** 2 + lr[:, 1:2, :, :] ** 2).sqrt()
    hrmagnitude = (hr[:, 0:1, :, :] ** 2 + hr[:, 1:2, :, :] ** 2).sqrt()
    lrcpu = lrmagnitude[0,0,:,:].cpu().numpy()
    hrcpu = hrmagnitude[0,0,:,:].cpu().numpy()
    srcpu = srmagnitude[0,0,:,:].cpu().numpy()
    if ref is not None:
        refmagnitude = (ref[:, 0:1, :, :] ** 2 + ref[:, 1:2, :, :] ** 2).sqrt()
        refcpu = refmagnitude[0,0,:,:].cpu().numpy()

    peak_signal = (hrmagnitude.max()-hrmagnitude.min()).item()
    mse = (srmagnitude - hrmagnitude).pow(2).mean().item()
    errormap = torch.abs(srmagnitude - hrmagnitude).cpu().numpy()
    errormap = errormap[0,0,:,:]
    psnr = 10*log10(peak_signal**2/mse)
    # 数据范围[0,1]
    ssim = ssimcalcu(srcpu,hrcpu,data_range=1.0)

    # 保存可视化值逻辑
    if save:    
        pthroot = os.path.join('./savefigresult','{:s}'.format(savefile), 'x{:.1f}_{:.1f}'.format(scale[0],scale[1]))
        if not os.path.exists(pthroot):
            os.makedirs(pthroot)
        img_path = os.path.join(pthroot, 'results_{:s}.png'.format(img_name))
        srresult = [psnr,ssim,mse]

        #BUG 这里*255不一定说明.mat数据中像素值范围是[0,255]，只是现在结果要保存为.png所以*255
        save_fig(lrcpu*255, hrcpu*255, srcpu*255, img_path, srresult)
    return psnr,ssim,mse
