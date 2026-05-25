import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from timm.utils import AverageMeter

import datasets
import models
import utils



### 批量预测函数
def batched_predict(model, inp, coord, cell, bsize):
    #print('############################################',coord.shape)  # 打印coord的形状
    #print('############################################',cell.shape)   # 打印cell的形状

    with torch.no_grad():
        model.gen_feat(inp) # 获取latent code
        n = coord.shape[1] # 总共查询n个位置坐标
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n) # 保证最后一个batch不超过n
            #TODO pred = model(inp, coord, cell)相当于调用了model.forward()
            #pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :]) # 查询RGB
            pred = model(inp, coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


### 绘图设置
if True:
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"
    import seaborn as sns

    #   Set figure parameters
    large = 24;
    med = 24;
    small = 24 
    sns_text_size = 4
    params = {'axes.titlesize': large,
              'legend.fontsize': med,
              'figure.figsize': (16, 10),
              'axes.labelsize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("white") 
    sns.set(font_scale=sns_text_size)
    # plt.rc('font', **{'family': 'Times New Roman'})
    plt.rcParams['axes.unicode_minus'] = False


def analyze_erf(source, dest="heatmap.png", ALGRITHOM=lambda x: np.power(x - 1, 0.25)):
    def heatmap(data, camp='RdYlGn', figsize=(10, 10), ax=None, save_path=None, cbar=False):
        plt.figure(figsize=figsize, dpi=40)
        ax = sns.heatmap(data,
                         xticklabels=False,
                         yticklabels=False, cmap=camp,
                         center=0, annot=False, ax=ax, cbar=cbar, annot_kws={"size": 24}, fmt='.2f') 
        if cbar: 
            ax.collections[0].set_clim(0,1) 
        plt.savefig(save_path)

    def analyze_erf(args):
        data = args.source
        print(np.max(data))
        print(np.min(data))
        data = args.ALGRITHOM(data + 1)  # the scores differ in magnitude. take the logarithm for better readability
        data = data / np.max(data)  # rescale to [0,1] for the comparability among models
        heatmap(data, save_path=args.heatmap_save)
        print('heatmap saved at ', args.heatmap_save)

    class Args():
        ...

    args = Args()
    args.source = source
    args.heatmap_save = dest
    args.ALGRITHOM = ALGRITHOM
    os.makedirs(os.path.dirname(args.heatmap_save), exist_ok=True)
    analyze_erf(args) 



# copied from https://github.com/DingXiaoH/RepLKNet-pytorch
def visualize_erf(MODEL: nn.Module = None, 
                  num_images=1000, 
                  loader=None,
                  data_norm=None,
                  resolution=None,
                  ):
    def get_input_grad(model, inp, coord, cell, resolution, data_norm): 
        # import pdb 
        # pdb.set_trace()
        t = data_norm['gt']
        gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
        gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
        
        h = resolution[0]
        w = resolution[1]

        ### 模型核心推理过程（是否批量预测）
        outputs = model(inp, coord, cell)
        #outputs = batched_predict(model, inp, coord, cell, eval_bsize=30000)
        outputs = (outputs * gt_div + gt_sub).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).unsqueeze(0)
        #print('##############################', outputs.shape) # [1, 97200, 3]=>[1, 3, 360, 270]
        out_size = outputs.size()
        central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
        grad = torch.autograd.grad(central_point, inp)
        grad = grad[0]
        grad = torch.nn.functional.relu(grad)
        aggregated = grad.sum((0, 1))
        grad_map = aggregated.cpu().numpy()
        return grad_map

    def main(args, MODEL: nn.Module = None):
        
        ### 直接传入数据加载器
        test_loader = args.loader
        data_norm = args.data_norm
        resolution = args.resolution
        
        if data_norm is None:
            data_norm = {
                'inp': {'sub': [0], 'div': [1]},
                'gt': {'sub': [0], 'div': [1]}
            }
        t = data_norm['inp']
        inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
        

        model = MODEL
        model.cuda() 
        model.eval()

        optimizer = torch.optim.SGD(model.parameters(), lr=0, weight_decay=0)

        meter = AverageMeter()
        optimizer.zero_grad()
        
        ### 遍历数据加载器
        for idx,data_sample in enumerate(test_loader):
            if meter.count == args.num_images:
                return meter.avg
            # 迁移数据到GPU上
            for k, v in data_sample.items():
                data_sample[k] = v.cuda(non_blocking=True)

            inp = (data_sample['inp'] - inp_sub) / inp_div  # [1, 3, 135, 180]
            coord = data_sample['coord']
            cell = data_sample['cell']

            #samples = data_sample[0] 
            _, _, H, W = inp.size()
            #samples = samples.type(torch.FloatTensor).cuda(non_blocking=True)
            inp.requires_grad = True
            optimizer.zero_grad()

            ### 包含模型推理过程
            contribution_scores = get_input_grad(model, inp, coord, cell, resolution, data_norm)
            torch.cuda.empty_cache()
            if np.isnan(np.sum(contribution_scores)):
                print('got NAN, next image')
                continue
            else:
                print(f'accumulat{idx}')
                meter.update(contribution_scores)

        return meter.avg


    class Args():
        ...

    args = Args()
    args.num_images = num_images
    args.loader = loader
    args.data_norm = data_norm
    args.resolution = resolution
    return main(args, MODEL)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    #parser.add_argument('--scale_max', default='4')
    parser.add_argument('--gpu', default='0')
    #parser.add_argument('--mcell', default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    ### 创建数据加载器
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)
    data_norm = config['data_norm']
    resolution = config['resolution'] 
    
    ### 创建模型
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    


    method = "EDSR-LTE"
    save_dir = "/home/caoxinyu/Arbitrary-scale/liif-main/results/ERF" 

    ### 创建ERF
    grad_map = visualize_erf(model, num_images=10, loader=loader, data_norm=data_norm, resolution=resolution)
    analyze_erf(source=grad_map, dest=os.path.join(save_dir, "ERF","{}_ERF.png".format(method)))