import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

# 批量预测函数
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


#TODO eval_metrics()的原始版本，在train中有引用
def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, scale_max=4,
              verbose=False, mcell=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    # eval_type决定评价数据集以及评价指标（eg:div2k-4）
    if eval_type is None:
        metric_fn = utils.calc_psnr
    #TODO 根据eval_type的值决定输入metric_fn中的参数
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    elif eval_type.startswith('ssim'): # 通过改动eval_type计算SSIM
        #log_name = 'SSIM.txt'
        #scale = int(eval_type.split('-')[1])
        metric_fn = utils.ssim
    elif eval_type.startswith('mse'):
        metric_fn = utils.mse
    else:
        raise NotImplementedError


    #TODO 用于计算所有验证批次的平均评价指标
    val_res = utils.Averager()
    

    ################################# 测试遍历 #####################################
    pbar = tqdm(loader, leave=False, desc='val')
    count = 0 # 设置计数器
    for batch in pbar:
        count += 1
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch['inp'] - inp_sub) / inp_div # [1, 3, 135, 180]
        
        coord = batch['coord']
        cell = batch['cell']

        #TODO SRNO
        if mcell == False: c = 1
        else : c = max(scale/scale_max, 1)

        
        # eval_bsize决定是否批量预测
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell*c)
        else:
            pred = batched_predict(model, inp, coord, cell*c, eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        # 为评价指标的计算reshape pred和batch['gt']
        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
        
        res = metric_fn(pred, batch['gt'])
        
        val_res.add(res.item(), inp.shape[0])
      
        if verbose:
            pbar.set_description('val {:.4f}'.format(res.item()))

    return val_res.item()


#TODO 这里定义计算PSNR SSIM MSE的方法
def eval_metrics(loader, model, data_norm=None, eval_type=None, eval_bsize=None, scale_max=4,
              verbose=False, mcell=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    # eval_type决定评价数据集以及评价指标（eg:div2k-4）
    if eval_type is None:
        metric_fn = utils.calc_psnr
    #TODO 根据eval_type的值决定输入metric_fn中的参数
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    elif eval_type.startswith('ssim'): # 通过改动eval_type计算SSIM
        #log_name = 'SSIM.txt'
        #scale = int(eval_type.split('-')[1])
        metric_fn = utils.ssim
    elif eval_type.startswith('mse'):
        metric_fn = utils.mse
    #TODO ['metrics-2', 'metrics-3', 'metrics-4'......] 计算所有三个指标
    elif eval_type.startswith('metrics'):
        scale = int(eval_type.split('-')[1])
        psnr_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
        ssim_fn = utils.ssim
        mse_fn = utils.mse
    else:
        raise NotImplementedError


    #TODO 用于计算所有验证批次的平均评价指标
    val_res_psnr = utils.Averager()
    val_res_ssim = utils.Averager()
    val_res_mse = utils.Averager()

    ################################# 测试遍历 #####################################
    pbar = tqdm(loader, leave=False, desc='val')
    count = 0 # 设置计数器
    for batch in pbar:
        count += 1
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch['inp'] - inp_sub) / inp_div # [1, 3, 135, 180]
        
        coord = batch['coord']
        cell = batch['cell']

        #TODO SRNO
        if mcell == False: c = 1
        else : c = max(scale/scale_max, 1)

        
        # eval_bsize决定是否批量预测
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell*c)
        else:
            pred = batched_predict(model, inp, coord, cell*c, eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        # 为评价指标的计算reshape pred和batch['gt']
        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
        
        psnr_res = psnr_fn(pred, batch['gt'])
        ssim_res = ssim_fn(pred, batch['gt'])
        mse_res = mse_fn(pred, batch['gt'])
        val_res_psnr.add(psnr_res.item(), inp.shape[0])
        val_res_ssim.add(ssim_res.item(), inp.shape[0])
        val_res_mse.add(mse_res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(psnr_res.item()))

    return val_res_psnr.item(), val_res_ssim.item(), val_res_mse.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--mcell', default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    

    import time
    t1= time.time()
    #TODO 这里的输出的res是一个元组，res[0]=PSNR; res[1]=SSIM; res[2]=RMSE
    res = eval_metrics(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        scale_max = int(args.scale_max),
        verbose=True,
        mcell=bool(args.mcell)
        )
    t2 =time.time()
    # '#######################[div2k-x2]#######################'在测试脚本中设置的，四舍五入保留后四位
    print('$$$$$$$$$$$$$$$$$$$$$[result]$$$$$$$$$$$$$$$$$$$$$: PSNR: {:.4f}   SSIM: {:.4f}   MSE: {:.9f}'.format(res[0], res[1], res[2]), utils.time_text(t2-t1))
