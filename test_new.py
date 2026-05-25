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

# FLOPs计算库
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# -------------------- 批量预测 --------------------
def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        left = 0
        preds = []
        while left < n:
            right = min(left + bsize, n)
            pred = model(inp, coord[:, left:right, :], cell[:, left:right, :])
            preds.append(pred)
            left = right
        pred = torch.cat(preds, dim=1)
    return pred

# -------------------- 测试指标函数 --------------------
def eval_metrics(loader, model, data_norm=None, eval_type=None, eval_bsize=None, scale_max=4,
                 verbose=False, mcell=False):
    model.eval()
    if data_norm is None:
        data_norm = {'inp': {'sub': [0], 'div': [1]}, 'gt': {'sub': [0], 'div': [1]}}
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    # 选择评价指标
    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    elif eval_type.startswith('ssim'):
        metric_fn = utils.ssim
    elif eval_type.startswith('mse'):
        metric_fn = utils.mse
    elif eval_type.startswith('metrics'):
        scale = int(eval_type.split('-')[1])
        psnr_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
        ssim_fn = utils.ssim
        mse_fn = utils.mse
    else:
        raise NotImplementedError

    val_res_psnr = utils.Averager()
    val_res_ssim = utils.Averager()
    val_res_mse = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)
        inp = (batch['inp'] - inp_sub) / inp_div
        coord = batch['coord']
        cell = batch['cell']
        if mcell == False:
            c = 1
        else:
            c = max(scale / scale_max, 1)

        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell * c)
        else:
            pred = batched_predict(model, inp, coord, cell * c, eval_bsize)

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None:
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape).permute(0, 3, 1, 2).contiguous()

        psnr_res = psnr_fn(pred, batch['gt'])
        ssim_res = ssim_fn(pred, batch['gt'])
        mse_res = mse_fn(pred, batch['gt'])
        val_res_psnr.add(psnr_res.item(), inp.shape[0])
        val_res_ssim.add(ssim_res.item(), inp.shape[0])
        val_res_mse.add(mse_res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(psnr_res.item()))

    return val_res_psnr.item(), val_res_ssim.item(), val_res_mse.item()
def count_params(module):
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total_params, trainable_params
# -------------------- 主函数 --------------------
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

    # 数据加载
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=8, pin_memory=True)


   # -------------------- 模型加载 --------------------
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    model.eval()

    # -------------------- Dummy 输入 --------------------
    B, C, H, W = 1, 3, 128, 128
    dummy_inp = torch.randn(B, C, H, W).cuda()
    dummy_coord = torch.zeros(B, H*W, 2).cuda()
    dummy_cell = torch.zeros(B, H*W, 2).cuda()

    # -------------------- 整体参数量 --------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {total_params/1e6:.2f} M")
    print(f"Trainable Params: {trainable_params/1e6:.2f} M")


    
    # -------------------- 模块化参数量 --------------------
    if hasattr(model, 'encoder'):
        total, trainable = count_params(model.encoder)
        print("Encoder Parameters:")
        print(f"  Total Params: {total/1e6:.2f} M")
        print(f"  Trainable Params: {trainable/1e6:.2f} M")

    if hasattr(model, 'cs_attn') and model.non_local_attn:
        total, trainable = count_params(model.cs_attn)
        print("CrossScaleAttention Parameters:")
        print(f"  Total Params: {total/1e6:.2f} M")
        print(f"  Trainable Params: {trainable/1e6:.2f} M")

    if hasattr(model, 'imnet_q'):
        total, trainable = count_params(model.imnet_q)
        print("imnet_q Parameters:")
        print(f"  Total Params: {total/1e6:.2f} M")
        print(f"  Trainable Params: {trainable/1e6:.2f} M")

    if hasattr(model, 'imnet_k'):
        total, trainable = count_params(model.imnet_k)
        print("imnet_k Parameters:")
        print(f"  Total Params: {total/1e6:.2f} M")
        print(f"  Trainable Params: {trainable/1e6:.2f} M")

    if hasattr(model, 'imnet_v'):
        total, trainable = count_params(model.imnet_v)
        print("imnet_v Parameters:")
        print(f"  Total Params: {total/1e6:.2f} M")
        print(f"  Trainable Params: {trainable/1e6:.2f} M")



    # -------------------- 整体 FLOPs --------------------
    flops = FlopCountAnalysis(model, (dummy_inp, dummy_coord, dummy_cell))
    print(f"===== Model FLOPs =====\nTotal FLOPs: {flops.total() / 1e9:.2f} G")

    # -------------------- 模块化 FLOPs --------------------
    # Encoder FLOPs
    if hasattr(model, 'encoder'):
        feat = model.encoder(dummy_inp)
        flops_enc = FlopCountAnalysis(model.encoder, dummy_inp)
        print(f"Encoder FLOPs: {flops_enc.total()/1e9:.2f} G")

    # CrossScaleAttention FLOPs
    if hasattr(model, 'cs_attn') and model.non_local_attn:
        B, C, H, W = 1, 64, 48, 48
        dummy_cs_inp = torch.randn(B, C, H, W).cuda()
        flops_cs = FlopCountAnalysis(model.cs_attn, dummy_cs_inp)
        print(f"CrossScaleAttention FLOPs: {flops_cs.total()/1e9:.2f} G")

    # imnet_q/k/v FLOPs (模拟 query 输入)
    # 注意这里的输入需要与实际 query 输入维度对应
    if hasattr(model, 'imnet_q'):
        bs, C_feat, H_feat, W_feat = feat.shape
        dummy_q_in = torch.randn(bs*H_feat*W_feat, 640).cuda()
        dummy_k_in = torch.randn(bs*H_feat*W_feat, 580).cuda()
        dummy_v_in = torch.randn(bs*H_feat*W_feat, 644).cuda()
        flops_q = FlopCountAnalysis(model.imnet_q, dummy_q_in)
        flops_k = FlopCountAnalysis(model.imnet_k, dummy_k_in)
        flops_v = FlopCountAnalysis(model.imnet_v, dummy_v_in)
        print(f"imnet_q FLOPs: {flops_q.total()/1e9:.2f} G")
        print(f"imnet_k FLOPs: {flops_k.total()/1e9:.2f} G")
        print(f"imnet_v FLOPs: {flops_v.total()/1e9:.2f} G")

    # -------------------- Forward 推理时间 --------------------
    # 预热
    for _ in range(10):
        _ = model(dummy_inp, dummy_coord, dummy_cell)

    # CUDA 事件计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    num_runs = 100
    torch.cuda.synchronize()
    start_event.record()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_inp, dummy_coord, dummy_cell)
    end_event.record()
    torch.cuda.synchronize()
    avg_time_ms = start_event.elapsed_time(end_event) / num_runs
    print(f"===== Forward Inference =====\nAverage single forward pass time: {avg_time_ms:.3f} ms")

    # -------------------- 计算指标 --------------------
    # import time
    # t1 = time.time()
    # res = eval_metrics(loader, model,
    #                    data_norm=config.get('data_norm'),
    #                    eval_type=config.get('eval_type'),
    #                    eval_bsize=config.get('eval_bsize'),
    #                    scale_max=int(args.scale_max),
    #                    verbose=True,
    #                    mcell=bool(args.mcell))
    # t2 = time.time()
    # print('$$$$$$$$$$$$$$$$$$$$$[result]$$$$$$$$$$$$$$$$$$$$$: PSNR: {:.4f}   SSIM: {:.4f}   MSE: {:.9f}   time: {:.3f}s'.format(
    #     res[0], res[1], res[2], t2 - t1))
