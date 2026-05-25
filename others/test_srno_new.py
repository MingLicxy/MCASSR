import argparse
import os
import math
import sys
import time
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets
import models
import utils


def batched_predict(model, inp, coord, cell, bsize):
    """
    分块预测，适配 SRNO 输入维度
    coord: (B, N, 2)
    cell: (B, 2)
    """
    with torch.no_grad():
        model.gen_feat(inp)
        B, N, _ = coord.shape
        H = W = int(math.sqrt(N))  # 假设 coord 是均匀网格
        ql = 0
        preds = []

        while ql < N:
            qr = min(ql + bsize, N)
            # reshape成 (B, H, W, 2)
            coord_batched = coord[:, ql:qr, :].view(B, H, W, 2)
            pred = model.query_rgb(coord_batched, cell)  # cell 不切片
            preds.append(pred)
            ql = qr

        pred = torch.cat(preds, dim=2)  # 按 width 拼接
    return pred


def eval_metrics(loader, model, data_norm=None, eval_type=None, eval_bsize=None, scale_max=4,
                 verbose=False, mcell=False):
    """
    计算 PSNR / SSIM / MSE
    """
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

    # 指标选择
    if eval_type is None:
        psnr_fn = utils.calc_psnr
        ssim_fn = utils.ssim
        mse_fn = utils.mse
    elif eval_type.startswith('metrics'):
        scale = int(eval_type.split('-')[1])
        psnr_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
        ssim_fn = utils.ssim
        mse_fn = utils.mse
    elif eval_type.startswith('ssim'):
        ssim_fn = utils.ssim
        psnr_fn = utils.calc_psnr
        mse_fn = utils.mse
    elif eval_type.startswith('mse'):
        mse_fn = utils.mse
        psnr_fn = utils.calc_psnr
        ssim_fn = utils.ssim
    else:
        raise NotImplementedError(f"Unknown eval_type: {eval_type}")

    val_res_psnr = utils.Averager()
    val_res_ssim = utils.Averager()
    val_res_mse = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch['inp'] - inp_sub) / inp_div
        coord = batch['coord']  # (B, N, 2)
        cell = batch['cell']    # (B, 2)

        if mcell:
            c = max(scale / scale_max, 1)
        else:
            c = 1

        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell * c)
        else:
            pred = batched_predict(model, inp, coord, cell * c, eval_bsize)

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        psnr_res = psnr_fn(pred, batch['gt'])
        ssim_res = ssim_fn(pred, batch['gt'])
        mse_res = mse_fn(pred, batch['gt'])

        val_res_psnr.add(psnr_res.item(), inp.shape[0])
        val_res_ssim.add(ssim_res.item(), inp.shape[0])
        val_res_mse.add(mse_res.item(), inp.shape[0])

        if verbose:
            pbar.set_description(f'val PSNR: {val_res_psnr.item():.4f}')

    return val_res_psnr.item(), val_res_ssim.item(), val_res_mse.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
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

    # 模型加载
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    # -------------------- 统计 Params / FLOPs / forward --------------------
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    B, C, H, W = 1, 3, 128, 128
    dummy_inp = torch.randn(B, C, H, W).cuda()
    dummy_coord = torch.zeros(B, H, W, 2).cuda()
    dummy_cell = torch.zeros(B, 2).cuda()  # SRNO 的 cell 维度为 (B,2)

    # Params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {total_params/1e6:.2f} M")
    print(f"Trainable Params: {trainable_params/1e6:.2f} M")

    # FLOPs
    try:
        flops = FlopCountAnalysis(model, (dummy_inp, dummy_coord, dummy_cell))
        print("Total FLOPs: {:.2f} G".format(flops.total() / 1e9))
    except Exception as e:
        print("FLOPs计算失败:", e)

    # forward 时间
    warmup_runs = 10    # 预热次数
    measure_runs = 100   # 实际测量推理次数

    # -------------------------
    # Warm-up
    # -------------------------
    for _ in range(warmup_runs):
        _ = model(dummy_inp, dummy_coord, dummy_cell)

    # -------------------------
    # Measure multiple runs
    # -------------------------
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    with torch.no_grad():
        for _ in range(measure_runs):
            _ = model(dummy_inp, dummy_coord, dummy_cell)

    end_event.record()
    torch.cuda.synchronize()

    # 平均单次推理时间
    elapsed_ms = start_event.elapsed_time(end_event) / measure_runs

    print(f"Average single forward pass time: {elapsed_ms:.3f} ms")


    # -------------------- 计算指标 --------------------
    # t1 = time.time()
    # res = eval_metrics(loader, model,
    #                    data_norm=config.get('data_norm'),
    #                    eval_type=config.get('eval_type'),
    #                    eval_bsize=config.get('eval_bsize'),
    #                    scale_max=int(args.scale_max),
    #                    verbose=True,
    #                    mcell=bool(args.mcell))
    # t2 = time.time()

    # print('$$$$$$$$$$$$$$$$$$$$$[result]$$$$$$$$$$$$$$$$$$$$$')
    # print('PSNR: {:.4f} | SSIM: {:.4f} | MSE: {:.9f} | Forward Time: {:.3f}s'.format(
    #     res[0], res[1], res[2], t2 - t1))
