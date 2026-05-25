import argparse
import os
import random
import sys

import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets
import models
import utils
from test import eval_psnr
from scheduler import GradualWarmupScheduler

torch.cuda.empty_cache()


# -----------------------------
# DataLoader 创建函数
# -----------------------------
def make_data_loader(dataset, batch_size, shuffle=False, collate_fn=None):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=None,
        collate_fn=collate_fn  # 传入原始 dataset 的 collate_fn
    )
    return loader


# -----------------------------
# 模型与优化器初始化
# -----------------------------
def prepare_training(config, log):
    """与原函数逻辑相同，每折训练都重新初始化模型和优化器"""
    model = models.make(config['model']).cuda()
    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    epoch_start = 1
    if config.get('multi_step_lr') is not None:
        lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
    elif config.get('warmup_step_lr') is not None:
        cosine = CosineAnnealingLR(
            optimizer, config['epoch_max'] - config['warmup_step_lr']['total_epoch']
        )
        lr_scheduler = GradualWarmupScheduler(
            optimizer, **config['warmup_step_lr'], after_scheduler=cosine
        )
    else:
        lr_scheduler = None
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


# -----------------------------
# 单轮训练
# -----------------------------
def train_one_epoch(train_loader, model, optimizer, epoch, writer, config):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()
    metric_fn = utils.calc_psnr

    # 数据归一化
    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    iteration = 0
    pbar = tqdm(train_loader, leave=False, desc='train')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)
        inp = (batch['inp'] - inp_sub) / inp_div
        gt = (batch['gt'] - gt_sub) / gt_div

        pred = model(inp, batch['coord'], batch['cell'])
        loss = loss_fn(pred, gt)
        psnr = metric_fn(pred, gt)

        writer.add_scalars('loss', {'train': loss.item()}, (epoch - 1) * len(train_loader) + iteration)
        writer.add_scalars('psnr', {'train': psnr}, (epoch - 1) * len(train_loader) + iteration)
        iteration += 1

        train_loss.add(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description('train {:.4f}'.format(train_loss.item()))
    return train_loss.item()


# -----------------------------
# K折训练
# -----------------------------
def kfold_train(config, save_path, K=5):
    # 创建完整数据集
    dataset_orig = datasets.make(config['train_dataset']['dataset'])
    dataset = datasets.make(config['train_dataset']['wrapper'], args={'dataset': dataset_orig})

    log, writer = utils.set_save_path(save_path)

    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), 1):
        log(f"=== Fold {fold}/{K} ===")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # ✅ 传入原始 dataset 的 collate_fn
        train_loader = make_data_loader(train_subset, config['train_dataset']['batch_size'],
                                        shuffle=True, collate_fn=dataset.collate_fn)
        val_loader = make_data_loader(val_subset, config['train_dataset']['batch_size'],
                                      shuffle=False, collate_fn=dataset.collate_fn)

        # 初始化模型
        model, optimizer, epoch_start, lr_scheduler = prepare_training(config, log)

        n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        if n_gpus > 1:
            model = nn.DataParallel(model)

        max_val_v = -1e18
        timer = utils.Timer()
        epoch_max = config['epoch_max']
        epoch_val = config.get('epoch_val')
        epoch_save = config.get('epoch_save')

        for epoch in range(epoch_start, epoch_max + 1):
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            train_loss = train_one_epoch(train_loader, model, optimizer, epoch, writer, config)
            if lr_scheduler is not None:
                lr_scheduler.step()

            if n_gpus > 1:
                model_ = model.module
            else:
                model_ = model
            model_spec = config['model']
            model_spec['sd'] = model_.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()
            state = torch.get_rng_state()
            sv_file = {'model': model_spec, 'optimizer': optimizer_spec, 'epoch': epoch, 'state': state}
            torch.save(sv_file, os.path.join(save_path, f'fold{fold}-epoch-last.pth'))

            # 验证
            if (epoch == 1) or (epoch_val is not None and epoch % epoch_val == 0):
                val_res = eval_psnr(val_loader, model_,
                                    data_norm=config['data_norm'],
                                    eval_type=config.get('eval_type'),
                                    eval_bsize=config.get('eval_bsize'))
                log(f"Fold {fold}, Epoch {epoch}, Val PSNR: {val_res:.4f}")
                writer.add_scalars('psnr', {f'val_fold{fold}': val_res}, epoch)
                if val_res > max_val_v:
                    max_val_v = val_res
                    torch.save(sv_file, os.path.join(save_path, f'fold{fold}-epoch-best.pth'))

        fold_metrics.append(max_val_v)
        log(f"=== Fold {fold} Best PSNR: {max_val_v:.4f} ===")

    mean_psnr = sum(fold_metrics) / K
    log(f"K-Fold Average PSNR: {mean_psnr:.4f}")


# -----------------------------
# 主入口
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    setup_seed(2454)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    save_name = args.name or '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('/home/caoxinyu/Arbitrary-scale/liif-main/save', save_name)

    kfold_train(config, save_path, K=5)
