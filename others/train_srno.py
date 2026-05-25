import argparse
import os
import random
import sys


import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets
import models
import utils
from test_srno import eval_psnr
from scheduler import GradualWarmupScheduler


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=(tag == 'train'),
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=utils.numpy_init_dict[tag],
        collate_fn=dataset.collate_fn # 批处理函数
        )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


# 训练前准备（创建网络模型，优化器，学习率计划等）
def prepare_training():
    # 预训练
    if config.get('pre_train') is not None:
        print('loading pre_train model...', config['pre_train'])
        log('loading pre_train model... ' + config['pre_train'])
        model = models.make(config['model']).cuda()
        model_dict = model.state_dict()

        # 加载预训练模型参数
        sv_file = torch.load(config['pre_train'])
        pretrained_dict = sv_file['model']['sd']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # 加载预训练优化器
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        epoch_start = 1

        if config.get('multi_step_lr') is not None:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        elif config.get('warmup_step_lr') is not None:
            cosine = CosineAnnealingLR(optimizer, config['epoch_max']-config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer,**config['warmup_step_lr'],after_scheduler=cosine)
        else: 
            lr_scheduler = None

    # 中途恢复训练
    elif config.get('resume') is not None: 
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1

        #TODO 先前训练过程中没有保存state
        # state = sv_file['state']
        # torch.set_rng_state(state)
        # print(f'Resuming from epoch {epoch_start}...')
        # log(f'Resuming from epoch {epoch_start}...')

        if config.get('multi_step_lr') is not None:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        elif config.get('warmup_step_lr') is not None:
            # 余弦退火调度器结合逐步热身调度器（SRNO）
            cosine = CosineAnnealingLR(optimizer, config['epoch_max']-config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer,**config['warmup_step_lr'],after_scheduler=cosine)
        else: 
            lr_scheduler = None

        # 从某个特定的epoch开始训练时，确保学习率调度器已正确步进到该epoch对应的学习率
        # lr_scheduler.last_epoch = epoch_start - 1
        for _ in range(epoch_start - 1):
            lr_scheduler.step()

    # 从头开始训练
    else:
        print('prepare_training from start')
        #TODO 根据配置文件中有关model的超参数创建相应模型
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1

        #TODO 学习率调度
        if config.get('multi_step_lr') is not None:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        elif config.get('warmup_step_lr') is not None:
            cosine = CosineAnnealingLR(optimizer, config['epoch_max']-config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer,**config['warmup_step_lr'],after_scheduler=cosine)
        else: 
            lr_scheduler = None
        # 从头开始训练，该循环不起作用
        for _ in range(epoch_start - 1):
            lr_scheduler.step()

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    log('model: #struct={}'.format(model))
    return model, optimizer, epoch_start, lr_scheduler



def train(train_loader, model, optimizer, \
         epoch):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()
    metric_fn = utils.calc_psnr

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    
    #num_dataset = 800 # DIV2K
    #iter_per_epoch = int(num_dataset / config.get('train_dataset')['batch_size'] \
    #                    * config.get('train_dataset')['dataset']['args']['repeat'])
    iteration = 0
    pbar = tqdm(train_loader, leave=False, desc='train')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch['inp'] - inp_sub) / inp_div
        pred = model(inp, batch['coord'], batch['cell'])

        gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)
        #psnr = metric_fn(pred, gt)
        
        # tensorboard
        #writer.add_scalars('loss', {'train': loss.item()}, (epoch-1)*iter_per_epoch + iteration)
        #writer.add_scalars('psnr', {'train': psnr}, (epoch-1)*iter_per_epoch + iteration)
        iteration += 1
        
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None
        pbar.set_description('train {:.4f}'.format(train_loss.item()))
        
    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=True)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, \
                           epoch)
        if lr_scheduler is not None:
            #TODO 传入epoch是否必要？lr_scheduler.step(epoch)
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()

        #TODO 保存日志
        state = torch.get_rng_state()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch,
            'state': state
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if ((epoch_save is not None) and (epoch % epoch_save == 0)):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch == 1) or ((epoch_val is not None) and (epoch % epoch_val == 0)):
            if n_gpus > 1: #and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_res))
            # writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='./configs/train_edsr-sronet.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0,1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 设置随机种子保证实验可重复
    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # sets the seed for cpu
        torch.cuda.manual_seed(seed)  # Sets the seed for the current GPU.
        torch.cuda.manual_seed_all(seed)  #  Sets the seed for the all GPU.
        # torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    setup_seed(2454)  #2021

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('/home/caoxinyu/Arbitrary-scale/liif-main/save', save_name)

    main(config, save_path)

