import argparse
import os
import sys
import yaml
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets
import models
import dl_utils
from functools import partial
from tqdm import tqdm
from tensorboardX import SummaryWriter

####################################### 定义辅助函数 #######################################
def check_updown(up_down):
    ud = up_down.split('-')[0]  # bic / avg / none
    ud_scale = int(up_down.split('-')[1])
    return ud, ud_scale


def eval_psnr_ope(loader, model):
    model.eval()

    metric_fn_psnr = dl_utils.calc_psnr

    val_res_psnr = dl_utils.Averager()

    pbar = tqdm(loader, leave=False, desc='eval_psnr')
    with torch.no_grad():
        for batch in pbar:
            batch_lr = batch['lr'].cuda()
            gt = batch['hr'].cuda()
            gt_size = gt.shape[2]
            pred, _ = model.inference(batch_lr, h=gt_size, w=gt_size)
            pred.clamp_(-1, 1)

            res_psnr = metric_fn_psnr(pred, gt)

            val_res_psnr.add(res_psnr.item(), gt.shape[0])

            pbar.set_description('val {:.4f}'.format(val_res_psnr.item()))

    return val_res_psnr.item(), 0


def eval_psnr(loader, model, eval_type=None, eval_bsize=None, window_size=0, scale_max=4, fast=False,
              verbose=False):
    model.eval()

    if eval_type is None:
        metric_fn = dl_utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(dl_utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(dl_utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = dl_utils.Averager()

    pbar = tqdm(loader, leave=False, desc='eval_psnr')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']
        # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]

            coord = dl_utils.make_coord((scale * (h_old + h_pad), scale * (w_old + w_pad))).unsqueeze(0).cuda()
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / inp.shape[-2] / scale
            cell[:, :, 1] *= 2 / inp.shape[-1] / scale
        else:
            h_pad = 0
            w_pad = 0

            coord = batch['coord']
            cell = batch['cell']

        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell)
        else:
            if fast:
                pred = model(inp, coord, cell * max(scale / scale_max, 1))
            else:
                pred = dl_utils.batched_predict(model, inp, coord, cell * max(scale / scale_max, 1),
                                                eval_bsize)  # cell clip for extrapolation
        if type(pred) is tuple:
            pred = pred[0].clamp_(-1, 1)
        else:
            pred = pred.clamp_(-1, 1)

        if eval_type is not None and fast == False:  # reshape for shaving-eval
            # gt reshape
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

            # prediction reshape
            ih += h_pad
            iw += w_pad
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


def test_both_ope(loader, model, log_fn, log_name, eval_type=None, up_down=None):
    model.eval()
    ud = 'none'
    ud_scale = 1
    if up_down is not None:
        ud, ud_scale = check_updown(up_down)
    metric_fn_ssim = dl_utils.calc_ssim
    if eval_type is None:
        metric_fn_psnr = dl_utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn_psnr = partial(dl_utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn_psnr = partial(dl_utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res_psnr = dl_utils.Averager()
    val_res_ssim = dl_utils.Averager()
    avg_time_encoder = dl_utils.Averager()
    avg_time_render = dl_utils.Averager()
    avg_time_all = dl_utils.Averager()
    pbar = tqdm(loader, leave=False, desc='test_both')
    id = 0
    with torch.no_grad():
        for batch in pbar:
            torch.cuda.empty_cache()
            batch_lr = batch['lr'].cuda()
            gt = batch['gt'].cuda()
            gt_size = gt.shape[-2:]
            if ud == 'none':
                pred, run_time = model.inference(batch_lr, h=gt_size[0], w=gt_size[1])
                pred.clamp_(-1, 1)
            elif ud == 'bic':
                pred, run_time = model.inference(batch_lr, h=gt_size[0] * ud_scale, w=gt_size[1] * ud_scale)

                pred = dl_utils.resize_img(pred, (gt_size[0], gt_size[1])).cuda()
                pred.clamp_(-1, 1)
            elif ud == 'avg':
                pred, run_time = model.inference(batch_lr, h=gt_size[0] * ud_scale, w=gt_size[1] * ud_scale)
                m = nn.AdaptiveAvgPool2d((gt_size[0], gt_size[1]))
                pred = m(pred)
                pred.clamp_(-1, 1)

            else:
                RuntimeError('updown fault')

            res_psnr = metric_fn_psnr(pred, gt)
            res_ssim = metric_fn_ssim(pred, gt)
            log_fn(
                f'test_img: {id}, psnr: {res_psnr.item()}, ssim: {res_ssim.item()}, time: {run_time[0]}s/{run_time[1]}s/{run_time[2]}s',
                filename=log_name)
            val_res_psnr.add(res_psnr.item(), gt.shape[0])
            val_res_ssim.add(res_ssim.item(), gt.shape[0])
            avg_time_encoder.add(run_time[0], gt.shape[0])
            avg_time_render.add(run_time[1], gt.shape[0])
            avg_time_all.add(run_time[2], gt.shape[0])

            id += 1

            pbar.set_description('img:{}, psnr: {:.4f}, ssim: {:.4f}'.format(id - 1, res_psnr.item(), res_ssim.item()))

    return val_res_psnr.item(), val_res_ssim.item(), [avg_time_encoder.item(), avg_time_render.item(),
                                                      avg_time_all.item()]


def single_img_sr(lr_img, model, h, w, gt=None, up_down=None, flip=None):
    model.eval()
    ud = 'none'
    ud_scale = 1
    if up_down is not None:
        ud, ud_scale = check_updown(up_down)
    with torch.no_grad():
        if flip is not None:
            pred, run_time = model.inference(lr_img, h=h, w=w, flip_conf=flip)
            pred.clamp_(-1, 1)
        else:
            if ud == 'none':
                pred, run_time = model.inference(lr_img, h=h, w=w)
                pred.clamp_(-1, 1)
            elif ud == 'bic':
                pred, run_time = model.inference(lr_img, h=h * ud_scale, w=w * ud_scale)

                pred = dl_utils.resize_img(pred, (h, w)).cuda()
                pred.clamp_(-1, 1)
            elif ud == 'avg':
                pred, run_time = model.inference(lr_img, h=h * ud_scale, w=w * ud_scale)
                m = nn.AdaptiveAvgPool2d((h, w))
                pred = m(pred)
                pred.clamp_(-1, 1)

            else:
                RuntimeError('updown fault')

        if gt is not None:
            metric_fn_psnr = dl_utils.calc_psnr
            metric_fn_ssim = dl_utils.calc_ssim
            res_psnr = metric_fn_psnr(pred, gt)
            res_ssim = metric_fn_ssim(pred, gt)
            return pred, res_psnr, res_ssim, run_time
        else:
            return pred, None, None, run_time


######################################## test #########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='save/train_rdn-ope')
    parser.add_argument('--test_config', default='configs/test-configs/test_CIR-SR-set14-x6.yaml')
    #parser.add_argument('--model') #TODO
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test_name = args.test_config.split('/')[-1].split('.')[-2]
    # 创建保存测试结果的目录路径
    save_dir = os.path.join(args.exp_folder, 'TEST_folder/' + test_name)
    log, _ = dl_utils.set_save_path(save_dir, writer=False)
    os.makedirs(save_dir, exist_ok=True)

    # 获取实验文件夹中所有符合条件的检查点文件（以0.pth或5.pth结尾）
    ckpt_list = [int(ckpt_name.split('.')[0].split('-')[-1]) for ckpt_name in os.listdir(args.exp_folder) if
                 ckpt_name.endswith('0.pth') or ckpt_name.endswith('5.pth')]
    ckpt_list = sorted(ckpt_list)

    # 检查测试信息文件是否存在，如果存在，则加载已测试的检查点信息
    test_dic = {}
    if 'test_info.json' in os.listdir(save_dir):
        test_dic = dl_utils.load_json(path=os.path.join(save_dir, 'test_info.json'))
        tested_ckpt_list = [int(key) for key in test_dic.keys()]
        tested_ckpt_list = sorted(tested_ckpt_list)
    else:
        tested_ckpt_list = []
    
    #TODO 只测试epoch-1000.pth
    for i in range(1000, 1001):
        if i in ckpt_list and i not in tested_ckpt_list:
            # perform test
            ckpt_num = i
            print(f'testing ckpt: {ckpt_num}')
            ckpt_name = f'epoch-{ckpt_num}.pth'
            # 获取预训练模型路径
            resume_path = os.path.join(args.exp_folder, ckpt_name)

            log_name = ckpt_name.split('.')[-2] + '_log.txt'

            with open(args.test_config, 'r') as f:
                test_config = yaml.load(f, Loader=yaml.FullLoader)
            
            # 创建数据集和数据加载器
            test_spec = test_config['test_dataset']
            dataset = datasets.make(test_spec['dataset'])
            dataset = datasets.make(test_spec['wrapper'], args={'dataset': dataset})
            loader = DataLoader(dataset, batch_size=test_spec['batch_size'],
                                num_workers=8, pin_memory=True)

            # 加载检查点模型
            sv_file = torch.load(resume_path, map_location=lambda storage, loc: storage)
            model = models.make(sv_file['model'], load_sd=True).cuda()

            #model_spec = torch.load(args.model)['model']
            #model = models.make(model_spec, load_sd=True).cuda()

            test_psnr, test_ssim, test_run_time = test_both_ope(loader, model, log, log_name,
                                                                eval_type=test_config.get('eval_type'), up_down=test_config.get('up_down'))

            log('test avg: psnr={:.4f}'.format(test_psnr), filename=log_name)
            log('test avg: ssim={:.4f}'.format(test_ssim), filename=log_name)
            log(f'test avg encoder time: {test_run_time[0]}s', filename=log_name)
            log(f'test avg render time: {test_run_time[1]}s', filename=log_name)
            log(f'test avg all time: {test_run_time[2]}s', filename=log_name)

            test_dic.update({str(i): [test_psnr, test_ssim]})
            dl_utils.save_json(path=os.path.join(save_dir, 'test_info.json'), save_dic=test_dic)

    writer = SummaryWriter(os.path.join(save_dir, 'runs'))
    all_keys = sorted([int(key) for key in test_dic.keys()])
    for key in all_keys:
        writer.add_scalar('scalar/test_psnr', test_dic[str(key)][0], key)
        writer.add_scalar('scalar/test_ssim', test_dic[str(key)][1], key)
