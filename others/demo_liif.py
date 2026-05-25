import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 将图像转化为张量，并且将RGB值归一化到[0,1]
    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))
    # 加载预训练好的模型
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    # h,w是output分辨率
    h, w = list(map(int, args.resolution.split(',')))

    # coord网格中心点坐标维度[h,w,2],坐标范围[-1,1]=>cell像素尺寸[2/h,2/w]
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    # 保存预测结果
    transforms.ToPILImage()(pred).save(args.output)
