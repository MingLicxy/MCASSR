import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict

# 用于注册钩子函数，从而捕获模型的某一层的输出特征图
# module_name：是指定的模型层; layer_index：层的索引，用于选择特定的子层
class FM_visualize:
    def __init__(self, module_name, layer_index):
        if layer_index is None:
            self.hook = module_name.register_forward_hook(self.hook_fn)
        else:
            self.hook = module_name[layer_index].register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    #parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #TODO 将图像转化为张量，并且将RGB值归一化到[0,1]
    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))

    # 加载预训练好的模型
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    print('Check the module name and number of the model!!')
    print(model)

    ##################### 核心逻辑(需要依据模型真实结构来设置) ####################
    #visual = FM_visualize(model.encoder.body[0].body[12], None)
    visual = FM_visualize(model.encoder.conv_after_body, None)   # Enter the model module and number to visualize
    out_channal = 64   # Enter the last channel of the model layer to visualize
    ############################################################################


    # h,w是output分辨率
    h, w = list(map(int, args.resolution.split(',')))
    # coord网格中心点坐标维度[h,w,2],坐标范围[-1,1]=>cell像素尺寸[2/h,2/w]
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    
    # 模型预测
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    
    # 获取特征图
    activations = visual.features
    rows = int(out_channal/8)
    columns = 8
    fig, axes = plt.subplots(rows,columns,figsize=(30, 30))
    for row in range(rows):
        for column in range(columns):
            axis = axes[row][column]
            axis.get_xaxis().set_ticks([])
            axis.get_yaxis().set_ticks([])
            axis.imshow(activations[0][row*8+column])
    plt.savefig('/home/caoxinyu/Arbitrary-scale/liif-main/results/features/CSNLA_visual.jpg')
    print('finished!!!!!!!!!!!')
    

    # 保存预测结果
    #pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    #transforms.ToPILImage()(pred).save(args.output)
