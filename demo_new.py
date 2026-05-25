import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict


def is_image_file(name):
    return name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', required=True, help='LR image folder')
    parser.add_argument('--output_dir', required=True, help='SR output folder')
    parser.add_argument('--model', required=True, help='pretrained model path')
    parser.add_argument('--scale', type=float, default=2, help='upsampling scale, e.g. 2 or 4')
    parser.add_argument('--gpu', default='0')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------
    # Load model
    # ------------------------
    model = models.make(
        torch.load(args.model, map_location='cpu')['model'],
        load_sd=True
    ).to(device)
    model.eval()

    to_tensor = transforms.ToTensor()

    img_names = sorted([f for f in os.listdir(args.input_dir) if is_image_file(f)])
    print(f'Found {len(img_names)} images in {args.input_dir}')

    with torch.no_grad():
        for name in img_names:
            in_path = os.path.join(args.input_dir, name)
            out_path = os.path.join(args.output_dir, name)

            # ------------------------
            # Read image
            # ------------------------
            img = Image.open(in_path).convert('RGB')
            img_tensor = to_tensor(img)  # [3, H, W], range [0,1]

            _, h_lr, w_lr = img_tensor.shape
            h_sr = int(h_lr * args.scale)
            w_sr = int(w_lr * args.scale)

            # ------------------------
            # Make coord & cell
            # ------------------------
            coord = make_coord((h_sr, w_sr)).to(device)
            cell = torch.ones_like(coord)
            cell[:, 0] *= 2 / h_sr
            cell[:, 1] *= 2 / w_sr

            # ------------------------
            # Predict
            # ------------------------
            pred = batched_predict(
                model,
                ((img_tensor - 0.5) / 0.5).unsqueeze(0).to(device),
                coord.unsqueeze(0),
                cell.unsqueeze(0),
                bsize=30000
            )[0]

            # ------------------------
            # Post-process & save
            # ------------------------
            pred = (pred * 0.5 + 0.5).clamp(0, 1)
            pred = pred.view(h_sr, w_sr, 3).permute(2, 0, 1).cpu()

            transforms.ToPILImage()(pred).save(out_path)

            print(f'[OK] {name} -> {out_path}')

    print('All images processed.')
