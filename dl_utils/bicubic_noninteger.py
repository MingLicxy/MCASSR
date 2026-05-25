import os
from pathlib import Path
import cv2

def downsample_images(input_folder, scale, hr_output_folder, lr_output_folder):
    # 确保输出文件夹存在
    Path(hr_output_folder).mkdir(parents=True, exist_ok=True)
    Path(lr_output_folder).mkdir(parents=True, exist_ok=True)
    
    # 遍历输入文件夹中的所有图像
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        
        # 检查是否是图像文件
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        
        # 读取图像
        img = cv2.imread(input_path)
        if img is None:
            print(f"无法读取图像: {input_path}")
            continue
        
        # 保存HR图像
        hr_output_path = os.path.join(hr_output_folder, file_name)
        cv2.imwrite(hr_output_path, img)
        
        # 计算LR尺寸并进行双三次下采样
        lr_width = int(img.shape[1] / scale)
        lr_height = int(img.shape[0] / scale)
        lr_img = cv2.resize(img, (lr_width, lr_height), interpolation=cv2.INTER_CUBIC)
        
        # 保存LR图像
        lr_output_path = os.path.join(lr_output_folder, file_name)
        cv2.imwrite(lr_output_path, lr_img)
        print(f"处理完成: {file_name}")

# 使用示例
input_folder = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/others/GT"  # 输入文件夹路径
scale = 3.1  # 缩小倍数 # 3.1  4.5  5.9  7.3
hr_output_folder = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/others/bicubic/X3.1/HR"  # HR图像保存路径
lr_output_folder = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/others/bicubic/X3.1/LR"  # LR图像保存路径

downsample_images(input_folder, scale, hr_output_folder, lr_output_folder)
