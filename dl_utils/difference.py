import os
from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def apply_colormap(diff_image, cmap='jet'):
    # 将PIL图像转换为NumPy数组
    diff_np = np.array(diff_image)

    # 归一化像素值到0到1之间
    norm_diff = (diff_np - diff_np.min()) / (diff_np.max() - diff_np.min())

    # 使用新的 colormap 获取方法
    colormap = plt.colormaps.get_cmap(cmap)
    colored_diff = colormap(norm_diff)[:, :, :3]  # 忽略alpha通道

    # 转换回PIL图像
    colored_diff_img = Image.fromarray((colored_diff * 255).astype(np.uint8))
    return colored_diff_img

def visualize_image_difference(folder1, folder2, output_folder, colormap='jet'):
    # 确保目标文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取两个文件夹中的文件列表，并进行排序以确保文件对应
    images1 = sorted([f for f in os.listdir(folder1) if f.endswith(('.png', '.jpg', '.jpeg'))])
    images2 = sorted([f for f in os.listdir(folder2) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # 确保两个文件夹中的图像数量一致
    if len(images1) != len(images2):
        print("两个文件夹中的图像数量不一致。")
        return

    # 遍历图像并计算差值
    for img1, img2 in zip(images1, images2):
        # 打开两幅图像
        path1 = os.path.join(folder1, img1)
        path2 = os.path.join(folder2, img2)
        image1 = Image.open(path1)
        image2 = Image.open(path2)

        # 确保图像尺寸一致
        if image1.size != image2.size:
            print(f"图像尺寸不一致：{img1} 和 {img2}")
            continue

        # 计算图像差值
        diff_image = ImageChops.difference(image1, image2).convert('L')  # 转换为灰度图

        # 应用热成像风格的颜色映射
        colored_diff_image = apply_colormap(diff_image, colormap)

        # 将差值图像保存到输出文件夹
        output_path = os.path.join(output_folder, f"diff_{img1}")
        colored_diff_image.save(output_path)
        print(f"保存差值图像：{output_path}")





# 使用示例
folder1 = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/others/GT"
folder2 = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/others/bicubic/X6/SR"
output_folder = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/others/bicubic/X6/Diff"

visualize_image_difference(folder1, folder2, output_folder)
