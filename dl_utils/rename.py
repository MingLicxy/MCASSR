
import os
import re
from PIL import Image

# bmp_to_png
def batch_convert_bmp_to_png(input_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中所有 BMP 文件的列表
    bmp_files = [f for f in os.listdir(input_folder) if f.endswith('.bmp')]

    # 遍历每个 BMP 文件进行转换和重命名
    for idx, bmp_file in enumerate(bmp_files):
        # 构建输入和输出文件的完整路径
        bmp_path = os.path.join(input_folder, bmp_file)
        png_file = f'image_{idx+1:03d}.png'  # 生成序号格式的 PNG 文件名
        png_path = os.path.join(output_folder, png_file)

        try:
            # 打开 BMP 图像
            img = Image.open(bmp_path)

            # 保存为 PNG 格式
            img.save(png_path, format='PNG')
            print(f"转换成功: {bmp_file} -> {png_file}")

        except IOError as e:
            print(f"转换失败: {bmp_file} -> {e}")

# 顺序命名
def rename_png_files(folder_path):
    """
    将指定文件夹中的 .png 图片按序号重命名。
    
    参数:
    folder_path (str): 要处理的文件夹路径。
    """
    # 获取文件夹中的所有 .png 文件
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # 对文件进行排序
    png_files.sort()

    # 重命名文件
    for i, file_name in enumerate(png_files, start=1):
        old_path = os.path.join(folder_path, file_name)
        new_file_name = f'{i:05d}.png'
        new_path = os.path.join(folder_path, new_file_name)
        os.rename(old_path, new_path)
        print(f'已重命名: {file_name} -> {new_file_name}')


if __name__ == "__main__":

    # # 输入文件夹和输出文件夹路径 
    # input_folder = '/home/caoxinyu/Arbitrary-scale/data/train_data/IR700'
    # output_folder = '/home/caoxinyu/Arbitrary-scale/data/train_data/IR700_clean'

    # # 执行批量转换
    # batch_convert_bmp_to_png(input_folder, output_folder)





    # 使用示例
    rename_png_files('/home/caoxinyu/UNet-based/data/CT_PNG/GT')
