import os
from PIL import Image, ImageOps

# scale=[2, 3, 4]
def bicubic_downsample(image, scale):

    # 获取原始图像的尺寸
    original_width, original_height = image.size

    # 计算填充后的尺寸，使其能被scale整除
    padding_width = (scale - original_width % scale) % scale
    padding_height = (scale - original_height % scale) % scale
    
    # 添加填充
    image_padded = ImageOps.expand(image, (0, 0, padding_width, padding_height), fill=0)

    # 计算下采样后的新尺寸（可以整除）
    new_width = image_padded.size[0] // scale
    new_height = image_padded.size[1] // scale
    
    # 下采样图像（核心功能代码）
    downsampled_image = image_padded.resize((new_width, new_height), Image.BICUBIC)
    
    # 返回填充图像与下采样图像
    return image_padded, downsampled_image

def process_images_in_folder(input_folder, output_hr_folder, output_lr_folder, scale):
    
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_hr_folder):
        os.makedirs(output_hr_folder)
    if not os.path.exists(output_lr_folder):
        os.makedirs(output_lr_folder)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # 检查文件是否为图像文件
        try:
            with Image.open(input_path) as img:
                # 对图像进行下采样处理
                image_padded, downsampled_image = bicubic_downsample(img, scale)
                
                # 保存下采样后的图像
                output_hr_path = os.path.join(output_hr_folder, filename)
                output_lr_path = os.path.join(output_lr_folder, filename)
                image_padded.save(output_hr_path)
                downsampled_image.save(output_lr_path)

                print(f"Processed and saved: {output_hr_path}")
                print(f"Processed and saved: {output_lr_path}")
        except IOError:
            print(f"Skipped non-image file: {input_path}")

# 示例使用
input_folder = '/home/caoxinyu/Arbitrary-scale/liif-main/demo/others/GT'
output_hr_folder = '/home/caoxinyu/Arbitrary-scale/liif-main/demo/others/bicubic/X8/HR'
output_lr_folder = '/home/caoxinyu/Arbitrary-scale/liif-main/demo/others/bicubic/X8/LR'
scale = 8  # 将图像缩小一半 [2, 3, 4]

# 对文件夹中的所有图像进行下采样处理
process_images_in_folder(input_folder, output_hr_folder, output_lr_folder, scale)
