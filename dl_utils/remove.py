# 从指定文件夹中随机选取100张图片复制到另一个指定文件夹中
import os
import random
import shutil

def copy_random_images(src_folder, dest_folder, num_images=100):
    # 确保目标文件夹存在
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # 获取源文件夹中所有图片文件
    image_files = [f for f in os.listdir(src_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    
    # 如果图片数量小于num_images，复制所有图片
    if len(image_files) < num_images:
        num_images = len(image_files)

    # 随机选取指定数量的图片
    selected_images = random.sample(image_files, num_images)
    
    # 复制选中的图片到目标文件夹
    for image in selected_images:
        src_path = os.path.join(src_folder, image)
        dest_path = os.path.join(dest_folder, image)
        shutil.copy(src_path, dest_path)
        print(f"Copied {image} to {dest_folder}")

# 使用示例
src_folder = "/home/caoxinyu/Arbitrary-scale/data/train_data/IR700_clean"
dest_folder = "/home/caoxinyu/Arbitrary-scale/data/test_data/IR700_test/GT"
copy_random_images(src_folder, dest_folder, num_images=100)
