import os
import shutil

# 定义父文件夹路径和目标文件夹路径
origin_dir = "/home/caoxinyu/Arbitrary-scale/data/BraTS/Origin"  # 替换为实际父文件夹路径
destination_dir = "/home/caoxinyu/Arbitrary-scale/data/BraTS/3D/T2_FLAIR"  # 替换为目标文件夹路径

# 确保目标文件夹存在，不存在则创建
os.makedirs(destination_dir, exist_ok=True)

# 遍历父文件夹及其子文件夹
for root, dirs, files in os.walk(origin_dir):
    for file in files:
        # 检查文件名是否以 "t1.nii.gz" 结尾
        if file.endswith("flair.nii.gz"):
            source_file = os.path.join(root, file)  # 获取源文件路径
            target_file = os.path.join(destination_dir, file)  # 目标文件路径

            # 如果目标文件夹中已存在同名文件，添加前缀避免覆盖
            if os.path.exists(target_file):
                base, ext = os.path.splitext(file)
                target_file = os.path.join(destination_dir, f"{base}_copy{ext}")

            # 移动文件
            shutil.move(source_file, target_file)
            print(f"Moved: {source_file} -> {target_file}")
