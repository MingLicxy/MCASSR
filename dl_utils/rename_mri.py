import os

# 文件夹路径
folder_path = '/home/caoxinyu/Arbitrary-scale/data/IXI/3D/T2    '

# 获取文件夹中的所有文件（按文件名排序）
files = sorted(os.listdir(folder_path))

# 遍历文件并重命名
for index, file_name in enumerate(files):
    # 获取文件的扩展名
    file_base, file_ext = os.path.splitext(file_name)
    
    # 生成新的文件名，添加顺序号前缀
    new_file_name = f"{index+1:03d}_{file_base}{file_ext}"
    
    # 原文件路径和新文件路径
    old_file_path = os.path.join(folder_path, file_name)
    new_file_path = os.path.join(folder_path, new_file_name)
    
    # 重命名文件
    os.rename(old_file_path, new_file_path)
    print(f"Renamed: {file_name} -> {new_file_name}")

print("All files have been renamed with sequence number prefix.")
