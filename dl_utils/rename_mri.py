import os

# 文件夹路径
folder_path = '/home/caoxinyu/Arbitrary-scale/data/PMC/3D/created-3D/3T_final/T2_3T'

# 获取文件夹中的所有文件（按文件名排序）
files = sorted(os.listdir(folder_path))

# 遍历文件并重命名
for index, file_name in enumerate(files):
    # 获取文件的扩展名
    file_base, file_ext = os.path.splitext(file_name)
    
    # 生成新的文件名，添加顺序号前缀
    new_file_name = f"{index+1:03d}_{file_base}{file_ext}" # 4位数
    
    # 原文件路径和新文件路径
    old_file_path = os.path.join(folder_path, file_name)
    new_file_path = os.path.join(folder_path, new_file_name)
    
    # 重命名文件
    os.rename(old_file_path, new_file_path)
    print(f"Renamed: {file_name} -> {new_file_name}")

print("All files have been renamed with sequence number prefix.")




###############TODO 移除前缀名 ###############
# import os

# # 文件夹路径
# folder_path = '/home/caoxinyu/Arbitrary-scale/data/BraTS/3D/T1'

# # 获取文件夹中的所有文件（按文件名排序）
# files = sorted(os.listdir(folder_path))

# # 遍历文件并移除序号前缀
# for file_name in files:
#     # 检查文件名是否包含前缀（例如 0001_）
#     if "_" in file_name:
#         # 分割文件名，移除前缀
#         new_file_name = "_".join(file_name.split("_")[1:])
        
#         # 原文件路径和新文件路径
#         old_file_path = os.path.join(folder_path, file_name)
#         new_file_path = os.path.join(folder_path, new_file_name)
        
#         # 重命名文件
#         os.rename(old_file_path, new_file_path)
#         print(f"Renamed: {file_name} -> {new_file_name}")

# print("All file prefixes have been cleared.")
