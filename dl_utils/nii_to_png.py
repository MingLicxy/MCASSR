import nibabel as nib
import numpy as np
import os
import glob
import re
from PIL import Image

# 输入和输出路径 (只切PD与T2)
input_folder = '/home/caoxinyu/Arbitrary-scale/data/PMC/3D/created-3D/3T_final/T2_3T'
output_folder = '/home/caoxinyu/Arbitrary-scale/data/PMC/2D/valid/3T/T2'  # 注意[train, test, valid]

# TODO 切片范围和切片间距（取最中间的12张切片）（与选取的切片主轴密切相关）
slice_start = 73  # 起始切片索引 (130/2)-(3x6)
slice_end = 109    # 结束切片索引 (130/2)+(3x6)
slice_step = 6   # 切片间距 [12对应切3张]  [18对应切两张]

# 文件名前缀范围
prefix_start = 171     # train[001, 500] test[501, 560] valid[561, 578]
prefix_end = 181       # PMC: train[1, 150] test[151, 170] valid[171, 181] 切6张

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历所有 .nii.gz 文件
for nii_file_path in glob.glob(os.path.join(input_folder, '*.nii.gz')):
    # 提取文件名前缀
    file_name = os.path.basename(nii_file_path)
    match = re.match(r'^(\d+)_', file_name)

    if match:
        prefix = int(match.group(1))

        # 只处理前缀在指定范围内的文件
        if prefix_start <= prefix <= prefix_end:
            # 读取 NIfTI 文件
            img = nib.load(nii_file_path)
            data = img.get_fdata()
            
            # 文件名处理
            base_name = file_name.replace('.nii.gz', '')
            output_file_base = os.path.join(output_folder, base_name)

            # 获取 Z 轴的切片总数（默认沿着第三个维度进行切片）
            total_slices = data.shape[2]

            # 根据切片范围和间距设置索引
            slices = range(max(0, slice_start), min(total_slices, slice_end), slice_step)

            # 遍历指定范围内的切片并保存为图像
            for i in slices:
                slice_data = data[:, :, i]

                # 归一化到 [0, 255] 范围，确保保存为灰度图像
                slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
                slice_data = slice_data.astype(np.uint8)

                # 保存为单通道 PNG 图像
                output_file = f'{output_file_base}_slice_{i:03d}.png'
                Image.fromarray(slice_data).save(output_file)

            print(f"Converted {nii_file_path} into 2D slices (from slice {slice_start} to {slice_end} with step {slice_step}) and saved to {output_folder}")

print(f"All .nii.gz files with prefixes from {prefix_start} to {prefix_end} have been processed and saved to {output_folder}")



#TODO 只在中间位置切一次
# import nibabel as nib
# import numpy as np
# import os
# import glob
# import re
# from PIL import Image

# # 输入和输出路径
# input_folder = '/home/caoxinyu/Arbitrary-scale/data/BraTS/3D/T2'
# output_folder = '/home/caoxinyu/Arbitrary-scale/data/BraTS/2D/valid/T2'  # 注意[train, test, valid]

# # 文件名前缀范围
# prefix_start = 1201      # train[001, 500] test[501, 560] valid[561, 578]
# prefix_end = 1251        # BraTS: train[0001, 1000] test[1001, 1200] valid[1201, 1251]

# # 创建输出文件夹（如果不存在）
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # 遍历所有 .nii.gz 文件
# for nii_file_path in glob.glob(os.path.join(input_folder, '*.nii.gz')):
#     # 提取文件名前缀
#     file_name = os.path.basename(nii_file_path)
#     match = re.match(r'^(\d+)_', file_name)

#     if match:
#         prefix = int(match.group(1))

#         # 只处理前缀在指定范围内的文件
#         if prefix_start <= prefix <= prefix_end:
#             # 读取 NIfTI 文件
#             img = nib.load(nii_file_path)
#             data = img.get_fdata()

#             # 文件名处理
#             base_name = file_name.replace('.nii.gz', '')
#             output_file = os.path.join(output_folder, f'{base_name}_center_slice.png')

#             # 获取 Z 轴的切片总数（默认沿着第三个维度进行切片）
#             total_slices = data.shape[2]

#             # 计算中心切片索引
#             center_slice_idx = total_slices // 2

#             # 提取中心切片
#             slice_data = data[:, :, center_slice_idx]

#             # 归一化到 [0, 255] 范围，确保保存为灰度图像
#             slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
#             slice_data = slice_data.astype(np.uint8)

#             # 保存为单通道 PNG 图像
#             Image.fromarray(slice_data).save(output_file)

#             print(f"Saved center slice {center_slice_idx} from {nii_file_path} to {output_file}")

# print(f"All .nii.gz files with prefixes from {prefix_start} to {prefix_end} have been processed and their center slices saved to {output_folder}")
