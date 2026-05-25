import nibabel as nib
import numpy as np

# # 读取 .nii.gz 文件
# nii_img = nib.load('/home/caoxinyu/Arbitrary-scale/data/IXI/3D/T2/538_IXI607-Guys-1097-T2.nii.gz')

# # 获取图像数据，返回的是一个 NumPy 数组
# mri_data = nii_img.get_fdata()

# # 查看像素值的最小值和最大值
# min_value = np.min(mri_data)
# max_value = np.max(mri_data)

# print(f"Pixel value range: ({min_value}, {max_value})")


# # 查看数据类型
# print(f"Pixel data type: {mri_data.dtype}")


# 加载 .nii.gz 文件
file_path = "/home/caoxinyu/Arbitrary-scale/data/PMC/3D/created-3D/3T_final/PD_3T/1.nii.gz"  # 替换为实际文件路径
nii_data = nib.load(file_path)

# 获取数据的形状
data_shape = nii_data.shape

# 打印维度大小
print(f"数据的维度大小为: {data_shape}") 
# IXI   (256, 256, 130) 轴向130
# PMC   (150, 256, 181) 轴向181
# BraTS (240, 240, 155) 轴向155