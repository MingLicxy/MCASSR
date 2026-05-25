import os
import pydicom
import matplotlib.pyplot as plt

def visualize_and_save_dcm(dcm_file, output_folder):
    """
    读取 DICOM 文件并将可视化结果保存到指定文件夹。

    Args:
        dcm_file (str): DICOM 文件的路径。
        output_folder (str): 保存可视化结果的文件夹路径。
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取 DICOM 文件
    ds = pydicom.dcmread(dcm_file)
    
    # 提取影像数据
    pixel_array = ds.pixel_array
    
    # 创建可视化图像
    plt.figure(figsize=(8, 8))
    plt.imshow(pixel_array, cmap='gray')
    plt.axis('off')
    
    # 设置保存路径
    output_file = os.path.join(output_folder, os.path.basename(dcm_file).replace('.dcm', '.png'))
    
    # 保存图像
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"保存成功: {output_file}")

# 示例：处理一个文件
# dcm_file = "/home/caoxinyu/Arbitrary-scale/data/fastMRI/fastMRI_brain_DICOM/100506510965/647.dcm"  # 替换为你的 DICOM 文件路径
output_folder = "/home/caoxinyu/Arbitrary-scale/liif-main/results"  # 替换为你的输出文件夹路径
# visualize_and_save_dcm(dcm_file, output_folder)

# 示例：处理文件夹中的所有 DICOM 文件
dcm_folder = "/home/caoxinyu/Arbitrary-scale/data/fastMRI/fastMRI_brain_DICOM/100506510965"  # 替换为你的 DICOM 文件夹路径
for file_name in os.listdir(dcm_folder):
    if file_name.endswith('.dcm'):
        visualize_and_save_dcm(os.path.join(dcm_folder, file_name), output_folder)

