import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_frequency_analysis(input_path, output_path, plot_type='2D'):
    """
    对图像进行频域分析并输出二维柱状图或三维波浪图，增加不同颜色表示强度。
    Args:
        input_path (str): 输入图像路径
        output_path (str): 输出结果路径（包含图片的保存路径）
        plot_type (str): 可选 '2D' 或 '3D'
    """
    # 加载图像并转换为灰度
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot load image from {input_path}")

    # 计算傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # 获取频谱维度
    rows, cols = magnitude_spectrum.shape
    x = np.linspace(-cols // 2, cols // 2, cols)
    y = np.linspace(-rows // 2, rows // 2, rows)
    X, Y = np.meshgrid(x, y)

    # 创建可视化
    plt.figure(figsize=(10, 8))

    if plot_type == '2D':
        # 二维柱状图，增加颜色表示强度
        plt.imshow(magnitude_spectrum, cmap='viridis', extent=[-cols // 2, cols // 2, -rows // 2, rows // 2])
        cbar = plt.colorbar(label='Frequency Magnitude')
        cbar.ax.set_ylabel('Magnitude Intensity', rotation=270, labelpad=15)  # 修改颜色条样式
        plt.title('2D Frequency Analysis with Intensity')
        plt.xlabel('Frequency X')
        plt.ylabel('Frequency Y')

    elif plot_type == '3D':
        # 三维波浪图，增加颜色条
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, magnitude_spectrum, cmap='viridis', edgecolor='none')
        
        # 添加颜色条，指示不同频率的强度
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Magnitude Intensity')

        ax.set_title('3D Frequency Analysis')
        ax.set_xlabel('Frequency X')
        ax.set_ylabel('Frequency Y')
        ax.set_zlabel('Magnitude')

    else:
        raise ValueError("plot_type must be '2D' or '3D'")

    # 保存图像
    output_file = f"{output_path}/ITSRN_39_frequency_analysis_{plot_type}.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Saved {plot_type} plot to {output_file}")

# 使用示例
input_image_path = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/DLS-NUC-100_others/bicubic/X4/SR_ITSRN/039.png"
output_directory = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/freq_graph"
visualize_frequency_analysis(input_image_path, output_directory, plot_type='2D')  # 二维柱状图
visualize_frequency_analysis(input_image_path, output_directory, plot_type='3D')  # 三维波浪图




# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def visualize_high_frequency(input_path, output_path, plot_type='2D', emphasis_factor=2.0):
#     """
#     对图像进行频域分析，增强高频信息的可视化，并输出二维柱状图或三维波浪图
#     Args:
#         input_path (str): 输入图像路径
#         output_path (str): 输出结果路径（包含图片的保存路径）
#         plot_type (str): 可选 '2D' 或 '3D'
#         emphasis_factor (float): 高频放大因子，用于增强高频分量
#     """
#     # 加载图像并转换为灰度
#     image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise ValueError(f"Cannot load image from {input_path}")

#     # 计算傅里叶变换
#     f = np.fft.fft2(image)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = np.abs(fshift)

#     # 获取频谱的中心位置
#     rows, cols = magnitude_spectrum.shape
#     crow, ccol = rows // 2, cols // 2

#     # 增强高频：增加中心区域的衰减
#     high_frequency = magnitude_spectrum.copy()
#     radius = min(rows, cols) // 8  # 选择低频的半径
#     for i in range(rows):
#         for j in range(cols):
#             dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
#             if dist < radius:
#                 high_frequency[i, j] *= 0.1  # 衰减低频
#             else:
#                 high_frequency[i, j] **= emphasis_factor  # 放大高频

#     # 对结果取对数增强对比
#     high_frequency_log = 20 * np.log(high_frequency + 1)

#     # 创建坐标网格
#     x = np.linspace(-cols // 2, cols // 2, cols)
#     y = np.linspace(-rows // 2, rows // 2, rows)
#     X, Y = np.meshgrid(x, y)

#     # 创建可视化
#     plt.figure(figsize=(10, 8))

#     if plot_type == '2D':
#         # 二维柱状图
#         plt.imshow(high_frequency_log, cmap='viridis', extent=[-cols // 2, cols // 2, -rows // 2, rows // 2])
#         plt.colorbar(label='Enhanced High Frequency Magnitude')
#         plt.title('2D Enhanced High Frequency Analysis')
#         plt.xlabel('Frequency X')
#         plt.ylabel('Frequency Y')

#     elif plot_type == '3D':
#         # 三维波浪图
#         fig = plt.figure(figsize=(12, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.plot_surface(X, Y, high_frequency_log, cmap='viridis', edgecolor='none')
#         ax.set_title('3D Enhanced High Frequency Analysis')
#         ax.set_xlabel('Frequency X')
#         ax.set_ylabel('Frequency Y')
#         ax.set_zlabel('Enhanced Magnitude')

#     else:
#         raise ValueError("plot_type must be '2D' or '3D'")

#     # 保存图像
#     output_file = f"{output_path}/enhanced_frequency_{plot_type}.png"
#     plt.savefig(output_file)
#     plt.close()
#     print(f"Saved {plot_type} plot to {output_file}")

# # 使用示例
# input_image_path = "/home/caoxinyu/Arbitrary-scale/data/train_data/DIV2K/DIV2K_train_HR/0001.png"
# output_directory = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/freq_graph"
# visualize_high_frequency(input_image_path, output_directory, plot_type='2D', emphasis_factor=2.0)  # 二维柱状图
# visualize_high_frequency(input_image_path, output_directory, plot_type='3D', emphasis_factor=2.0)  # 三维波浪图
