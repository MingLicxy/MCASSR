import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgb

def frequency_band_analysis(input_path, output_path, num_bands=10, bar_width=0.05, bar_spacing=0.1):
    """
    对图像进行频率分段分析，统计每个频率段的能量占比，并绘制柱状图。
    Args:
        input_path (str): 输入图像路径
        output_path (str): 输出结果路径（包含柱状图保存路径）
        num_bands (int): 将频率划分为的区间数量
        bar_width (float): 控制柱状图条带的宽度
        bar_spacing (float): 控制柱状图条带之间的间隔
    """
    # 加载图像并转换为灰度
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot load image from {input_path}")

    # 计算傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    # 获取频谱的中心位置
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2

    # 计算每个像素的频率半径
    y_indices, x_indices = np.ogrid[:rows, :cols]
    distances = np.sqrt((y_indices - crow) ** 2 + (x_indices - ccol) ** 2)

    # 确定最大频率半径
    max_distance = np.max(distances)

    # 划分频率范围
    band_edges = np.linspace(0, max_distance, num_bands + 1)
    band_proportions = []

    # 统计每个频率段的能量占比
    total_energy = np.sum(magnitude_spectrum)
    for i in range(num_bands):
        band_mask = (distances >= band_edges[i]) & (distances < band_edges[i + 1])
        band_energy = np.sum(magnitude_spectrum[band_mask])
        band_proportions.append(band_energy / total_energy)

    # 创建渐变色
    colors = generate_gradient_colors(num_bands)

    # 绘制柱状图并自定义样式
    band_labels = [f"{band_edges[i]:.1f}~{band_edges[i + 1]:.1f}" for i in range(num_bands)]
    plt.figure(figsize=(10, 6))

    bar_edgecolor = 'black'  # 条带边框颜色
    bar_alpha = 0.8  # 条带透明度

    # 设置条带的位置，这里我们控制每个条带的x轴位置，确保条带之间有固定间隔
    x_positions = np.arange(num_bands) * (bar_width + bar_spacing)

    # 绘制柱状图
    bars = plt.bar(x_positions, band_proportions, 
                   tick_label=band_labels, 
                   color=colors, 
                   edgecolor=bar_edgecolor, 
                   alpha=bar_alpha, 
                   width=bar_width)  # 控制条带的宽度

    # 为每个条带添加高度标签
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                 f'{bar.get_height():.2%}', 
                 ha='center', va='bottom', fontsize=10)

    # 设置标题和标签
    plt.xlabel('Frequency Range (Distance from Center)')
    plt.ylabel('Energy Proportion')
    plt.title('Frequency Band Energy Distribution')
    plt.xticks(rotation=45)

    # 保存柱状图
    output_file = f"{output_path}/ITSRN_table_039_x4.png" # 设置文件名
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved frequency band analysis plot to {output_file}")

def generate_gradient_colors(num_colors):
    """
    生成一个渐变的颜色列表，使用matplotlib的LinearSegmentedColormap，支持RGB格式。
    Args:
        num_colors (int): 需要的颜色数量
    """
    # 定义RGB颜色（0到1范围）
    color_start = to_rgb((153 / 255, 200 / 255, 224 / 255))  # skyblue RGB -> (135, 206, 235)
    color_end = to_rgb((242 / 255, 164 / 255, 129 / 255))    # salmon RGB -> (250, 128, 114)

    # 创建一个从color_start到color_end的渐变色
    cmap = LinearSegmentedColormap.from_list("gradient", [color_start, color_end])
    
    # 获取num_colors个渐变颜色
    gradient_colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
    
    return gradient_colors

# 使用示例
input_image_path = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/DLS-NUC-100_others/bicubic/X4/SR_ITSRN/039.png"
output_directory = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/freq_table"
frequency_band_analysis(input_image_path, output_directory, num_bands=10, bar_width=0.5, bar_spacing=0)  # 控制条带宽度和间隔



