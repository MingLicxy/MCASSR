import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def analyze_and_save(input_path, output_path):
    # 创建输出路径
    os.makedirs(output_path, exist_ok=True)

    # 加载图像并转换为灰度
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot load image from {input_path}")

    # 计算傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # 分离低频和高频信息
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)

    # 低频保留区域
    low_freq_radius = 30
    cv2.circle(mask, (ccol, crow), low_freq_radius, 1, -1)

    low_freq = fshift * mask
    high_freq = fshift * (1 - mask)

    # 反变换得到低频和高频图像
    low_img = np.abs(np.fft.ifft2(np.fft.ifftshift(low_freq)))
    high_img = np.abs(np.fft.ifft2(np.fft.ifftshift(high_freq)))

    # 保存原图、频谱、低频和高频图像
    cv2.imwrite(os.path.join(output_path, "original.jpg"), image)
    cv2.imwrite(os.path.join(output_path, "magnitude_spectrum.jpg"), magnitude_spectrum.astype(np.uint8))
    cv2.imwrite(os.path.join(output_path, "low_frequency.jpg"), low_img.astype(np.uint8))
    cv2.imwrite(os.path.join(output_path, "high_frequency.jpg"), high_img.astype(np.uint8))

    # 保存频谱和分量图的可视化
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title('Magnitude Spectrum')
    plt.imshow(magnitude_spectrum, cmap='viridis') # cmap参数与可视化风格相关

    plt.subplot(2, 2, 3)
    plt.title('Low Frequency')
    plt.imshow(low_img, cmap='viridis')

    plt.subplot(2, 2, 4)
    plt.title('High Frequency')
    plt.imshow(high_img, cmap='viridis')

    plt.savefig(os.path.join(output_path, "analysis_visualization.png"))
    plt.close()

# 使用示例
input_image_path = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/DLS-NUC-100_others/GT/080.png"
output_directory = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/visual/freq"
analyze_and_save(input_image_path, output_directory)
