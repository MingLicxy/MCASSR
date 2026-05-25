import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# cv2.COLORMAP_JET — 经典的热图配色，从蓝色（冷色）到红色（暖色）
# cv2.COLORMAP_HOT — 从黑色到红色，再到黄色，适用于表示强度或温度
# cv2.COLORMAP_COOL — 从青色到紫色，提供冷色调的风格
# cv2.COLORMAP_WINTER — 从深绿到白色，使用冷色调
# cv2.COLORMAP_BONE — 类似于骨灰色调，浅色的灰度
# cv2.COLORMAP_RAINBOW — 彩虹色，从红色到紫色，适合表示不同的值区间
# cv2.COLORMAP_OCEAN — 深蓝色到浅蓝色，模拟海洋的颜色风格
# cv2.COLORMAP_SUMMER — 绿色到黄色，带有自然的生机感
# cv2.COLORMAP_SPRING — 红色到黄色，暖色调的渐变
# cv2.COLORMAP_PARULA — 明亮的蓝绿渐变，非常适合科学可视化
# cv2.COLORMAP_TWILIGHT — 从蓝色到紫色的渐变，色调柔和
# cv2.COLORMAP_TWILIGHT_SHIFTED — 类似于COLORMAP_TWILIGHT，但色调稍微偏移
# cv2.COLORMAP_MAGMA — 黑色到黄色的渐变，适用于强度或温度图
# cv2.COLORMAP_INFERNO — 类似Magma，但更多的红色调
# cv2.COLORMAP_PLASMA — 从紫色到黄色的渐变，提供较强的对比

# 可视化注意力图的函数
def visualize_attention_map(image_path, attention_map, save_path=None):
    """
    可视化注意力图，叠加到原始图像上。
    
    Args:
        image_path (str): 原始图像的路径。
        attention_map (torch.Tensor): 注意力权重 (H, W)。
        save_path (str, optional): 保存结果图像的路径。如果为 None，则直接显示。
    """
    # 读取原始图像
    original_image = Image.open(image_path).convert("RGB")
    original_image = np.array(original_image)
    
    # 归一化注意力图
    attention_map = attention_map.cpu().detach().numpy()  # 转为 NumPy 数组
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    
    # 调整注意力图大小以匹配原始图像
    attention_map_resized = attention_map
    #attention_map_resized = cv2.resize(attention_map, (original_image.shape[1], original_image.shape[0]))
    
    # 将注意力图映射为热图
    heatmap = cv2.applyColorMap((attention_map_resized * 255).astype(np.uint8), cv2.COLORMAP_PLASMA) # cv2.COLORMAP_COOL/cv2.COLORMAP_HOT/cv2.COLORMAP_JET
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
    
    # 叠加热图和原始图像（可调节权重）
    #overlayed_image = (0.5 * original_image + 0.5 * heatmap).astype(np.uint8)
    
    # 保存或显示结果
    if save_path:
        Image.fromarray(heatmap).save(save_path)
        print(f"叠加注意力图已保存到: {save_path}")
    else:
        plt.imshow(heatmap)
        plt.axis('off')
        plt.show()

# 示例使用
if __name__ == "__main__":
    # 假设我们有一个注意力权重张量
    #attention_map = torch.rand(3, 3)  # 例如，从模型中提取的注意力权重
    attention_map = torch.tensor([[0.1, 0.1, 0.2], #0.4, 0.7, 0.2
                                  [0.2, 1.0, 0.6],
                                  [0.3, 0.7, 0.2]])

    # 原始图像路径
    image_path = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/DLS-NUC-100/GT/092.png"

    # 保存路径（可选）
    save_path = "/home/caoxinyu/Arbitrary-scale/liif-main/demo/atten_visual/av2_6.png"

    # 可视化注意力图
    visualize_attention_map(image_path, attention_map, save_path=save_path)
